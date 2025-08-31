"""Advanced optimized A* search implementation with waypoint support for finding brightest paths.
This extends the BidirectionalAStarSearch to support user-defined auxiliary waypoints with
performance optimizations matching the core algorithm.
"""

import heapq
import math
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any, Optional
import numba as nb
from numba import njit, prange, jit

from brightest_path_lib.algorithm.astar import (
    BidirectionalAStarSearch, array_equal, euclidean_distance_scaled,
    find_2D_neighbors_optimized, find_3D_neighbors_optimized
)
from brightest_path_lib.cost import Reciprocal
from brightest_path_lib.heuristic import Euclidean
from brightest_path_lib.image import ImageStats
from brightest_path_lib.input import CostFunction, HeuristicFunction
from brightest_path_lib.node import Node


# Numba-optimized function to check if all points are on the same z-plane
@nb.njit(cache=True)
def check_same_z_plane(start_point, goal_point, waypoints):
    """Check if all points are on the same z-plane for 3D optimization"""
    # If points are 2D, return True
    if len(start_point) == 2:
        return True
        
    # For 3D points, check z-coordinates
    if len(start_point) == 3:
        z_val = start_point[0]
        
        # Check if goal has same z
        if goal_point[0] != z_val:
            return False
            
        # Check all waypoints
        for i in range(len(waypoints)):
            if waypoints[i][0] != z_val:
                return False
                
        # All points have same z
        return True
        
    # For higher dimensions, don't use the optimization
    return False


# Numba-optimized function to create 2D points from 3D points
@nb.njit(cache=True)
def convert_3d_to_2d_points(start_point, goal_point, waypoints):
    """Convert 3D points to 2D by removing z-coordinate"""
    start_2d = np.array([start_point[1], start_point[2]], dtype=np.int32)
    goal_2d = np.array([goal_point[1], goal_point[2]], dtype=np.int32)
    
    # Convert waypoints
    waypoints_2d = np.empty((len(waypoints), 2), dtype=np.int32)
    for i in range(len(waypoints)):
        waypoints_2d[i, 0] = waypoints[i][1]
        waypoints_2d[i, 1] = waypoints[i][2]
        
    return start_2d, goal_2d, waypoints_2d


# Numba-optimized function to convert 2D path back to 3D
@nb.njit(cache=True)
def convert_2d_path_to_3d(path_2d, z_value):
    """Convert a 2D path back to 3D by adding z-coordinate"""
    path_3d = np.empty((len(path_2d), 3), dtype=np.int32)
    
    for i in range(len(path_2d)):
        path_3d[i, 0] = z_value
        path_3d[i, 1] = path_2d[i][0]
        path_3d[i, 2] = path_2d[i][1]
        
    return path_3d


class WaypointBidirectionalAStarSearch:
    """Advanced bidirectional A* search implementation with waypoint support
    
    This implementation allows users to specify intermediate points that
    the path should pass through, breaking down a complex search into
    multiple simpler searches between consecutive points.
    
    Performance optimization: When waypoints are on the same z-plane, 
    the algorithm automatically uses 2D pathfinding instead of 3D.
    """

    def __init__(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        waypoints: List[np.ndarray] = None,
        scale: Tuple = (1.0, 1.0),
        cost_function: CostFunction = CostFunction.RECIPROCAL,
        heuristic_function: HeuristicFunction = HeuristicFunction.EUCLIDEAN,
        open_nodes=None,
        use_hierarchical: bool = False,
        weight_heuristic: float = 1.0
    ):
        """Initialize waypoint-enabled A* search
        
        Parameters
        ----------
        image : numpy ndarray
            The image to search
        start_point, goal_point : numpy ndarray
            Start and goal coordinates
        waypoints : List[numpy ndarray], optional
            List of intermediate points that the path must pass through,
            in the order they should be visited
        scale : tuple
            Image scale factors
        cost_function, heuristic_function : Enum
            Functions to use for cost and heuristic
        open_nodes : Queue, optional
            Queue for visualization
        use_hierarchical : bool
            Whether to use hierarchical search for large images
        weight_heuristic : float
            Weight for heuristic (> 1.0 makes search faster but less optimal)
        """
        self._validate_inputs(image, start_point, goal_point, waypoints)
        
        # Basic parameters
        self.image = image
        self.image_stats = ImageStats(image)
        self.start_point = np.round(start_point).astype(np.int32)
        self.goal_point = np.round(goal_point).astype(np.int32)
        
        # Process and validate waypoints if provided
        if waypoints and len(waypoints) > 0:
            self.waypoints = np.array([np.round(wp).astype(np.int32) for wp in waypoints])
        else:
            # Empty array with correct shape for numba compatibility
            self.waypoints = np.empty((0, len(self.start_point)), dtype=np.int32)
        
        self.scale = scale
        self.open_nodes = open_nodes
        self.use_hierarchical = use_hierarchical
        self.weight_heuristic = weight_heuristic
        
        if cost_function == CostFunction.RECIPROCAL:
            self.cost_function = Reciprocal(
                min_intensity=self.image_stats.min_intensity, 
                max_intensity=self.image_stats.max_intensity)
        
        if heuristic_function == HeuristicFunction.EUCLIDEAN:
            self.heuristic_function = Euclidean(scale=self.scale)
        
        # State variables
        self.is_canceled = False
        self.found_path = False
        self.evaluated_nodes = 0
        self.result = []
        self.segment_results = []  # Store individual segment paths
        self.segment_evaluated_nodes = []  # Track nodes evaluated per segment
        
        # Determine if we should use 2D mode (all points have same z-coordinate)
        self.use_2d_mode = check_same_z_plane(
            self.start_point, self.goal_point, self.waypoints)
        
        # For 2D optimization when applicable
        self.z_value = self.start_point[0] if len(self.start_point) == 3 else None
        
    def _validate_inputs(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        waypoints: Optional[List[np.ndarray]] = None
    ):
        """Validate input parameters"""
        if image is None or start_point is None or goal_point is None:
            raise TypeError("Image, start_point, and goal_point cannot be None")
        if len(image) == 0 or len(start_point) == 0 or len(goal_point) == 0:
            raise ValueError("Image, start_point, and goal_point cannot be empty")
            
        # Validate waypoints if provided
        if waypoints:
            for i, wp in enumerate(waypoints):
                if wp is None:
                    raise TypeError(f"Waypoint {i} cannot be None")
                if len(wp) == 0:
                    raise ValueError(f"Waypoint {i} cannot be empty")
                if len(wp) != len(start_point):
                    raise ValueError(f"Waypoint {i} dimensions must match start_point dimensions")

    @property
    def found_path(self) -> bool:
        return self._found_path

    @found_path.setter
    def found_path(self, value: bool):
        if value is None:
            raise TypeError
        self._found_path = value

    @property
    def is_canceled(self) -> bool:
        return self._is_canceled

    @is_canceled.setter
    def is_canceled(self, value: bool):
        if value is None:
            raise TypeError
        self._is_canceled = value

    def search(self, verbose: bool = False) -> List[np.ndarray]:
        """Perform A* search with waypoints
        
        This method breaks down the search into multiple segments:
        start→waypoint₁, waypoint₁→waypoint₂, ..., waypointₙ→goal
        
        Performance optimization: Uses 2D mode when all points are on the same z-plane
        
        Returns
        -------
        List[np.ndarray]
            Complete path from start to goal through all waypoints
        """
        # Reset state
        self.result = []
        self.segment_results = []
        self.segment_evaluated_nodes = []
        self.evaluated_nodes = 0
        
        # Check if we can use 2D mode for performance optimization
        if self.use_2d_mode and len(self.start_point) == 3:
            if verbose:
                print("Using 2D mode optimization (all points on same z-plane)")
            return self._search_2d_mode(verbose)
        else:
            return self._search_normal_mode(verbose)
    
    def _search_2d_mode(self, verbose: bool = False) -> List[np.ndarray]:
        """Perform search in 2D mode for performance optimization
        
        This extracts a 2D slice from the 3D image when all points are on the same z-plane
        """
        # Extract 2D slice and convert points to 2D
        image_2d = self.image[self.z_value]
        start_2d, goal_2d, waypoints_2d = convert_3d_to_2d_points(
            self.start_point, self.goal_point, self.waypoints)
        
        # Create a list of all points in order, including start and goal
        wp_list = [waypoints_2d[i] for i in range(len(waypoints_2d))]
        all_points_2d = [start_2d] + wp_list + [goal_2d]
        
        # Track overall success
        overall_success = True
        
        if verbose:
            if len(wp_list) > 0:
                print(f"Starting 2D search with {len(wp_list)} waypoints")
            else:
                print("Starting 2D search (no waypoints)")
        
        # Process each segment (start to first waypoint, waypoint to waypoint, last waypoint to goal)
        for i in range(len(all_points_2d) - 1):
            if self.is_canceled:
                return []
                
            point_a = all_points_2d[i]
            point_b = all_points_2d[i+1]
            
            if verbose:
                if i == 0:
                    print(f"Searching from start to {'waypoint' if i+1 < len(all_points_2d)-1 else 'goal'} {i+1}")
                elif i+1 == len(all_points_2d)-1:
                    print(f"Searching from waypoint {i} to goal")
                else:
                    print(f"Searching from waypoint {i} to waypoint {i+1}")
            
            # Create A* search for this segment
            segment_search = BidirectionalAStarSearch(
                image=image_2d,
                start_point=point_a,
                goal_point=point_b,
                scale=self.scale[:2],  # Only use x,y scale
                cost_function=CostFunction.RECIPROCAL,
                heuristic_function=HeuristicFunction.EUCLIDEAN,
                open_nodes=self.open_nodes,
                use_hierarchical=self.use_hierarchical,
                weight_heuristic=self.weight_heuristic
            )
            
            # Run the search for this segment
            segment_path_2d = segment_search.search(verbose=verbose)
            
            # Track stats and results
            self.segment_evaluated_nodes.append(segment_search.evaluated_nodes)
            self.evaluated_nodes += segment_search.evaluated_nodes
            
            # Check if segment search was successful
            if segment_search.found_path and len(segment_path_2d) > 0:
                # Convert 2D path back to 3D
                segment_path_3d = convert_2d_path_to_3d(segment_path_2d, self.z_value)
                # Store as list of arrays for compatibility
                segment_path_list = [segment_path_3d[i] for i in range(len(segment_path_3d))]
                self.segment_results.append(segment_path_list)
                
                if verbose:
                    print(f"Found path for segment {i+1} ({len(segment_path_2d)} points, {segment_search.evaluated_nodes} nodes evaluated)")
            else:
                if verbose:
                    print(f"Failed to find path for segment {i+1}")
                overall_success = False
                break
        
        # If all segments were successful, combine the paths
        if overall_success:
            self._construct_complete_path()
            self.found_path = True
            
            if verbose:
                print(f"Complete path found: {len(self.result)} points, {self.evaluated_nodes} total nodes evaluated")
        else:
            if verbose:
                print("Failed to find a complete path through all waypoints")
        
        return self.result
    
    def _search_normal_mode(self, verbose: bool = False) -> List[np.ndarray]:
        """Perform search in normal mode (without special 2D optimization)"""
        # Create a list of all points in order, including start and goal
        wp_list = [self.waypoints[i] for i in range(len(self.waypoints))]
        all_points = [self.start_point] + wp_list + [self.goal_point]
        
        # Track overall success
        overall_success = True
        
        if verbose:
            if len(wp_list) > 0:
                print(f"Starting search with {len(wp_list)} waypoints")
            else:
                print("Starting search (no waypoints)")
        
        # Process each segment (start to first waypoint, waypoint to waypoint, last waypoint to goal)
        for i in range(len(all_points) - 1):
            if self.is_canceled:
                return []
                
            point_a = all_points[i]
            point_b = all_points[i+1]
            
            if verbose:
                if i == 0:
                    print(f"Searching from start to {'waypoint' if i+1 < len(all_points)-1 else 'goal'} {i+1}")
                elif i+1 == len(all_points)-1:
                    print(f"Searching from waypoint {i} to goal")
                else:
                    print(f"Searching from waypoint {i} to waypoint {i+1}")
            
            # Create A* search for this segment
            segment_search = BidirectionalAStarSearch(
                image=self.image,
                start_point=point_a,
                goal_point=point_b,
                scale=self.scale,
                cost_function=CostFunction.RECIPROCAL,
                heuristic_function=HeuristicFunction.EUCLIDEAN,
                open_nodes=self.open_nodes,
                use_hierarchical=self.use_hierarchical,
                weight_heuristic=self.weight_heuristic
            )
            
            # Run the search for this segment
            segment_path = segment_search.search(verbose=verbose)
            
            # Track stats and results
            self.segment_evaluated_nodes.append(segment_search.evaluated_nodes)
            self.evaluated_nodes += segment_search.evaluated_nodes
            
            # Check if segment search was successful
            if segment_search.found_path and len(segment_path) > 0:
                self.segment_results.append(segment_path)
                
                if verbose:
                    print(f"Found path for segment {i+1} ({len(segment_path)} points, {segment_search.evaluated_nodes} nodes evaluated)")
            else:
                if verbose:
                    print(f"Failed to find path for segment {i+1}")
                overall_success = False
                break
        
        # If all segments were successful, combine the paths
        if overall_success:
            self._construct_complete_path()
            self.found_path = True
            
            if verbose:
                print(f"Complete path found: {len(self.result)} points, {self.evaluated_nodes} total nodes evaluated")
        else:
            if verbose:
                print("Failed to find a complete path through all waypoints")
        
        return self.result
    
    def _construct_complete_path(self):
        """Combine segment paths into a complete path
        
        This handles removing duplicate points at segment boundaries.
        """
        if not self.segment_results:
            return
            
        # Start with the first segment
        self.result = self.segment_results[0].copy()
        
        # Add each subsequent segment (skipping the first point to avoid duplication)
        for i in range(1, len(self.segment_results)):
            segment = self.segment_results[i]
            
            # Skip the first point of each subsequent segment as it should 
            # be the same as the last point of the previous segment
            # But verify they're actually the same
            if len(segment) > 1 and np.array_equal(self.result[-1], segment[0]):
                self.result.extend(segment[1:])
            else:
                # Something's wrong - just append everything
                self.result.extend(segment)
    
    def get_segment_info(self):
        """Get information about each segment of the path
        
        Returns
        -------
        dict
            Dictionary with information about each path segment
        """
        if not self.segment_results:
            return None
            
        info = {
            'num_segments': len(self.segment_results),
            'segment_lengths': [len(path) for path in self.segment_results],
            'segment_evaluated_nodes': self.segment_evaluated_nodes,
            'total_path_length': len(self.result),
            'total_evaluated_nodes': self.evaluated_nodes,
            'used_2d_optimization': self.use_2d_mode and len(self.start_point) == 3
        }
        return info