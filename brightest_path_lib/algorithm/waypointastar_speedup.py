"""
Accurate Fast Waypoint Search with Nanometer Support and Improved Z-Range Detection
Optimized for speed while maintaining high path accuracy
Now includes intelligent z-positioning with proper distribution across Z-range
NO SUBDIVISION - Pure waypoint-to-waypoint processing
"""

import numpy as np
from typing import Tuple, List, Optional
import numba as nb
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import psutil
import math


@nb.njit(cache=True)
def calculate_segment_distance_accurate_nm(point_a_arr, point_b_arr, xy_spacing_nm):
    """Calculate 3D distance between points in nanometers"""
    # Convert pixel coordinates to nanometers
    z_diff_nm = (point_b_arr[0] - point_a_arr[0])
    y_diff_nm = (point_b_arr[1] - point_a_arr[1]) * xy_spacing_nm
    x_diff_nm = (point_b_arr[2] - point_a_arr[2]) * xy_spacing_nm
    
    distance_sq_nm = z_diff_nm * z_diff_nm + y_diff_nm * y_diff_nm + x_diff_nm * x_diff_nm
    return np.sqrt(distance_sq_nm)


@nb.njit(cache=True)
def calculate_segment_distance_accurate_pixels(point_a_arr, point_b_arr):
    """Calculate 3D distance between points in pixels (for internal use)"""
    distance_sq = 0.0
    for i in range(len(point_a_arr)):
        diff = point_b_arr[i] - point_a_arr[i]
        distance_sq += diff * diff
    return np.sqrt(distance_sq)

# Z-range detection functions ported from enhanced_waypointastar.py
@nb.njit(cache=True, parallel=True)
def find_intensity_transitions_at_point_optimized(image, y, x, intensity_threshold=0.3, z_click=-1):
    """
    Find z-frames where intensity transitions occur.
    If z_click is provided, finds the intensity interval closest to that z-plane.
    This handles overlapping dendrites by picking the relevant structure.
    """
    z_size = image.shape[0]
    intensities = np.zeros(z_size)
    
    # Extract intensity profile along z-axis
    for z in nb.prange(z_size):
        intensities[z] = image[z, y, x]
    
    # Find peak intensity locally or globally? 
    # Global max is safer to set threshold, even if we pick a smaller local peak later.
    max_intensity = np.max(intensities)
    threshold = max_intensity * intensity_threshold
    
    # Vectorized transition detection
    above_threshold = intensities >= threshold
    
    # If using z_click, find the specific interval containing or closest to z_click
    if z_click != -1:
        # Find all intervals
        intervals_start = []
        intervals_end = []
        
        in_interval = False
        current_start = -1
        last_valid_z = -1
        gap_tolerance = 2
        
        for z in range(z_size):
            if above_threshold[z]:
                if not in_interval:
                    # Check if this new start is close enough to merge with previous interval?
                    # Actually, simple state machine is better:
                    in_interval = True
                    current_start = z
                last_valid_z = z
            else:
                if in_interval:
                    # Check if we should close the interval
                    if z - last_valid_z > gap_tolerance:
                        in_interval = False
                        intervals_start.append(current_start)
                        intervals_end.append(last_valid_z) # End at the last valid pixel
        
        # Handle case where interval extends to the end or was kept open by tolerance
        if in_interval:
            intervals_start.append(current_start)
            intervals_end.append(last_valid_z)
            
        if len(intervals_start) > 0:
            # Find closest interval to z_click
            best_idx = 0
            min_dist = 100000
            
            for i in range(len(intervals_start)):
                s = intervals_start[i]
                e = intervals_end[i]
                
                # Distance logic:
                # If z_click is inside [s, e], dist is 0.
                # Else dist is min(|z_click - s|, |z_click - e|)
                if s <= z_click <= e:
                    dist = 0
                else:
                    dist = min(abs(z_click - s), abs(z_click - e))
                
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            return intervals_start[best_idx], intervals_end[best_idx], max_intensity
            
    # Fallback / Default: Find global start/end
    start_z = -1
    for z in range(z_size):
        if above_threshold[z]:
            start_z = z
            break
    
    end_z = -1
    for z in range(z_size - 1, -1, -1):
        if above_threshold[z]:
            end_z = z
            break
    
    # Handle edge cases
    if start_z == -1:
        start_z = 0
    if end_z == -1:
        end_z = z_size - 1
    
    return start_z, end_z, max_intensity


@nb.njit(cache=True, parallel=True)
def batch_find_optimal_z_with_intensity_priority(image, waypoint_coords, start_z, end_z, min_intensity_threshold=0.1):
    """
    Find optimal z-coordinates prioritizing intensity while maintaining reasonable distribution
    Uses a two-stage approach: first find bright regions, then distribute
    """
    num_waypoints = waypoint_coords.shape[0]
    optimal_z_values = np.zeros(num_waypoints, dtype=np.int32)
    
    if num_waypoints == 0:
        return optimal_z_values
    
    # Define the search range
    z_min_range = min(start_z, end_z)
    z_max_range = max(start_z, end_z)
    
    for i in nb.prange(num_waypoints):
        y, x = waypoint_coords[i, 0], waypoint_coords[i, 1]
        
        # First pass: find all positions above minimum intensity threshold
        candidate_positions = []
        candidate_intensities = []
        
        for z in range(max(0, z_min_range), min(image.shape[0], z_max_range + 1)):
            intensity = image[z, y, x]
            if intensity >= min_intensity_threshold:
                candidate_positions.append(z)
                candidate_intensities.append(intensity)
        
        if len(candidate_positions) == 0:
            # If no bright pixels found, find the best pixel in the range with distance penalty
            best_z = z_min_range
            best_score = -1000.0
            
            # Use target position for distance penalty
            if num_waypoints == 1:
                target_z = (start_z + end_z) // 2
            else:
                if end_z > start_z:
                    step = (end_z - start_z) / (num_waypoints + 1)
                    target_z = int(start_z + (i + 1) * step)
                elif start_z > end_z:
                    step = (start_z - end_z) / (num_waypoints + 1)
                    target_z = int(start_z - (i + 1) * step)
                else:
                    target_z = start_z

            for z in range(max(0, z_min_range), min(image.shape[0], z_max_range + 1)):
                intensity = image[z, y, x]
                # Heavy penalty to prevent jumping across the stack
                distance_penalty = abs(z - target_z) * 0.1
                score = intensity - distance_penalty
                if score > best_score:
                    best_score = score
                    best_z = z
            optimal_z_values[i] = best_z
        else:
            # Calculate target position for this waypoint
            if num_waypoints == 1:
                target_z = (start_z + end_z) // 2
            else:
                if end_z > start_z:
                    step = (end_z - start_z) / (num_waypoints + 1)
                    target_z = int(start_z + (i + 1) * step)
                elif start_z > end_z:
                    step = (start_z - end_z) / (num_waypoints + 1)
                    target_z = int(start_z - (i + 1) * step)
                else:
                    target_z = start_z
            
            # Find the best candidate considering both intensity and proximity to target
            best_score = -1.0
            best_z = candidate_positions[0]
            
            # Calculate max intensity for normalization
            max_candidate_intensity = max(candidate_intensities)
            
            for j in range(len(candidate_positions)):
                z_pos = candidate_positions[j]
                intensity = candidate_intensities[j]
                
                # Normalize intensity (0 to 1)
                normalized_intensity = intensity / max_candidate_intensity if max_candidate_intensity > 0 else 0
                
                # Calculate distance penalty (0 to 1, where 0 is no penalty)
                max_distance = abs(z_max_range - z_min_range)
                if max_distance > 0:
                    distance_penalty = abs(z_pos - target_z) / max_distance
                else:
                    distance_penalty = 0
                
                # Combined score: prioritize intensity but consider distribution
                # 70% intensity, 30% distribution
                score = 0.7 * normalized_intensity + 0.3 * (1.0 - distance_penalty)
                
                if score > best_score:
                    best_score = score
                    best_z = z_pos
            
            optimal_z_values[i] = best_z
    
    return optimal_z_values


@nb.njit(cache=True, parallel=True)
def batch_find_optimal_z_with_adaptive_search(image, waypoint_coords, start_z, end_z, min_intensity_threshold=0.05):
    """
    Adaptive Z-optimization that expands search range if no bright pixels found
    """
    num_waypoints = waypoint_coords.shape[0]
    optimal_z_values = np.zeros(num_waypoints, dtype=np.int32)
    
    if num_waypoints == 0:
        return optimal_z_values
    
    # Calculate initial target positions for distribution
    target_positions = np.zeros(num_waypoints, dtype=np.int32)
    
    if num_waypoints == 1:
        target_positions[0] = (start_z + end_z) // 2
    else:
        if end_z > start_z:
            step = (end_z - start_z) / (num_waypoints + 1)
            for i in range(num_waypoints):
                target_positions[i] = int(start_z + (i + 1) * step)
        elif start_z > end_z:
            step = (start_z - end_z) / (num_waypoints + 1)
            for i in range(num_waypoints):
                target_positions[i] = int(start_z - (i + 1) * step)
        else:
            target_positions.fill(start_z)
    
    for i in nb.prange(num_waypoints):
        y, x = waypoint_coords[i, 0], waypoint_coords[i, 1]
        target_z = target_positions[i]
        
        # Start with a small search range and expand if needed
        search_ranges = [3, 5, 8, 12]  # Progressively larger search ranges
        found_good_position = False
        
        for search_range in search_ranges:
            if found_good_position:
                break
                
            # Define search bounds
            z_min_search = max(0, target_z - search_range)
            z_max_search = min(image.shape[0] - 1, target_z + search_range)
            
            # Also respect the overall start_z to end_z range
            z_min_search = max(z_min_search, min(start_z, end_z) - search_range)
            z_max_search = min(z_max_search, max(start_z, end_z) + search_range)
            
            best_z = target_z
            best_score = -1.0
            
            # Find the best position in this range
            for z in range(z_min_search, z_max_search + 1):
                intensity = image[z, y, x]
                
                if intensity >= min_intensity_threshold:
                    # Calculate distance penalty
                    distance_penalty = abs(z - target_z) * 0.02  # Small penalty
                    
                    # Score = intensity - distance penalty
                    score = intensity - distance_penalty
                    
                    if score > best_score:
                        best_score = score
                        best_z = z
                        found_good_position = True
            
            optimal_z_values[i] = best_z
            
            # If we found a good position, stop expanding search range
            if found_good_position and image[best_z, y, x] >= min_intensity_threshold:
                break
        
        # If still no good position found, use full range but with heavy distance penalty
        if not found_good_position:
            z_min_full = max(0, min(start_z, end_z))
            z_max_full = min(image.shape[0] - 1, max(start_z, end_z))
            
            best_z = target_z
            best_score = -1000.0
            
            for z in range(z_min_full, z_max_full + 1):
                intensity = image[z, y, x]
                # Heavy distance penalty to keep it close to target interpolation
                # 0.1 means 10 slices away = -1.0 score penalty
                distance_penalty = abs(z - target_z) * 0.1
                score = intensity - distance_penalty
                
                if score > best_score:
                    best_score = score
                    best_z = z
            
            optimal_z_values[i] = best_z
    
    return optimal_z_values


@nb.njit(cache=True)
def filter_close_waypoints_optimized(waypoints, min_distance=3):
    """Remove waypoints that are too close to each other - OPTIMIZED"""
    if len(waypoints) <= 1:
        return waypoints
    
    # Always keep first waypoint
    filtered_indices = [0]
    
    for i in range(1, len(waypoints)):
        # Check distance to all previously kept waypoints
        keep_waypoint = True
        for j in filtered_indices:
            distance_sq = 0.0
            for k in range(waypoints.shape[1]):
                diff = waypoints[i, k] - waypoints[j, k]
                distance_sq += diff * diff
            
            if distance_sq < min_distance * min_distance:
                keep_waypoint = False
                break
        
        if keep_waypoint:
            filtered_indices.append(i)
    
    # Create filtered array
    filtered = np.zeros((len(filtered_indices), waypoints.shape[1]), dtype=waypoints.dtype)
    for i, idx in enumerate(filtered_indices):
        for j in range(waypoints.shape[1]):
            filtered[i, j] = waypoints[idx, j]
    
    return filtered


@nb.njit(cache=True)
def calculate_euclidean_distance_fast(point1, point2):
    """Fast euclidean distance calculation - OPTIMIZED"""
    distance_sq = 0.0
    for i in range(len(point1)):
        diff = point1[i] - point2[i]
        distance_sq += diff * diff
    return math.sqrt(distance_sq)


def find_start_end_points_optimized(points_list):
    """
    Find the two points with maximum distance as start and end points - OPTIMIZED
    Returns remaining points as waypoints.
    """
    if len(points_list) < 2:
        raise ValueError("Need at least 2 points to find start and end")
    
    # Convert to numpy array for faster computation
    points_array = np.array(points_list, dtype=np.float64)
    
    max_distance = 0.0
    start_idx = 0
    end_idx = 1
    
    # Find the two points with maximum distance
    for i in range(len(points_array)):
        for j in range(i + 1, len(points_array)):
            distance = calculate_euclidean_distance_fast(points_array[i], points_array[j])
            if distance > max_distance:
                max_distance = distance
                start_idx = i
                end_idx = j
    
    # Extract start and end points
    start_point = points_array[start_idx].copy()
    end_point = points_array[end_idx].copy()
    
    # Create waypoints array excluding start and end points
    waypoint_indices = []
    for i in range(len(points_array)):
        if i != start_idx and i != end_idx:
            waypoint_indices.append(i)
    
    if waypoint_indices:
        waypoints = points_array[waypoint_indices]
    else:
        waypoints = np.empty((0, points_array.shape[1]))
    
    return waypoints.tolist(), start_point, end_point


def should_use_hierarchical_search(image, num_waypoints):
    """Determine if hierarchical search should be used - OPTIMIZED"""
    image_size = np.prod(image.shape)
    complexity_factor = image_size * (num_waypoints + 1)
    
    # Use hierarchical search for very large images
    return complexity_factor > 100_000_000  # 100M operations threshold


class ZRangeCache:
    """Cache z-range calculations to avoid recomputation - OPTIMIZED"""
    def __init__(self):
        self.cache = {}
    
    def get_z_range(self, image, y, x, intensity_threshold, z_click=-1):
        # Include z_click in cache key if provided, as it affects the result
        key = (y, x, intensity_threshold, z_click)
        if key not in self.cache:
            self.cache[key] = find_intensity_transitions_at_point_optimized(image, y, x, intensity_threshold, z_click)
        return self.cache[key]


@dataclass
class SearchStrategy:
    """Search strategy that prioritizes accuracy"""
    use_hierarchical: bool
    hierarchical_factor: int
    weight_heuristic: float  # Always 1.0 for medical accuracy
    refine_path: bool  # Whether to refine path at full resolution
    suitable_for_parallel: bool  # Whether this segment can be processed in parallel


class Optimizer:
    """Optimizer that maintains accuracy while improving speed - now with nanometer support"""
    
    def __init__(self, image_shape, xy_spacing_nm=94.0, enable_parallel=True, 
                 max_parallel_workers=None, my_weight_heuristic=1.0):
        self.image_shape = image_shape
        self.image_volume = np.prod(image_shape)
        self.my_weight_heuristic = my_weight_heuristic
        
        # Store spacing in nanometers
        self.xy_spacing_nm = xy_spacing_nm
        
        # Conservative thresholds in nanometers (converted from pixels for compatibility)
        self.large_image_threshold = 30_000_000   # voxels
        self.huge_image_threshold = 100_000_000   # voxels
        
        # Parallel processing settings
        self.enable_parallel = enable_parallel
        if max_parallel_workers is None:
            # Auto-detect optimal number of workers
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Conservative worker count based on memory and cores
            max_workers_by_memory = max(1, int(available_memory_gb / 2))  # 2GB per worker
            max_workers_by_cpu = max(1, cpu_count - 1)  # Leave one core free
            
            self.max_parallel_workers = min(max_workers_by_memory, max_workers_by_cpu, 4)  # Cap at 4
        else:
            self.max_parallel_workers = max_parallel_workers
        
        print(f"Parallel processing: {self.enable_parallel}, Max workers: {self.max_parallel_workers}")
        print(f"Spacing: XY={self.xy_spacing_nm:.1f} nm/pixel")
    
    def determine_accurate_strategy(self, distance_nm: float, segment_idx: int, total_segments: int) -> SearchStrategy:
        """Determine strategy that maintains accuracy with parallel processing - now using nanometer thresholds"""
        
        # Default to high accuracy
        strategy = SearchStrategy(
            use_hierarchical=False,
            hierarchical_factor=4,
            weight_heuristic=self.my_weight_heuristic,  # ALWAYS 1.0 for medical accuracy
            refine_path=False,
            suitable_for_parallel=False
        )
        
        # Determine if suitable for parallel processing - using nanometer thresholds
        # Criteria: moderate to long segments in nanometers, not too complex
        if (self.enable_parallel and 
            distance_nm > 9400.0 and      # 9.4 μm (was 100 pixels × 94 nm/pixel)
            distance_nm < 56400.0 and     # 56.4 μm (was 600 pixels × 94 nm/pixel)
            total_segments > 2):          # Multiple segments available
            strategy.suitable_for_parallel = True
        
        # Only use hierarchical for very large images and long segments - nanometer thresholds
        if self.image_volume > self.huge_image_threshold and distance_nm > 28200.0:  # 28.2 μm (300 pixels × 94 nm)
            # Very conservative hierarchical search
            strategy.use_hierarchical = True
            strategy.hierarchical_factor = 4  # Small factor to preserve detail
            strategy.refine_path = True       # Always refine for accuracy
            strategy.weight_heuristic = self.my_weight_heuristic  # Always optimal
            
        elif self.image_volume > self.large_image_threshold and distance_nm > 37600.0:  # 37.6 μm (400 pixels × 94 nm)
            # Only for very long segments on large images
            strategy.use_hierarchical = True
            strategy.hierarchical_factor = 3  # Very conservative factor
            strategy.refine_path = True
            strategy.weight_heuristic = self.my_weight_heuristic 
        
        return strategy


class FasterWaypointSearch:
    """Fast waypoint search that maintains high accuracy with parallel processing and Z-range detection"""
    
    def __init__(self, image, points_list, xy_spacing_nm=94.0, 
                 enable_z_optimization=True, intensity_threshold=0.3, 
                 min_intensity_threshold=0.1, **kwargs):
        self.image = image
        self.points_list = points_list
        self.verbose = kwargs.get('verbose', True)
        self.my_weight_heuristic = kwargs.get('weight_heuristic', 1.0) 
        self.max_workers = kwargs.get('max_workers', None)

        # Store spacing
        self.xy_spacing_nm = xy_spacing_nm
        
        # Z-range detection settings
        self.enable_z_optimization = enable_z_optimization
        self.intensity_threshold = intensity_threshold
        self.min_intensity_threshold = min_intensity_threshold
        
        # Initialize z-range cache for performance
        self.z_range_cache = ZRangeCache()
        
        # Parallel processing settings
        enable_parallel = kwargs.get('enable_parallel', True)
        max_parallel_workers = kwargs.get('max_parallel_workers', None)
        
        # Initialize optimizer with parallel settings and spacing
        self.optimizer = Optimizer(
            image.shape, 
            xy_spacing_nm=xy_spacing_nm,
            enable_parallel=enable_parallel,
            max_parallel_workers=max_parallel_workers,
            my_weight_heuristic=self.my_weight_heuristic
        )
        
        # Configuration - conservative settings in nanometers
        self.enable_refinement = kwargs.get('enable_refinement', True)
        self.filter_close_waypoints = kwargs.get('filter_close_waypoints', True)
        self.min_waypoint_distance = kwargs.get('min_waypoint_distance', 3.0)
        
        if self.verbose:
            print(f"Initializing enhanced accurate fast search for image shape: {image.shape}")
            print(f"Image volume: {self.optimizer.image_volume:,} voxels")
            print(f"Parallel processing: {self.optimizer.enable_parallel}")
            print(f"Spacing: XY={self.xy_spacing_nm:.1f} nm/pixel")
            print(f"Z-range optimization: {self.enable_z_optimization}")
            print("NO SUBDIVISION - Pure waypoint-to-waypoint processing with Z-optimization")

    def _process_points_list_optimized(self, points_list):
        """Process a list of points to extract start, end, and waypoints with intelligent Z-positioning"""
        if len(points_list) < 2:
            raise ValueError("Need at least 2 points")
        
        if self.verbose:
            print(f"Processing {len(points_list)} input points...")
        
        # Find start and end points (farthest apart) - OPTIMIZED
        waypoint_coords, start_coords, end_coords = find_start_end_points_optimized(points_list)
        
        if self.verbose:
            print(f"Auto-detected start: {start_coords}, end: {end_coords}")
            print(f"Remaining waypoints: {len(waypoint_coords)}")
        
        # Apply z-range detection if enabled
        if self.enable_z_optimization and len(self.image.shape) == 3:
            # Find optimal z-ranges for start and end points using transition detection
            # Pass the user's Z-click to disambiguate overlapping dendrites
            start_z_click = int(start_coords[0])
            start_z_min, start_z_max, start_max_intensity = self.z_range_cache.get_z_range(
                self.image, int(start_coords[1]), int(start_coords[2]), self.intensity_threshold, start_z_click)
            
            end_z_click = int(end_coords[0])
            end_z_min, end_z_max, end_max_intensity = self.z_range_cache.get_z_range(
                self.image, int(end_coords[1]), int(end_coords[2]), self.intensity_threshold, end_z_click)
            
            if self.verbose:
                print(f"Start point transition frames: appears at {start_z_min}, disappears at {start_z_max} (max intensity: {start_max_intensity:.3f})")
                print(f"End point transition frames: appears at {end_z_min}, disappears at {end_z_max} (max intensity: {end_max_intensity:.3f})")
            
            # Update start and end points with transition-based z-coordinates
            if start_z_min >= 0:
                start_coords[0] = start_z_min  # Use appearance frame for start
                
            if end_z_max >= 0:
                end_coords[0] = end_z_max  # Use disappearance frame for end
            
            # Store the detected z-range for waypoint processing
            detected_z_min = int(start_coords[0])
            detected_z_max = int(end_coords[0])
            
            if self.verbose:
                print(f"Using transition-based coordinates - Start: {start_coords}, End: {end_coords}")
                print(f"Path z-range: {detected_z_min} to {detected_z_max}")
        else:
            detected_z_min = min(int(start_coords[0]), int(end_coords[0]))
            detected_z_max = max(int(start_coords[0]), int(end_coords[0]))
        
        # Process waypoints with intelligent z-positioning - OPTIMIZED with proper distribution
        processed_waypoints = []
        if waypoint_coords and self.enable_z_optimization:
            if self.verbose:
                print("Optimizing waypoint z-positions with distribution constraints...")
            
            # Convert waypoint coordinates to numpy array for batch processing
            waypoint_coords_array = np.array(waypoint_coords)
            
            # Extract Y,X coordinates for batch processing
            waypoint_yx = waypoint_coords_array[:, 1:3].astype(np.int32)
            
            # Use the adaptive search that prioritizes intensity
            optimal_z_values = batch_find_optimal_z_with_adaptive_search(
                self.image, waypoint_yx, detected_z_min, detected_z_max, 
                min_intensity_threshold=self.min_intensity_threshold)
            
            # Create processed waypoints
            for i, waypoint in enumerate(waypoint_coords):
                original_z = waypoint[0]
                optimized_z = optimal_z_values[i]
                processed_waypoint = np.array([optimized_z, waypoint[1], waypoint[2]], dtype=np.int32)
                processed_waypoints.append(processed_waypoint)
                
                if self.verbose:
                    intensity_at_optimized = self.image[optimized_z, int(waypoint[1]), int(waypoint[2])]
                    print(f"Waypoint {i+1}: {waypoint} -> {processed_waypoint} (intensity: {intensity_at_optimized:.3f})")
        else:
            # Use waypoints as-is
            processed_waypoints = [np.array(wp, dtype=np.int32) for wp in waypoint_coords]
        
        # Convert to numpy arrays
        start_point = np.array(start_coords, dtype=np.int32)
        goal_point = np.array(end_coords, dtype=np.int32)
        waypoints = np.array(processed_waypoints) if processed_waypoints else np.empty((0, 3), dtype=np.int32)
        
        # Filter close waypoints if enabled - OPTIMIZED
        if self.filter_close_waypoints and len(waypoints) > 1:
            original_count = len(waypoints)
            waypoints = filter_close_waypoints_optimized(waypoints, self.min_waypoint_distance)
            if self.verbose and len(waypoints) != original_count:
                print(f"Filtered waypoints from {original_count} to {len(waypoints)} (removed close duplicates)")
        
        return start_point, goal_point, waypoints
    
    def search_segment_accurate_wrapper(self, segment_data):
        """Wrapper for parallel processing"""
        point_a, point_b, segment_idx, strategy = segment_data
        return self.search_segment_accurate(point_a, point_b, segment_idx, strategy)
    
    def search_segments_parallel(self, parallel_segments):
        """Search multiple segments in parallel"""
        if not parallel_segments:
            return []
        
        if self.verbose:
            print(f"Processing {len(parallel_segments)} segments in parallel...")
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = self.max_workers
        if max_workers is None:
            max_workers = self.optimizer.max_parallel_workers
        print(f"Using up to {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parallel segments
            parallel_results = list(executor.map(self.search_segment_accurate_wrapper, parallel_segments))
        
        return parallel_results
    
    def search_segment_accurate(self, point_a, point_b, segment_idx, strategy: SearchStrategy):
        """Search segment with high accuracy"""
        from brightest_path_lib.algorithm.astar import BidirectionalAStarSearch
        from brightest_path_lib.input import CostFunction, HeuristicFunction
        
        # Calculate distance in nanometers for logging
        distance_nm = calculate_segment_distance_accurate_nm(
            np.array(point_a, dtype=np.float64), 
            np.array(point_b, dtype=np.float64),
            self.xy_spacing_nm,
        )
        
        if self.verbose:
            print(f"  Segment {segment_idx}: distance={distance_nm:.1f} nm, "
                  f"hierarchical={strategy.use_hierarchical}, "
                  f"refine={strategy.refine_path}")
        
        start_time = time.time()
        
        # First pass - potentially hierarchical for speed
        if strategy.use_hierarchical:
            # Create conservative hierarchical image
            hierarchical_image = self._create_conservative_hierarchical_image(strategy.hierarchical_factor)
            
            # Scale coordinates for hierarchical search
            scale_factor = strategy.hierarchical_factor
            start_scaled = np.array(point_a, dtype=np.int32) // scale_factor
            goal_scaled = np.array(point_b, dtype=np.int32) // scale_factor
            
            # Ensure scaled coordinates are valid
            start_scaled = np.clip(start_scaled, 0, np.array(hierarchical_image.shape) - 1)
            goal_scaled = np.clip(goal_scaled, 0, np.array(hierarchical_image.shape) - 1)
            
            # Hierarchical search
            hierarchical_search = BidirectionalAStarSearch(
                image=hierarchical_image,
                start_point=start_scaled,
                goal_point=goal_scaled,
                scale=(1.0, 1.0, 1.0),
                cost_function=CostFunction.RECIPROCAL,
                heuristic_function=HeuristicFunction.EUCLIDEAN,
                use_hierarchical=False,  # Don't nest hierarchical
                weight_heuristic=self.my_weight_heuristic    # Always optimal
            )
            
            hierarchical_path = hierarchical_search.search(verbose=False)
            
            if not hierarchical_path:
                print(f"    Hierarchical search failed, falling back to full resolution")
                strategy.use_hierarchical = False
                strategy.refine_path = False
        
        # Second pass - refine at full resolution if needed
        if strategy.refine_path and strategy.use_hierarchical and hierarchical_path:
            # Scale hierarchical path back to full resolution
            scaled_path = [point * strategy.hierarchical_factor for point in hierarchical_path]
            
            # Refine path by searching in a corridor around the hierarchical path
            refined_path = self._refine_path_in_corridor(scaled_path, point_a, point_b)
            segment_path = refined_path
        else:
            # Direct search at full resolution
            search = BidirectionalAStarSearch(
                image=self.image,
                start_point=np.array(point_a, dtype=np.int32),
                goal_point=np.array(point_b, dtype=np.int32),
                scale=(1.0, 1.0, 1.0),
                cost_function=CostFunction.RECIPROCAL,
                heuristic_function=HeuristicFunction.EUCLIDEAN,
                use_hierarchical=False,
                weight_heuristic=self.my_weight_heuristic  # Always optimal for accuracy
            )
            
            segment_path = search.search(verbose=False)
        
        search_time = time.time() - start_time
        
        if self.verbose:
            print(f"    Completed in {search_time:.2f}s, "
                  f"{len(segment_path) if segment_path else 0} points")
        
        return segment_path
    
    def _create_conservative_hierarchical_image(self, factor):
        """Create hierarchical image with conservative downsampling"""
        if not hasattr(self, '_hierarchical_cache'):
            self._hierarchical_cache = {}
        
        if factor not in self._hierarchical_cache:
            # Use maximum intensity to preserve bright structures
            z, y, x = self.image.shape
            new_z = max(1, z // factor)
            new_y = max(1, y // factor)
            new_x = max(1, x // factor)
            
            downsampled = np.zeros((new_z, new_y, new_x), dtype=self.image.dtype)
            
            for i in range(new_z):
                for j in range(new_y):
                    for k in range(new_x):
                        z_start, z_end = i * factor, min((i + 1) * factor, z)
                        y_start, y_end = j * factor, min((j + 1) * factor, y)
                        x_start, x_end = k * factor, min((k + 1) * factor, x)
                        
                        region = self.image[z_start:z_end, y_start:y_end, x_start:x_end]
                        # Use maximum to preserve bright structures
                        downsampled[i, j, k] = np.max(region)
            
            self._hierarchical_cache[factor] = downsampled
        
        return self._hierarchical_cache[factor]
    
    def _refine_path_in_corridor(self, coarse_path, original_start, original_goal):
        """Refine a coarse path by searching in a corridor around it"""
        from brightest_path_lib.algorithm.astar import BidirectionalAStarSearch
        from brightest_path_lib.input import CostFunction, HeuristicFunction
    
        # This ensures maximum accuracy
        search = BidirectionalAStarSearch(
            image=self.image,
            start_point=np.array(original_start, dtype=np.int32),
            goal_point=np.array(original_goal, dtype=np.int32),
            scale=(1.0, 1.0, 1.0),
            cost_function=CostFunction.RECIPROCAL,
            heuristic_function=HeuristicFunction.EUCLIDEAN,
            use_hierarchical=False,
            weight_heuristic=self.my_weight_heuristic  # Always optimal
        )
        
        return search.search(verbose=False)
    
    def search(self):
        """Perform accurate fast search with Z-range optimization and parallel processing"""
        start_time = time.time()
        
        # Apply Z-range optimization if enabled
        if self.enable_z_optimization:
            start_point, goal_point, waypoints = self._process_points_list_optimized(self.points_list)
            all_points = [start_point] + [waypoints[i] for i in range(len(waypoints))] + [goal_point]
        else:
            # NO SUBDIVISION - Use original waypoints directly (fallback)
            waypoint_coords, start_coords, end_coords = find_start_end_points_optimized(self.points_list)
            all_points = [start_coords] + waypoint_coords + [end_coords]
        
        if self.verbose:
            optimization_info = "with Z-range optimization" if self.enable_z_optimization else "without Z-optimization"
            print(f"Starting enhanced accurate fast search {optimization_info}")
            print(f"Original points: {len(self.points_list)} -> Optimized points: {len(all_points)}")
        
        # Analyze all segments and determine strategies
        segment_data = []
        strategies = []
        
        for i in range(len(all_points) - 1):
            point_a = all_points[i]
            point_b = all_points[i + 1]
            
            # Determine strategy for this segment using nanometer distance
            distance_nm = calculate_segment_distance_accurate_nm(
                np.array(point_a, dtype=np.float64), 
                np.array(point_b, dtype=np.float64),
                self.xy_spacing_nm,
            )
            
            strategy = self.optimizer.determine_accurate_strategy(
                distance_nm, i + 1, len(all_points) - 1)
            
            strategies.append(strategy)
            segment_data.append((point_a, point_b, i + 1, strategy))
        
        # Separate segments for parallel vs sequential processing
        parallel_segments = []
        sequential_segments = []
        parallel_indices = []
        sequential_indices = []
        
        for idx, (point_a, point_b, segment_idx, strategy) in enumerate(segment_data):
            if strategy.suitable_for_parallel and len(parallel_segments) < self.optimizer.max_parallel_workers:
                parallel_segments.append((point_a, point_b, segment_idx, strategy))
                parallel_indices.append(idx)
            else:
                sequential_segments.append((point_a, point_b, segment_idx, strategy))
                sequential_indices.append(idx)
        
        if self.verbose:
            print(f"Parallel segments: {len(parallel_segments)}, Sequential: {len(sequential_segments)}")
        
        # Initialize results array to maintain order
        all_paths = [None] * len(segment_data)
        
        # Process parallel segments
        if parallel_segments:
            parallel_start_time = time.time()
            parallel_results = self.search_segments_parallel(parallel_segments)
            parallel_time = time.time() - parallel_start_time
            
            # Store parallel results in correct positions
            for idx, result in zip(parallel_indices, parallel_results):
                all_paths[idx] = result
            
            if self.verbose:
                print(f"Parallel processing completed in {parallel_time:.2f}s")
        
        # Process sequential segments
        if sequential_segments:
            if self.verbose:
                print("Processing sequential segments...")
            
            for segment_idx, (point_a, point_b, seg_num, strategy) in enumerate(sequential_segments):
                path = self.search_segment_accurate(point_a, point_b, seg_num, strategy)
                
                # Store in correct position
                original_idx = sequential_indices[segment_idx]
                all_paths[original_idx] = path
        
        # Check if all segments succeeded
        if all(path is not None for path in all_paths):
            # Combine paths in correct order
            result = all_paths[0].copy()
            for path in all_paths[1:]:
                result.extend(path[1:])  # Skip first point to avoid duplication
        else:
            failed_segments = [i + 1 for i, path in enumerate(all_paths) if path is None]
            print(f"ERROR: Failed to find path for segments: {failed_segments}")
            return None
        
        total_time = time.time() - start_time
        
        if self.verbose:
            optimization_info = "with Z-range optimization" if self.enable_z_optimization else "without Z-optimization"
            print(f"Enhanced accurate fast search completed in {total_time:.2f}s {optimization_info}")
            print(f"Total path length: {len(result)}")
            
            # Calculate theoretical speedup from parallelization
            if parallel_segments:
                sequential_time_estimate = total_time + (len(parallel_segments) - 1) * (total_time / len(all_points))
                parallel_speedup = sequential_time_estimate / total_time
                print(f"Estimated parallel speedup: {parallel_speedup:.1f}x")
        
        return result


# Conservative optimization settings for medical use - now with nanometer thresholds and Z-optimization
def create_accurate_settings_nm(xy_spacing_nm=94.0, enable_z_optimization=True):
    """Create settings that prioritize accuracy for medical applications - using nanometer thresholds with Z-optimization"""
    return {
        'enable_refinement': True,             # Always refine hierarchical paths
        'hierarchical_threshold': 100_000_000, # Only for very large images
        'weight_heuristic': 1.0,               # ALWAYS optimal for medical accuracy
        'enable_parallel': True,               # Enable parallel processing
        'max_parallel_workers': None,          # Auto-detect optimal workers
        'xy_spacing_nm': xy_spacing_nm,        # XY pixel spacing 
        'enable_z_optimization': enable_z_optimization,  # Enable intelligent Z-positioning
        'intensity_threshold': 0.3,            # Threshold for Z-range detection
        'min_intensity_threshold': 0.1,        # Minimum intensity for waypoint placement
        'filter_close_waypoints': True,        # Filter waypoints that are too close
        'min_waypoint_distance': 3.0,          # Minimum distance between waypoints
    }


def quick_accurate_optimized_search(image, points_list, xy_spacing_nm=94.0,
                                   my_weight_heuristic=2.0, verbose=True, enable_parallel=True,
                                   enable_z_optimization=True, intensity_threshold=0.3,
                                   min_intensity_threshold=0.1, max_workers=None):
    """
    Quick accurate optimized search with nanometer support and Z-range optimization - NO SUBDIVISION
    
    Args:
        image: 3D image array
        points_list: List of [z, y, x] waypoints in pixel coordinates
        xy_spacing_nm: XY pixel spacing in nanometers per pixel
        my_weight_heuristic: A* weight heuristic (1.0 = optimal)
        verbose: Print progress information
        enable_parallel: Enable parallel processing
        enable_z_optimization: Enable intelligent Z-positioning based on intensity transitions
        intensity_threshold: Threshold for Z-range detection (fraction of peak intensity)
        min_intensity_threshold: Minimum intensity required for waypoint placement
    """
    
    # Use conservative settings with nanometer support and Z-optimization
    settings = create_accurate_settings_nm(xy_spacing_nm, enable_z_optimization)
    settings['weight_heuristic'] = my_weight_heuristic
    settings['enable_parallel'] = enable_parallel
    settings['intensity_threshold'] = intensity_threshold
    settings['min_intensity_threshold'] = min_intensity_threshold
    settings['max_workers'] = max_workers
    
    # Remove spacing from settings to avoid duplicate keyword arguments
    settings.pop('xy_spacing_nm', None)
    
    search = FasterWaypointSearch(
        image=image,
        points_list=points_list,
        xy_spacing_nm=xy_spacing_nm,
        verbose=verbose,
        **settings
    )
    
    return search.search()


# if __name__ == "__main__":
#     print('Enhanced fast waypoint search with Z-range optimization ready - NO SUBDIVISION!')
#     print('Features:')
#     print('  - Intelligent Z-positioning based on intensity transitions')
#     print('  - Automatic start/end point detection with appearance/disappearance frames')
#     print('  - Parallel processing for speed')
#     print('  - Nanometer-aware thresholds')
#     print('  - Waypoint filtering and optimization')
#     print('  - Medical-grade accuracy (weight_heuristic=1.0)')
#     print('  - Proper Z-distribution with intensity awareness')