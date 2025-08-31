import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFrame, QListWidget, QListWidgetItem,
    QProgressBar, QSpinBox, QGroupBox, QCheckBox, QDoubleSpinBox
)
from qtpy.QtCore import Signal
import sys
sys.path.append('../path_tracing/brightest-path-lib')
# Import the tube data generation
from brightest_path_lib.visualization.tube_data import create_tube_data  # Now uses minimal version
from skimage.feature import blob_log
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt, label, binary_erosion, gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
import skimage as ski


def detect_spines_with_angles(tube_data, frame_index, 
                             min_sigma_2d=5, max_sigma_2d=10, threshold_2d=0.05,
                             min_sigma_tube=5, max_sigma_tube=5, threshold_tube=0.02,
                             max_distance_threshold=15, angle_threshold=20, 
                             angle_weight=0.7):
    """
    Detect spines using combined angle and distance-to-center matching.
    Updated to work with minimal tube data structure.
    """
    # Extract data for the specified frame
    frame_data = tube_data[frame_index]
    center = frame_data['position']
    actual_z = int(center[0])  # Store the actual Z coordinate
    
    # Get dendrite direction from basis vectors (only forward is stored now)
    dendrite_direction = frame_data['basis_vectors']['forward']
    
    # === 2D VIEW PROCESSING ===
    view_2d = frame_data['zoom_patch']
    mask_2d = frame_data['zoom_patch_ref']
    footprint = ski.morphology.disk(1)
    mask_2d = ski.morphology.dilation(mask_2d, footprint)
    
    if mask_2d is None:
        subtracted_plane_2d = view_2d
    else:
        subtracted_plane_2d = view_2d - mask_2d
        subtracted_plane_2d[subtracted_plane_2d < 0] = 0
    
    blobs_2d = blob_log(subtracted_plane_2d, 
                        min_sigma=min_sigma_2d, 
                        max_sigma=max_sigma_2d, 
                        threshold=threshold_2d,
                        exclude_border=True)
    
    # === TUBULAR VIEW PROCESSING ===
    normal_plane = np.rot90(frame_data['normal_plane'])
    colored_plane = frame_data['colored_plane']
    # footprint = ski.morphology.disk(1)
    # colored_plane = ski.morphology.dilation(colored_plane, footprint)
    
    if colored_plane is not None:
        colored_plane = np.rot90(colored_plane)
        if colored_plane.ndim == 3:
            colored_plane = np.mean(colored_plane, axis=-1)
        subtracted_plane_tube = normal_plane - colored_plane
        subtracted_plane_tube[subtracted_plane_tube < 0] = 0
    else:
        subtracted_plane_tube = normal_plane
    
    blobs_tube = blob_log(subtracted_plane_tube,
                         min_sigma=min_sigma_tube,
                         max_sigma=max_sigma_tube,
                         threshold=threshold_tube)
    
    filtered_blobs_2d = blobs_2d
    filtered_blobs_tube = blobs_tube
    
    # === ANGLE-BASED MATCHING ===
    confirmed_spines = []
    view_2d_angles = []
    tube_angles = []
    
    if len(blobs_2d) > 0 and len(blobs_tube) > 0:
        # Get center of tubular view (dendrite center)
        tube_center = np.array([subtracted_plane_tube.shape[0]//2, 
                               subtracted_plane_tube.shape[1]//2])
        
        # Get center of 2D view
        view_2d_center = np.array([subtracted_plane_2d.shape[0]//2, 
                                  subtracted_plane_2d.shape[1]//2])
        
        # No filtering - use all blobs as in original code
        
        # Calculate angles for tubular blobs
        tube_angles = []
        for blob in filtered_blobs_tube:
            spine_vector = np.array([blob[0], blob[1]]) - tube_center
            angle = np.degrees(np.arctan2(spine_vector[0], spine_vector[1]))
            tube_angles.append(angle)
        
        # Calculate angles for 2D blobs
        view_2d_angles = []
        dendrite_2d = np.array([dendrite_direction[1], dendrite_direction[2]])
        dendrite_2d = dendrite_2d / (np.linalg.norm(dendrite_2d) + 1e-8)
        
        for blob in filtered_blobs_2d:
            spine_vector = np.array([blob[0], blob[1]]) - view_2d_center
            spine_vector = spine_vector / (np.linalg.norm(spine_vector) + 1e-8)
            
            dot_product = np.dot(spine_vector, dendrite_2d)
            cross_product = np.cross(spine_vector, dendrite_2d)
            angle = np.degrees(np.arctan2(cross_product, dot_product))
            view_2d_angles.append(angle)
        
        # Match based on combined angle and distance
        used_tube_indices = set()
        
        for i, (blob_2d, angle_2d) in enumerate(zip(filtered_blobs_2d, view_2d_angles)):
            best_match_idx = -1
            best_score = float('inf')
            
            spine_2d_dist = np.linalg.norm(np.array([blob_2d[0], blob_2d[1]]) - view_2d_center)
            
            for j, (blob_tube, angle_tube) in enumerate(zip(blobs_tube, tube_angles)):
                if j in used_tube_indices:
                    continue
                
                angle_diff = min(abs(angle_2d - angle_tube), abs(angle_2d + angle_tube))
                
                if angle_diff > angle_threshold:
                    continue
                
                spine_tube_dist = np.linalg.norm(np.array([blob_tube[0], blob_tube[1]]) - tube_center)
                distance_diff = abs(spine_2d_dist - spine_tube_dist)
                
                angle_score = angle_diff / angle_threshold
                max_dist_diff = max(spine_2d_dist, spine_tube_dist)
                distance_score = distance_diff / max_dist_diff if max_dist_diff > 0 else 0
                
                combined_score = angle_weight * angle_score + (1 - angle_weight) * distance_score
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_match_idx = j
            
            if best_match_idx >= 0:
                used_tube_indices.add(best_match_idx)
                corresponding_tube_blob = blobs_tube[best_match_idx]
                
                confirmed_spines.append({
                    'blob_2d': blob_2d,
                    'blob_tube': corresponding_tube_blob,
                    'coords_2d': (blob_2d[1], blob_2d[0]),
                    'coords_tube': (corresponding_tube_blob[1], corresponding_tube_blob[0]),
                    'actual_z': actual_z
                })
    
    return {
        'frame_index': frame_index,
        'actual_z': actual_z,
        'blobs_2d': blobs_2d,
        'blobs_tube': blobs_tube,
        'confirmed_spines': confirmed_spines,
        'num_confirmed_spines': len(confirmed_spines),
        'view_2d_angles': view_2d_angles,
        'tube_angles': tube_angles
    }


class SpineTracker:
    """Smart spine tracker that uses intensity analysis and watershed segmentation"""
    
    def __init__(self, min_intensity_ratio=0.3, min_distance_separation=8):
        self.min_intensity_ratio = min_intensity_ratio  # Minimum intensity ratio to background
        self.min_distance_separation = min_distance_separation  # Minimum distance between spine centers
        
    def segment_spine_areas_with_watershed(self, image, spine_positions, window_size=20):
        """
        Use watershed segmentation to find spine areas around detected points
        
        Args:
            image: 3D image volume
            spine_positions: List of spine center positions [z, y, x]
            window_size: Size of local window for watershed analysis
            
        Returns:
            List of spine areas with segmentation masks
        """
        spine_areas = []
        
        print(f"Segmenting spine areas using watershed for {len(spine_positions)} positions...")
        
        for i, spine_pos in enumerate(spine_positions):
            z, y, x = int(spine_pos[0]), int(spine_pos[1]), int(spine_pos[2])
            
            # Check bounds
            if (z < 0 or z >= image.shape[0] or 
                y < window_size or y >= image.shape[1] - window_size or
                x < window_size or x >= image.shape[2] - window_size):
                continue
            
            # Extract local region around spine
            y_min, y_max = y - window_size, y + window_size
            x_min, x_max = x - window_size, x + window_size
            
            local_region = image[z, y_min:y_max, x_min:x_max].astype(np.float32)
            
            # Apply slight smoothing to reduce noise
            smoothed = gaussian_filter(local_region, sigma=0.8)
            
            # Create binary mask using Otsu thresholding
            try:
                threshold = threshold_otsu(smoothed)
                binary_mask = smoothed > threshold
            except:
                # Fallback if Otsu fails
                threshold = np.percentile(smoothed, 75)
                binary_mask = smoothed > threshold
            
            # Remove small objects
            binary_mask = remove_small_objects(binary_mask, min_size=5)
            
            if not np.any(binary_mask):
                continue
            
            # Create distance transform for watershed
            distance = distance_transform_edt(binary_mask)
            
            # Find local maxima as markers
            local_maxima = peak_local_max(distance, min_distance=5, threshold_abs=0.3 * np.max(distance))
            
            if len(local_maxima) == 0:
                continue
            
            # Create markers for watershed
            markers = np.zeros_like(distance, dtype=int)
            for j, (max_y, max_x) in enumerate(local_maxima):
                if 0 <= max_y < markers.shape[0] and 0 <= max_x < markers.shape[1]:
                    markers[max_y, max_x] = j + 1
            
            # Apply watershed
            labels = watershed(-distance, markers, mask=binary_mask)
            
            # Find the label that contains our spine center
            center_y, center_x = window_size, window_size  # Spine center in local coordinates
            spine_label = labels[center_y, center_x]
            
            if spine_label > 0:
                # Extract the spine area
                spine_mask = (labels == spine_label)
                
                # Calculate spine properties
                spine_area = np.sum(spine_mask)
                
                # Find spine boundary
                eroded = binary_erosion(spine_mask)
                boundary = spine_mask & ~eroded
                
                # Calculate spine metrics
                centroid_y, centroid_x = np.mean(np.where(spine_mask), axis=1)
                centroid_global_y = y_min + centroid_y
                centroid_global_x = x_min + centroid_x
                
                spine_info = {
                    'original_position': spine_pos,
                    'centroid': np.array([z, centroid_global_y, centroid_global_x]),
                    'area_pixels': spine_area,
                    'local_mask': spine_mask,
                    'boundary_mask': boundary,
                    'local_region_coords': (y_min, y_max, x_min, x_max),
                    'max_intensity': np.max(smoothed[spine_mask]),
                    'mean_intensity': np.mean(smoothed[spine_mask]),
                    'background_intensity': np.mean(smoothed[~binary_mask]) if np.any(~binary_mask) else 0
                }
                
                spine_areas.append(spine_info)
                
                if i < 5:  # Print details for first few spines
                    print(f"Spine {i}: area={spine_area} pixels, max_intensity={spine_info['max_intensity']:.2f}")
        
        print(f"Successfully segmented {len(spine_areas)} spine areas using watershed")
        return spine_areas
        
    def analyze_spine_intensity_across_frames(self, image, spine_position, frame_range=3):
        """
        Analyze spine intensity across multiple frames to determine visibility
        
        Args:
            image: 3D image volume
            spine_position: [z, y, x] coordinates of spine
            frame_range: Number of frames to check before/after
            
        Returns:
            dict: Frame analysis results
        """
        z, y, x = int(spine_position[0]), int(spine_position[1]), int(spine_position[2])
        results = {}
        
        # Check frames around the spine position
        for z_offset in range(-frame_range, frame_range + 1):
            target_z = z + z_offset
            
            # Check bounds
            if target_z < 0 or target_z >= image.shape[0]:
                continue
                
            if y < 0 or y >= image.shape[1] or x < 0 or x >= image.shape[2]:
                continue
            
            # Analyze intensity in this frame
            window_size = 4  # Slightly larger window for better analysis
            y_min = max(0, y - window_size)
            y_max = min(image.shape[1], y + window_size + 1)
            x_min = max(0, x - window_size)
            x_max = min(image.shape[2], x + window_size + 1)
            
            # Get spine region
            spine_region = image[target_z, y_min:y_max, x_min:x_max]
            if spine_region.size == 0:
                continue
            
            # Get spine center intensity
            center_y = min(window_size, y - y_min)
            center_x = min(window_size, x - x_min)
            spine_intensity = spine_region[center_y, center_x]
            
            # Get local maximum in spine region
            local_max = np.max(spine_region)
            
            # Get background intensity (larger region)
            bg_size = 12
            bg_y_min = max(0, y - bg_size)
            bg_y_max = min(image.shape[1], y + bg_size + 1)
            bg_x_min = max(0, x - bg_size)
            bg_x_max = min(image.shape[2], x + bg_size + 1)
            
            bg_region = image[target_z, bg_y_min:bg_y_max, bg_x_min:bg_x_max]
            if bg_region.size == 0:
                continue
                
            # Use 25th percentile as background (more robust than median)
            background_intensity = np.percentile(bg_region, 25)
            
            # Calculate metrics
            intensity_ratio = spine_intensity / (background_intensity + 1e-6)
            local_prominence = spine_intensity / (local_max + 1e-6)
            
            results[target_z] = {
                'intensity_ratio': intensity_ratio,
                'local_prominence': local_prominence,
                'spine_intensity': spine_intensity,
                'background_intensity': background_intensity,
                'is_visible': intensity_ratio > self.min_intensity_ratio and local_prominence > 0.3
            }
        
        return results
    
    def create_spine_tracks_with_areas(self, initial_spines, image, frame_range=3):
        """
        Create spine tracks with watershed-based area segmentation
        
        Args:
            initial_spines: List of initial spine detections
            image: 3D image volume
            frame_range: Number of frames to extend analysis
            
        Returns:
            Tuple of (spine_positions, spine_areas)
        """
        print(f"Creating smart spine tracks with watershed areas for {len(initial_spines)} initial detections...")
        
        # Group spines by spatial proximity (same physical spine across frames)
        spine_groups = self._group_spines_by_proximity(initial_spines)
        print(f"Grouped {len(initial_spines)} detections into {len(spine_groups)} unique spines")
        
        final_spine_positions = []
        all_spine_areas = []
        
        for group_id, spine_group in enumerate(spine_groups):
            # Find the best representative position for this spine group
            best_spine = self._find_best_spine_in_group(spine_group, image)
            
            # Analyze intensity across frames for this spine
            intensity_analysis = self.analyze_spine_intensity_across_frames(
                image, best_spine['position'], frame_range
            )
            
            # Add spine positions only in frames where it's visible
            visible_frames = [z for z, analysis in intensity_analysis.items() if analysis['is_visible']]
            
            if visible_frames:
                print(f"Spine {group_id}: visible in frames {visible_frames}")
                
                # Create spine positions for visible frames
                frame_positions = []
                for frame_z in visible_frames:
                    spine_pos = np.array([
                        float(frame_z),
                        float(best_spine['position'][1]),
                        float(best_spine['position'][2])
                    ])
                    frame_positions.append(spine_pos)
                    final_spine_positions.append(spine_pos)
                
                # Segment spine areas using watershed for this group
                if frame_positions:
                    spine_areas = self.segment_spine_areas_with_watershed(image, frame_positions)
                    all_spine_areas.extend(spine_areas)
        
        print(f"Final spine tracking: {len(final_spine_positions)} spine positions with {len(all_spine_areas)} segmented areas")
        return final_spine_positions, all_spine_areas
    
    def create_spine_tracks(self, initial_spines, image, frame_range=3):
        """
        Create spine tracks by analyzing intensity across frames and removing duplicates
        
        Args:
            initial_spines: List of initial spine detections
            image: 3D image volume
            frame_range: Number of frames to extend analysis
            
        Returns:
            List of spine positions with frame-specific visibility
        """
        positions, areas = self.create_spine_tracks_with_areas(initial_spines, image, frame_range)
        return positions
    
    def _group_spines_by_proximity(self, spines, max_distance=10):
        """Group spines that are likely the same physical spine"""
        if not spines:
            return []
        
        spine_positions = np.array([spine['position'] for spine in spines])
        
        # Calculate distances in Y-X plane (ignore Z for grouping)
        yx_positions = spine_positions[:, 1:3]  # Only Y and X coordinates
        
        groups = []
        used_indices = set()
        
        for i, spine in enumerate(spines):
            if i in used_indices:
                continue
                
            # Start a new group with this spine
            current_group = [spine]
            used_indices.add(i)
            
            # Find all other spines within max_distance in Y-X plane
            for j, other_spine in enumerate(spines):
                if j in used_indices:
                    continue
                    
                # Calculate Y-X distance
                yx_distance = np.linalg.norm(yx_positions[i] - yx_positions[j])
                
                if yx_distance <= max_distance:
                    current_group.append(other_spine)
                    used_indices.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def _find_best_spine_in_group(self, spine_group, image):
        """Find the best representative spine in a group (highest intensity)"""
        best_spine = None
        best_intensity = 0
        
        for spine in spine_group:
            z, y, x = int(spine['position'][0]), int(spine['position'][1]), int(spine['position'][2])
            
            # Check bounds
            if (0 <= z < image.shape[0] and 
                0 <= y < image.shape[1] and 
                0 <= x < image.shape[2]):
                
                intensity = image[z, y, x]
                if intensity > best_intensity:
                    best_intensity = intensity
                    best_spine = spine
        
        return best_spine if best_spine is not None else spine_group[0]


def process_all_frames_with_smart_tracking(tube_data, image, brightest_path, max_distance_threshold=15, 
                                          frame_range=2, manual_spine_points=None, progress_callback=None):
    """
    Process frames with smart spine tracking that avoids duplicates and uses intensity analysis
    
    This function:
    1. Detects spines using tube data at specific frames along the path
    2. Groups spines that are likely the same physical spine
    3. Analyzes intensity across frames to determine where each spine is visible
    4. Returns final spine positions with smart frame assignment
    """
    # Fixed parameters (matching original code)
    detection_params = {
        'min_sigma_2d': 4, 'max_sigma_2d': 12, 'threshold_2d': 0.04,
        'min_sigma_tube': 4, 'max_sigma_tube': 10, 'threshold_tube': 0.025,
        'angle_threshold': 25, 'angle_weight': 0.5
    }
    
    initial_spine_positions = []
    all_results = []
    
    total_frames = len(tube_data)
    
    # Process tube data frames to detect spines
    print("Detecting spines from tube data...")
    for frame_idx in range(total_frames):
        if progress_callback:
            progress = int((frame_idx / total_frames) * 40)
            progress_callback(progress, 100)
        
        results = detect_spines_with_angles(
            tube_data, frame_idx,
            max_distance_threshold=max_distance_threshold,
            **detection_params
        )
        
        all_results.append(results)
        
        if results['confirmed_spines']:
            frame_position = tube_data[frame_idx]['position']
            z = results['actual_z']  # Use the actual Z from results
            y_base = frame_position[1]
            x_base = frame_position[2]
            frame_shape_x, frame_shape_y = tube_data[frame_idx]['zoom_patch'].shape
            
            for spine in results['confirmed_spines']:
                coord_x, coord_y = spine['coords_2d']
                # Calculate absolute position in the full image
                spine_y = y_base - int(frame_shape_y/2) + coord_y
                spine_x = x_base - int(frame_shape_x/2) + coord_x
                
                spine_3d_position = {
                    'z': z,
                    'y': spine_y,
                    'x': spine_x,
                    'position': np.array([float(z), float(spine_y), float(spine_x)])
                }
                initial_spine_positions.append(spine_3d_position)
    
    initial_spine_count = len(initial_spine_positions)
    print(f"Initial detection: {initial_spine_count} spines found")
    
    # Add manually clicked points to the initial spine positions
    manual_count = 0
    if manual_spine_points is not None and len(manual_spine_points) > 0:
        print(f"Adding {len(manual_spine_points)} manually clicked spine points...")
        for manual_point in manual_spine_points:
            # Convert manual point to the same format as detected spines
            spine_3d_position = {
                'z': int(manual_point[0]),
                'y': int(manual_point[1]),
                'x': int(manual_point[2]),
                'position': np.array([float(manual_point[0]), float(manual_point[1]), float(manual_point[2])])
            }
            initial_spine_positions.append(spine_3d_position)
            manual_count += 1
        print(f"Added {manual_count} manual spine points")
    
    if progress_callback:
        progress_callback(50, 100)
    
    # Use smart spine tracking with watershed areas
    if len(initial_spine_positions) > 0:
        print("Applying smart spine tracking with watershed area segmentation...")
        
        # Initialize spine tracker
        spine_tracker = SpineTracker(
            min_intensity_ratio=0.3,  # Minimum intensity ratio to background
            min_distance_separation=8  # Minimum distance between spine centers
        )
        
        # Create smart tracks with watershed areas
        final_positions, spine_areas = spine_tracker.create_spine_tracks_with_areas(
            initial_spine_positions, image, frame_range
        )
        
        # Convert to numpy array
        if final_positions:
            final_positions = np.array(final_positions)
        else:
            final_positions = np.empty((0, 3))
            
        # Store spine area information
        spine_area_info = {
            'areas': spine_areas,
            'total_spine_areas': len(spine_areas),
            'avg_area_pixels': np.mean([area['area_pixels'] for area in spine_areas]) if spine_areas else 0,
            'total_area_pixels': sum([area['area_pixels'] for area in spine_areas])
        }
    else:
        final_positions = np.empty((0, 3))
        spine_area_info = {'areas': [], 'total_spine_areas': 0, 'avg_area_pixels': 0, 'total_area_pixels': 0}
    
    if progress_callback:
        progress_callback(100, 100)
    
    # Calculate final statistics
    tracked_count = len(final_positions) - initial_spine_count if len(final_positions) > initial_spine_count else 0
    
    print(f"Smart tracking with watershed results:")
    print(f"  Initial detections: {initial_spine_count}")
    print(f"  Manual points: {manual_count}")
    print(f"  Final spine positions: {len(final_positions)}")
    print(f"  Watershed segmented areas: {spine_area_info['total_spine_areas']}")
    print(f"  Total segmented area: {spine_area_info['total_area_pixels']} pixels")
    if spine_area_info['avg_area_pixels'] > 0:
        print(f"  Average spine area: {spine_area_info['avg_area_pixels']:.1f} pixels")
    print(f"  Duplicates removed and smart frame assignment applied")
    
    return final_positions, brightest_path, all_results, spine_area_info


class SpineDetectionWidget(QWidget):
    """Widget for detecting dendritic spines along brightest paths using optimized tube data"""
    
    spines_detected = Signal(str, list)
    
    def __init__(self, viewer, image, state):
        super().__init__()
        self.viewer = viewer
        self.image = image
        self.state = state
        
        if 'spine_layers' not in self.state:
            self.state['spine_layers'] = {}
        
        self.handling_event = False
        
        # Get initial pixel spacing from state
        self.pixel_spacing_nm = self.state.get('pixel_spacing_nm', 94.0)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        
        layout.addWidget(QLabel("<b>Smart Spine Detection</b>"))
        layout.addWidget(QLabel("Intelligent spine tracking with intensity analysis"))
        layout.addWidget(QLabel("1. Select a segmented path\n2. Set parameters\n3. Optionally click points to add manual spines\n4. Click 'Detect Spines'"))
        layout.addWidget(QLabel("Note: To manually click points, select the current points layer from the left pane\nNow click on plus icon on the top and select points"))
        
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)
        
        # Path selection
        layout.addWidget(QLabel("Select a path for spine detection:"))
        self.path_list = QListWidget()
        self.path_list.setFixedHeight(80)
        self.path_list.itemSelectionChanged.connect(self.on_path_selection_changed)
        layout.addWidget(self.path_list)
        
        # Parameters group
        params_group = QGroupBox("Smart Detection Parameters (in nanometers)")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(2)
        params_layout.setContentsMargins(5, 5, 5, 5)
        
        # Max distance threshold in nanometers
        max_distance_layout = QHBoxLayout()
        max_distance_layout.addWidget(QLabel("Max Distance:"))
        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(100.0, 50000.0)
        self.max_distance_spin.setValue(2820.0)  # Default 2820 nm (30 pixels × 94 nm/pixel)
        self.max_distance_spin.setDecimals(0)
        self.max_distance_spin.setSuffix(" nm")
        self.max_distance_spin.setToolTip("Maximum distance threshold for grouping spines")
        max_distance_layout.addWidget(self.max_distance_spin)
        params_layout.addLayout(max_distance_layout)
        
        # Frame range for analysis
        frame_range_layout = QHBoxLayout()
        frame_range_layout.addWidget(QLabel("Frame Analysis:"))
        self.frame_range_spin = QSpinBox()
        self.frame_range_spin.setRange(1, 5)
        self.frame_range_spin.setValue(3)
        self.frame_range_spin.setToolTip("Number of frames to analyze for spine visibility (smart tracking)")
        frame_range_layout.addWidget(self.frame_range_spin)
        params_layout.addLayout(frame_range_layout)
        
        # Intensity threshold
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity Threshold:"))
        self.intensity_threshold_spin = QDoubleSpinBox()
        self.intensity_threshold_spin.setRange(0.1, 1.0)
        self.intensity_threshold_spin.setSingleStep(0.05)
        self.intensity_threshold_spin.setValue(0.1)
        self.intensity_threshold_spin.setDecimals(2)
        self.intensity_threshold_spin.setToolTip("Minimum intensity ratio to background for spine visibility")
        intensity_layout.addWidget(self.intensity_threshold_spin)
        params_layout.addLayout(intensity_layout)
        
        # Enable parallel processing
        self.enable_parallel_cb = QCheckBox("Enable Parallel Processing")
        self.enable_parallel_cb.setChecked(True)
        self.enable_parallel_cb.setToolTip("Use parallel processing for faster tube data generation")
        params_layout.addWidget(self.enable_parallel_cb)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)
        
        # Detection button
        self.detect_spines_btn = QPushButton("Detect Spines (Smart)")
        self.detect_spines_btn.setFixedHeight(22)
        self.detect_spines_btn.clicked.connect(self.run_spine_detection)
        self.detect_spines_btn.setEnabled(False)
        layout.addWidget(self.detect_spines_btn)
        
        # Progress bar
        self.detection_progress = QProgressBar()
        self.detection_progress.setValue(0)
        layout.addWidget(self.detection_progress)
        
        # Results section
        self.results_label = QLabel("Results: No spines detected yet")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)
        
        # Status message
        self.status_label = QLabel("Status: Select a segmented path to begin")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
    
    def get_manual_spine_points(self, path_name=None):
        """
        Get manually clicked spine points from points layers in the viewer.
        This includes points added to existing spine layers or new point layers.
        
        Args:
            path_name: Name of the current path to check its spine layer
        """
        manual_points = []
        original_spine_positions = []
        
        # If we have a path name, check if there's an existing spine layer for this path
        if path_name:
            spine_layer_name = f"Spines - {path_name}"
            for layer in self.viewer.layers:
                if (hasattr(layer, 'name') and layer.name == spine_layer_name and 
                    hasattr(layer, 'data') and len(layer.data) > 0):
                    
                    # Get the original spine positions from our stored data
                    path_id = getattr(self, 'selected_path_id', None)
                    if (path_id and 'spine_data' in self.state and 
                        path_id in self.state['spine_data'] and 
                        'original_positions' in self.state['spine_data'][path_id]):
                        original_spine_positions = self.state['spine_data'][path_id]['original_positions']
                        print(f"Found {len(original_spine_positions)} original spine positions")
                    
                    # Check if current layer has more points than originally detected
                    current_points = layer.data
                    if len(current_points) > len(original_spine_positions):
                        # Find new points that weren't in the original detection
                        for point in current_points:
                            point_coords = [point[0], point[1], point[2]]
                            
                            # Check if this point was in the original detection
                            is_original = False
                            for orig_point in original_spine_positions:
                                # Allow small tolerance for floating point differences
                                if (abs(point_coords[0] - orig_point[0]) < 0.1 and 
                                    abs(point_coords[1] - orig_point[1]) < 0.1 and 
                                    abs(point_coords[2] - orig_point[2]) < 0.1):
                                    is_original = True
                                    break
                            
                            if not is_original:
                                manual_points.append(point_coords)
                                print(f"Found new manual spine point at {point_coords[0]:.1f}, {point_coords[1]:.1f}, {point_coords[2]:.1f}")
                    
                    break
        
        # Also check other points layers (excluding system layers)
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and hasattr(layer, 'name'):
                # Check if it's a points layer with data
                if (hasattr(layer, 'size') and  # This indicates it's a points layer
                    len(layer.data) > 0 and 
                    layer.data.ndim == 2 and 
                    layer.data.shape[1] >= 3):  # At least 3D coordinates
                    
                    # Skip system layers and the spine layer we already processed
                    skip_layer = False
                    skip_names = ['Path', 'Point Selection', 'Traced Path']
                    if path_name:
                        skip_names.append(f"Spines - {path_name}")
                    
                    for skip_name in skip_names:
                        if skip_name in layer.name:
                            skip_layer = True
                            break
                    
                    if skip_layer:
                        continue
                    
                    # Include points from this layer
                    for point in layer.data:
                        # Ensure we have at least [z, y, x] coordinates
                        if len(point) >= 3:
                            manual_points.append([point[0], point[1], point[2]])
                            print(f"Found manual spine point at {point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f} from layer '{layer.name}'")
        
        if manual_points:
            print(f"Total manual spine points found: {len(manual_points)}")
        
        return manual_points
    
    def update_path_list(self):
        """Update the path list with current paths"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            self.path_list.clear()
            
            for path_id, path_data in self.state['paths'].items():
                path_name = path_data['name']
                seg_layer_name = f"Segmentation - {path_name}"
                
                # Check if segmentation exists
                has_segmentation = any(layer.name == seg_layer_name for layer in self.viewer.layers)
                
                if has_segmentation:
                    item = QListWidgetItem(path_data['name'])
                    item.setData(100, path_id)
                    self.path_list.addItem(item)
            
            self.detect_spines_btn.setEnabled(
                self.path_list.count() > 0 and
                self.path_list.currentRow() >= 0
            )
        except Exception as e:
            napari.utils.notifications.show_info(f"Error updating path list: {str(e)}")
            self.status_label.setText(f"Status: {str(e)}")
        finally:
            self.handling_event = False
    
    def on_path_selection_changed(self):
        """Handle path selection changes"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            if len(selected_items) == 1:
                path_id = selected_items[0].data(100)
                if path_id in self.state['paths']:
                    self.selected_path_id = path_id
                    path_name = self.state['paths'][path_id]['name']
                    
                    # Check if spines already exist
                    spine_layer_name = f"Spines - {path_name}"
                    has_spines = any(layer.name == spine_layer_name for layer in self.viewer.layers)
                    
                    # Check for manual points
                    manual_points = self.get_manual_spine_points(path_name)
                    manual_info = f" (+{len(manual_points)} manual points)" if manual_points else ""
                    
                    if has_spines:
                        self.status_label.setText(f"Status: Path '{path_name}' ready for smart spine detection{manual_info}")
                    else:
                        self.status_label.setText(f"Status: Path '{path_name}' selected for smart spine detection{manual_info}")
                    
                    self.detect_spines_btn.setEnabled(True)
            else:
                if hasattr(self, 'selected_path_id'):
                    delattr(self, 'selected_path_id')
                self.detect_spines_btn.setEnabled(False)
                self.status_label.setText("Status: Select a segmented path to begin")
        except Exception as e:
            napari.utils.notifications.show_info(f"Error handling path selection: {str(e)}")
        finally:
            self.handling_event = False
    
    def enable_for_path(self, path_id):
        """Enable spine detection for a specific segmented path"""
        for i in range(self.path_list.count()):
            item = self.path_list.item(i)
            if item.data(100) == path_id:
                self.path_list.setCurrentItem(item)
                break
        
        self.update_path_list()
    
    def run_spine_detection(self):
        """Run smart spine detection on the selected path"""
        if not hasattr(self, 'selected_path_id'):
            napari.utils.notifications.show_info("Please select a path for spine detection")
            return
        
        path_id = self.selected_path_id
        if path_id not in self.state['paths']:
            napari.utils.notifications.show_info("Selected path no longer exists")
            self.update_path_list()
            return
        
        try:
            # Get path data
            path_data = self.state['paths'][path_id]
            path_name = path_data['name']
            brightest_path = np.array(path_data['data'])
            
            # Get manually clicked spine points
            manual_spine_points = self.get_manual_spine_points(path_name)
            
            # Update UI
            manual_info = f" (including {len(manual_spine_points)} manual points)" if manual_spine_points else ""
            self.status_label.setText(f"Status: Running smart spine detection on {path_name}{manual_info}...")
            self.detection_progress.setValue(10)
            self.detect_spines_btn.setEnabled(False)
            
            # Get segmentation mask
            segmentation_mask = None
            seg_layer_name = f"Segmentation - {path_name}"
            for layer in self.viewer.layers:
                if layer.name == seg_layer_name:
                    segmentation_mask = layer.data
                    break
            
            if segmentation_mask is None:
                napari.utils.notifications.show_info(f"Segmentation not found for {path_name}")
                self.status_label.setText(f"Status: Segmentation not found for {path_name}")
                self.detect_spines_btn.setEnabled(True)
                return
            
            # Get parameters in nanometers
            max_distance_nm = self.max_distance_spin.value()
            frame_range = self.frame_range_spin.value()
            intensity_threshold = self.intensity_threshold_spin.value()
            enable_parallel = self.enable_parallel_cb.isChecked()
            
            # Use fixed default values for FOV and zoom size (not user-configurable)
            fov_nm = 50  # Fixed at 3760 nm (40 pixels × 94 nm/pixel default)
            zoom_size_nm = 40  # Fixed at 3760 nm (40 pixels × 94 nm/pixel default)
            
            # Convert nanometers to pixels using current pixel spacing
            pixel_spacing = self.pixel_spacing_nm
            max_distance_pixels = int(max_distance_nm / pixel_spacing)
            # fov_pixels = int(fov_nm / pixel_spacing)
            # zoom_size_pixels = int(zoom_size_nm / pixel_spacing)

            fov_pixels = fov_nm
            zoom_size_pixels = zoom_size_nm
            
            verbose = True
            if verbose:
                print(f"Smart spine detection parameters:")
                print(f"  Pixel spacing: {pixel_spacing:.1f} nm/pixel")
                print(f"  Max distance: {max_distance_nm:.0f} nm = {max_distance_pixels} pixels")
                print(f"  Frame analysis range: ±{frame_range} frames")
                print(f"  Intensity threshold: {intensity_threshold:.2f}")
                print(f"  Field of view: {fov_nm:.0f} nm = {fov_pixels} pixels (fixed)") 
                print(f"  Zoom size: {zoom_size_nm:.0f} nm = {zoom_size_pixels} pixels (fixed)")
                if manual_spine_points:
                    print(f"  Manual spine points: {len(manual_spine_points)}")
            
            self.detection_progress.setValue(20)
            napari.utils.notifications.show_info(f"Creating tube data for smart detection on {path_name}...")
            
            # Check if we have a pre-computed path to pass
            existing_path = brightest_path if brightest_path is not None else None
            
            # Create tube data using the optimized minimal function with pixel values
            tube_data = create_tube_data(
                image=self.image,
                points_list=[brightest_path[0], brightest_path[-1]],  # Start and end points
                existing_path=existing_path,  # Pass the existing path
                view_distance=1,  # This parameter is ignored in minimal version
                field_of_view=fov_pixels,  # Converted to pixels
                zoom_size=zoom_size_pixels,  # Converted to pixels
                reference_image=segmentation_mask,
                enable_parallel=enable_parallel,
                verbose=True
            )
            
            self.detection_progress.setValue(40)
            napari.utils.notifications.show_info("Running smart spine detection with watershed analysis...")
            
            # Progress callback
            def update_progress(current, total):
                progress = int(40 + (current / total) * 50)
                self.detection_progress.setValue(progress)
            
            # Run smart detection with watershed area analysis
            spine_positions, _, all_results, spine_area_info = process_all_frames_with_smart_tracking(
                tube_data=tube_data,
                image=self.image,
                brightest_path=brightest_path,
                max_distance_threshold=max_distance_pixels,  # Use pixel value
                frame_range=frame_range,
                manual_spine_points=manual_spine_points,  # Pass manual points
                progress_callback=update_progress
            )
            
            self.detection_progress.setValue(90)
            
            del tube_data
            import gc
            gc.collect()
            
            # Process results
            if spine_positions is not None and len(spine_positions) > 0:
                spine_layer_name = f"Spines - {path_name}"
                
                # Remove existing layer
                for layer in list(self.viewer.layers):
                    if layer.name == spine_layer_name:
                        self.viewer.layers.remove(layer)
                        break
                
                # Add new spine layer
                spine_layer = self.viewer.add_points(
                    spine_positions,
                    name=spine_layer_name,
                    size=8,
                    face_color='red',
                    opacity=0.8,
                    symbol='cross'
                )
                
                # Store references
                self.state['spine_layers'][path_id] = spine_layer
                self.state['spine_positions'] = spine_positions
                
                # Calculate statistics
                initial_count = sum(r['num_confirmed_spines'] for r in all_results)
                frames_with_spines = len([r for r in all_results if r['num_confirmed_spines'] > 0])
                manual_count = len(manual_spine_points) if manual_spine_points else 0
                
                # Calculate unique spines (approximate by grouping nearby positions)
                unique_spines = self._estimate_unique_spines(spine_positions)
                
                # Update UI with smart detection and watershed info
                algorithm_info = f" (smart tracking, watershed areas)" if enable_parallel else f" (smart tracking, watershed, sequential)"
                manual_info = f"Manual points: {manual_count}\n" if manual_count > 0 else ""
                watershed_info = f"Watershed areas: {spine_area_info['total_spine_areas']}\n"
                area_info = f"Total area: {spine_area_info['total_area_pixels']} pixels"
                if spine_area_info['avg_area_pixels'] > 0:
                    area_info += f", Avg: {spine_area_info['avg_area_pixels']:.1f} px"
                area_info += "\n"
                
                self.results_label.setText(
                    f"Results: {len(spine_positions)} spine positions across frames{algorithm_info}\n"
                    f"Initial detections: {initial_count} spines\n"
                    f"{manual_info}"
                    f"Estimated unique spines: {unique_spines}\n"
                    f"{watershed_info}"
                    f"{area_info}"
                    f"Frames with detections: {frames_with_spines}\n"
                    f"Intensity threshold: {intensity_threshold:.2f}, Frame analysis: ±{frame_range}\n"
                    f"Smart features: duplicate removal, intensity analysis, watershed areas"
                )
                self.status_label.setText(f"Status: Smart spine detection completed for {path_name}")
                
                # Store enhanced spine data with smart tracking info
                if 'spine_data' not in self.state:
                    self.state['spine_data'] = {}
                
                self.state['spine_data'][path_id] = {
                    'original_positions': spine_positions,
                    'all_results': all_results,
                    'spine_area_info': spine_area_info,  # Store watershed area data
                    'manual_spine_points': manual_spine_points,  # Store manual points
                    'detection_method': 'smart_intensity_based_tracking_with_watershed',
                    'parameters': {
                        'max_distance_nm': max_distance_nm,
                        'max_distance_pixels': max_distance_pixels,
                        'field_of_view_nm': fov_nm,
                        'field_of_view_pixels': fov_pixels,
                        'zoom_size_nm': zoom_size_nm,
                        'zoom_size_pixels': zoom_size_pixels,
                        'frame_range': frame_range,
                        'intensity_threshold': intensity_threshold,
                        'enable_parallel': enable_parallel,
                        'pixel_spacing_nm': pixel_spacing,
                        'manual_points_count': manual_count,
                        'estimated_unique_spines': unique_spines,
                        'watershed_enabled': True,
                        'total_spine_areas': spine_area_info['total_spine_areas'],
                        'avg_spine_area_pixels': spine_area_info['avg_area_pixels']
                    }
                }
                
                # Emit signal
                self.spines_detected.emit(path_id, spine_positions.tolist())
                
                # Create comprehensive notification message
                detection_msg = f"Smart detection with watershed completed: {len(spine_positions)} spine positions"
                if unique_spines != len(spine_positions):
                    detection_msg += f" ({unique_spines} unique spines)"
                if spine_area_info['total_spine_areas'] > 0:
                    detection_msg += f", {spine_area_info['total_spine_areas']} areas segmented"
                if manual_count > 0:
                    detection_msg += f" including {manual_count} manual points"
                detection_msg += " - watershed areas, duplicates removed, intensity-based frame selection"
                
                napari.utils.notifications.show_info(detection_msg)
            else:
                self.results_label.setText("Results: No spines detected")
                self.status_label.setText(f"Status: No spines detected for {path_name}")
                napari.utils.notifications.show_info("No spines detected")
        
        except Exception as e:
            error_msg = f"Error during smart spine detection: {str(e)}"
            self.status_label.setText(f"Status: {error_msg}")
            self.results_label.setText("Results: Error during smart spine detection")
            napari.utils.notifications.show_info(error_msg)
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.detection_progress.setValue(100)
            self.detect_spines_btn.setEnabled(True)
    
    def _estimate_unique_spines(self, spine_positions, distance_threshold=15):
        """Estimate the number of unique spines by grouping nearby positions"""
        if len(spine_positions) == 0:
            return 0
        
        # Group positions that are close in Y-X space (same physical spine)
        yx_positions = spine_positions[:, 1:3]  # Only Y and X coordinates
        
        unique_groups = []
        used_indices = set()
        
        for i, pos in enumerate(yx_positions):
            if i in used_indices:
                continue
            
            # Start new group
            current_group = [i]
            used_indices.add(i)
            
            # Find nearby positions
            for j, other_pos in enumerate(yx_positions):
                if j in used_indices:
                    continue
                
                distance = np.linalg.norm(pos - other_pos)
                if distance <= distance_threshold:
                    current_group.append(j)
                    used_indices.add(j)
            
            unique_groups.append(current_group)
        
        return len(unique_groups)
    
    def update_pixel_spacing(self, new_spacing):
        """Update pixel spacing and recalculate default parameter values"""
        self.pixel_spacing_nm = new_spacing
        
        # Update default values based on new pixel spacing (keeping same pixel equivalents)
        default_max_distance_pixels = 15  # Original default in pixels
        
        # Convert to nanometers with new spacing
        new_max_distance_nm = default_max_distance_pixels * new_spacing
        
        # Update the UI values (only max distance, FOV and zoom are now fixed)
        self.max_distance_spin.setValue(new_max_distance_nm)
        
        print(f"Smart spine detection: Updated to {new_spacing:.1f} nm/pixel")
        print(f"  Max distance: {new_max_distance_nm:.0f} nm")
        print(f"  Field of view: {40 * new_spacing:.0f} nm (fixed)")
        print(f"  Zoom size: {40 * new_spacing:.0f} nm (fixed)")