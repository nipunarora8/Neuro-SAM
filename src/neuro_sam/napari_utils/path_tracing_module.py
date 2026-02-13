import napari
import numpy as np
import uuid
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFrame, QCheckBox, QComboBox, QDoubleSpinBox
)
from qtpy.QtCore import Signal
import sys
from neuro_sam.brightest_path_lib.algorithm.waypointastar_speedup import quick_accurate_optimized_search
from scipy.interpolate import splprep, splev


class PathSmoother:
    """B-spline based path smoothing for dendrite traces with anisotropic scaling support"""
    
    def __init__(self):
        pass
    
    def smooth_path(self, path_points, spacing_xyz=(1.0, 1.0, 1.0), smoothing_factor=None, 
                   num_points=None, preserve_endpoints=True, image_shape=None):
        """
        Smooth a 3D path using B-spline interpolation with anisotropic scaling
        
        Args:
            path_points: numpy array of shape (N, 3) with [z, y, x] coordinates
            spacing_xyz: voxel spacing in (x, y, z) format for proper distance calculation
            smoothing_factor: B-spline smoothing parameter (higher = more smooth)
            num_points: number of points in smoothed path (None = same as input)
            preserve_endpoints: whether to keep original start/end points
        
        Returns:
            Smoothed path as numpy array
        """
        if len(path_points) < 3:
            return path_points.copy()
        
        # Store original endpoints
        start_point = path_points[0].copy()
        end_point = path_points[-1].copy()
        
        # Apply B-spline smoothing with anisotropic scaling consideration
        smoothed_path = self._bspline_smooth_anisotropic(
            path_points, spacing_xyz, smoothing_factor, num_points, image_shape
        )
        
        # Restore endpoints if requested
        if preserve_endpoints and len(smoothed_path) > 0:
            smoothed_path[0] = start_point
            smoothed_path[-1] = end_point
            
        # Apply connectivity filter AFTER endpoints are restored
        # This catches cases where the start point is far from the spline start
        smoothed_path = self._filter_connectivity(smoothed_path)
        
        # Prune sharp hooks at the endpoints (e.g. from user clicks being slightly off-axis)
        smoothed_path = self._prune_hooks(smoothed_path)
            
        return smoothed_path

    def _prune_hooks(self, path_points, max_angle_deg=30.0, max_prune_count=3):
        """Prune end points that form a sharp hook with the rest of the path"""
        if len(path_points) < 4:
            return path_points
            
        # We'll prune up to max_prune_count points from each end
        # Working on a copy
        points = path_points.copy()
        
        # Prune END
        for _ in range(max_prune_count):
            if len(points) < 4:
                break
                
            # Get last 3 points: p[-3], p[-2], p[-1]
            p1 = points[-3]
            p2 = points[-2]
            p3 = points[-1]
            
            # Vectors
            v1 = p2 - p1 # Segment before last
            v2 = p3 - p2 # Last segment
            
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                break
                
            # Angle
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            # If angle > threshold, prune the last point
            if angle_deg > max_angle_deg:
                # print(f"Pruning end point: angle {angle_deg:.1f} > {max_angle_deg}")
                points = points[:-1]
            else:
                break
                
        # Prune START
        for _ in range(max_prune_count):
            if len(points) < 4:
                break
                
            # Get first 3 points: p[0], p[1], p[2]
            p1 = points[0]
            p2 = points[1]
            p3 = points[2]
            
            # Vectors
            v1 = p2 - p1 # First segment
            v2 = p3 - p2 # Segment after first
            
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                break
                
            # Angle
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            # If angle > threshold, prune the first point
            if angle_deg > max_angle_deg:
                # print(f"Pruning start point: angle {angle_deg:.1f} > {max_angle_deg}")
                points = points[1:]
            else:
                break
                
        return points

    def _filter_connectivity(self, path_points):
        """Remove detached segments by keeping largest connected component"""
        if len(path_points) < 3:
            return path_points
            
        # Calculate distances between consecutive points
        diffs = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
        
        # Jumps larger than this likely mean a break (e.g., 5 pixels)
        jump_threshold = 5.0 
        
        # Find indices where jumps occur
        jump_indices = np.where(diffs > jump_threshold)[0]
        
        if len(jump_indices) > 0:
            # Split path into components
            components = []
            start_idx = 0
            for jump_idx in jump_indices:
                end_idx = jump_idx + 1 
                components.append(path_points[start_idx:end_idx])
                start_idx = end_idx
            components.append(path_points[start_idx:])
            
            # Keep the largest component
            largest_component = max(components, key=len)
            
            if len(largest_component) > 2:
                return largest_component
                
        return path_points
    
    def _bspline_smooth_anisotropic(self, path_points, spacing_xyz, smoothing_factor=None, num_points=None, image_shape=None):
        """B-spline smoothing with anisotropic spacing consideration and outlier filtering"""
        if smoothing_factor is None:
            # Auto-determine smoothing factor based on path length and anisotropy
            path_length = len(path_points)
            # Consider anisotropy in smoothing factor calculation
            anisotropy_factor = max(spacing_xyz) / min(spacing_xyz)
            smoothing_factor = max(0, path_length - np.sqrt(2 * path_length)) * anisotropy_factor
        
        if num_points is None:
            num_points = len(path_points)
        
        try:
            # Scale coordinates by voxel spacing for proper distance calculation
            # path_points are in [z, y, x] format, spacing is in (x, y, z) format
            scaled_points = path_points.copy().astype(float)
            scaled_points[:, 0] *= spacing_xyz[2]  # Z scaling
            scaled_points[:, 1] *= spacing_xyz[1]  # Y scaling  
            scaled_points[:, 2] *= spacing_xyz[0]  # X scaling
            
            # Prepare coordinates for spline fitting
            x = scaled_points[:, 2]  # x coordinates (scaled)
            y = scaled_points[:, 1]  # y coordinates (scaled)
            z = scaled_points[:, 0]  # z coordinates (scaled)
            
            # Fit B-spline (k=3 for cubic, k=min(3, len-1) for short paths)
            k = min(3, len(path_points) - 1)
            tck, u = splprep([x, y, z], s=smoothing_factor, k=k)
            
            # Generate smoothed points
            u_new = np.linspace(0, 1, num_points)
            x_smooth, y_smooth, z_smooth = splev(u_new, tck)
            
            # Scale back to voxel coordinates
            x_smooth /= spacing_xyz[0]  # Unscale X
            y_smooth /= spacing_xyz[1]  # Unscale Y
            z_smooth /= spacing_xyz[2]  # Unscale Z
            
            # Combine back to [z, y, x] format
            smoothed_path = np.column_stack([z_smooth, y_smooth, x_smooth])

            # --- FILTERING LOGIC ---
            # Remove points that are wildly outside the valid range
            
            # 1. Determine valid Z-range from input points with margin
            z_input = path_points[:, 0]
            z_margin = 5  # Allow 5 slices of wiggle room
            valid_z_min = z_input.min() - z_margin
            valid_z_max = z_input.max() + z_margin
            
            # 2. Filter points
            valid_indices = []
            for i in range(len(smoothed_path)):
                z, y, x = smoothed_path[i]
                
                # Check Z-range
                if z < valid_z_min or z > valid_z_max:
                    continue
                    
                # Check image bounds if shape is provided
                if image_shape is not None:
                    depth, height, width = image_shape
                    if (z < 0 or z >= depth or 
                        y < 0 or y >= height or 
                        x < 0 or x >= width):
                        continue
                
                valid_indices.append(i)
            
            return smoothed_path
            
        except Exception as e:
            print(f"Anisotropic B-spline smoothing failed: {e}, falling back to original path")
            return path_points.copy()


class PathTracingWidget(QWidget):
    """Widget for tracing the brightest path with anisotropic scaling support."""
    
    # Define signals
    path_created = Signal(str, str, object)  # path_id, path_name, path_data
    path_updated = Signal(str, str, object)  # path_id, path_name, path_data
    
    def __init__(self, viewer, image, state, scaler, scaling_update_callback):
        """Initialize the path tracing widget with anisotropic scaling support.
        
        Parameters:
        -----------
        viewer : napari.Viewer
            The napari viewer instance
        image : numpy.ndarray
            3D or higher-dimensional image data (potentially scaled)
        state : dict
            Shared state dictionary between modules
        scaler : AnisotropicScaler
            The scaler instance for coordinate conversion
        scaling_update_callback : function
            Callback function when scaling is updated
        """
        super().__init__()
        self.viewer = viewer
        self.image = image
        self.state = state
        self.scaler = scaler
        self.scaling_update_callback = scaling_update_callback
        
        # List to store waypoints as they are clicked
        self.clicked_points = []
        
        # Settings for path finding
        self.next_path_number = 1
        self.color_idx = 0
        
        # Flag to prevent recursive event handling
        self.handling_event = False
        
        # Initialize path smoother with anisotropic support
        self.path_smoother = PathSmoother()
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        
        # Main instruction
        title = QLabel("<b>Path Tracing (Anisotropic)</b>")
        layout.addWidget(title)
        
        # Instructions section
        instructions_section = QWidget()
        instructions_layout = QVBoxLayout()
        instructions_layout.setSpacing(2)
        instructions_layout.setContentsMargins(2, 2, 2, 2)
        instructions_section.setLayout(instructions_layout)
        
        instructions = QLabel(
            "<b>Step 1: Configure Voxel Spacing</b><br>"
            "Set correct X, Y, Z spacing, then click 'Apply Scaling'<br><br>"
            "<b>Step 2: Trace Paths</b><br>"
            "1. Click points on dendrite structure<br>"
            "2. Click 'Find Path' to trace<br>"
            "3. Use 'Trace Another Path' for additional paths"
        )
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)
        
        layout.addWidget(instructions_section)
        
        # Add separator
        separator0 = QFrame()
        separator0.setFrameShape(QFrame.HLine)
        separator0.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator0)
        
        # Anisotropic scaling section
        self._add_scaling_controls(layout)
        
        # Add separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)
        
        # Current spacing display
        spacing_info_section = QWidget()
        spacing_info_layout = QVBoxLayout()
        spacing_info_layout.setSpacing(2)
        spacing_info_layout.setContentsMargins(2, 2, 2, 2)
        spacing_info_section.setLayout(spacing_info_layout)
        
        self.spacing_info_label = QLabel("Current voxel spacing: Not set")
        self.spacing_info_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        spacing_info_layout.addWidget(self.spacing_info_label)
        
        layout.addWidget(spacing_info_section)
        
        # Update spacing display
        self._update_spacing_display()
        
        # Waypoint controls section
        waypoints_section = QWidget()
        waypoints_layout = QVBoxLayout()
        waypoints_layout.setSpacing(2)
        waypoints_layout.setContentsMargins(2, 2, 2, 2)
        waypoints_section.setLayout(waypoints_layout)
        
        self.select_waypoints_btn = QPushButton("Start Point Selection")
        self.select_waypoints_btn.setFixedHeight(22)
        self.select_waypoints_btn.clicked.connect(self.activate_waypoints_layer)
        waypoints_layout.addWidget(self.select_waypoints_btn)
        
        self.waypoints_status = QLabel("Status: Click to start selecting points")
        waypoints_layout.addWidget(self.waypoints_status)
        layout.addWidget(waypoints_section)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Fast algorithm settings
        algorithm_section = QWidget()
        algorithm_layout = QVBoxLayout()
        algorithm_layout.setSpacing(2)
        algorithm_layout.setContentsMargins(2, 2, 2, 2)
        algorithm_section.setLayout(algorithm_layout)
        
        # Parallel processing checkbox
        self.enable_parallel_cb = QCheckBox("Enable Parallel Processing")
        self.enable_parallel_cb.setChecked(True)
        self.enable_parallel_cb.setToolTip("Use parallel processing for faster pathfinding")
        algorithm_layout.addWidget(self.enable_parallel_cb)
        
        # Weight heuristic parameter
        weight_heuristic_layout = QHBoxLayout()
        weight_heuristic_layout.setSpacing(2)
        weight_heuristic_layout.addWidget(QLabel("Weight Heuristic:"))
        self.weight_heuristic_spin = QDoubleSpinBox()
        self.weight_heuristic_spin.setRange(0.1, 10.0)
        self.weight_heuristic_spin.setSingleStep(0.1)
        self.weight_heuristic_spin.setValue(2.0)  # Default value
        self.weight_heuristic_spin.setDecimals(1)
        self.weight_heuristic_spin.setToolTip("Weight heuristic for A* search algorithm (higher = more heuristic-guided)")
        weight_heuristic_layout.addWidget(self.weight_heuristic_spin)
        algorithm_layout.addLayout(weight_heuristic_layout)
        
        layout.addWidget(algorithm_section)
        
        # Smoothing controls section with anisotropic support
        smoothing_section = QWidget()
        smoothing_layout = QVBoxLayout()
        smoothing_layout.setSpacing(2)
        smoothing_layout.setContentsMargins(2, 2, 2, 2)
        smoothing_section.setLayout(smoothing_layout)
        
        # Smoothing checkbox
        self.enable_smoothing_cb = QCheckBox("Enable Anisotropic B-spline Smoothing")
        self.enable_smoothing_cb.setChecked(True)
        self.enable_smoothing_cb.setToolTip("Apply B-spline smoothing that considers anisotropic voxel spacing")
        smoothing_layout.addWidget(self.enable_smoothing_cb)
        
        # Smoothing factor
        factor_layout = QHBoxLayout()
        factor_layout.setSpacing(2)
        factor_layout.addWidget(QLabel("Smoothing:"))
        self.smoothing_factor_spin = QDoubleSpinBox()
        self.smoothing_factor_spin.setRange(0.0, 10.0)
        self.smoothing_factor_spin.setSingleStep(0.1)
        self.smoothing_factor_spin.setValue(1.0)
        self.smoothing_factor_spin.setToolTip("Higher values = more smoothing (0 = no smoothing)")
        factor_layout.addWidget(self.smoothing_factor_spin)
        smoothing_layout.addLayout(factor_layout)
        
        layout.addWidget(smoothing_section)
        
        # Add separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)
        
        # Action buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(2)
        
        # Main path finding button
        self.find_path_btn = QPushButton("Find Path")
        self.find_path_btn.setFixedHeight(26)
        self.find_path_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.find_path_btn.clicked.connect(self.find_path)
        self.find_path_btn.setEnabled(False)
        buttons_layout.addWidget(self.find_path_btn)
        
        layout.addLayout(buttons_layout)
        
        # Management buttons
        management_layout = QHBoxLayout()
        management_layout.setSpacing(2)
        
        self.trace_another_btn = QPushButton("Trace Another Path")
        self.trace_another_btn.setFixedHeight(22)
        self.trace_another_btn.clicked.connect(self.trace_another_path)
        self.trace_another_btn.setEnabled(False)
        management_layout.addWidget(self.trace_another_btn)
        
        self.clear_points_btn = QPushButton("Clear All Points")
        self.clear_points_btn.setFixedHeight(22)
        self.clear_points_btn.clicked.connect(self.clear_points)
        management_layout.addWidget(self.clear_points_btn)
        
        layout.addLayout(management_layout)
        
        # Status messages
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        self.error_status = QLabel("")
        self.error_status.setStyleSheet("color: red;")
        layout.addWidget(self.error_status)
    
    def _add_scaling_controls(self, layout):
        """Add anisotropic scaling controls to the layout"""
        from qtpy.QtWidgets import (QGroupBox, QDoubleSpinBox, QComboBox, QCheckBox)
        
        # Scaling section
        scaling_group = QGroupBox("Anisotropic Voxel Spacing")
        scaling_layout = QVBoxLayout()
        scaling_layout.setSpacing(2)
        scaling_layout.setContentsMargins(5, 5, 5, 5)
        
        # Instructions
        info_label = QLabel("Set voxel spacing in nanometers (will reshape the dataset):")
        info_label.setWordWrap(True)
        scaling_layout.addWidget(info_label)
        
        # X spacing
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X spacing:"))
        self.x_spacing_spin = QDoubleSpinBox()
        self.x_spacing_spin.setRange(1.0, 10000.0)
        self.x_spacing_spin.setSingleStep(1.0)
        self.x_spacing_spin.setValue(self.scaler.current_spacing_xyz[0])
        self.x_spacing_spin.setDecimals(1)
        self.x_spacing_spin.setSuffix(" nm")
        self.x_spacing_spin.setToolTip("X-axis voxel spacing in nanometers")
        self.x_spacing_spin.valueChanged.connect(self._on_spacing_changed)
        x_layout.addWidget(self.x_spacing_spin)
        scaling_layout.addLayout(x_layout)
        
        # Y spacing
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y spacing:"))
        self.y_spacing_spin = QDoubleSpinBox()
        self.y_spacing_spin.setRange(1.0, 10000.0)
        self.y_spacing_spin.setSingleStep(1.0)
        self.y_spacing_spin.setValue(self.scaler.current_spacing_xyz[1])
        self.y_spacing_spin.setDecimals(1)
        self.y_spacing_spin.setSuffix(" nm")
        self.y_spacing_spin.setToolTip("Y-axis voxel spacing in nanometers")
        self.y_spacing_spin.valueChanged.connect(self._on_spacing_changed)
        y_layout.addWidget(self.y_spacing_spin)
        scaling_layout.addLayout(y_layout)
        
        # Z spacing
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z spacing:"))
        self.z_spacing_spin = QDoubleSpinBox()
        self.z_spacing_spin.setRange(1.0, 10000.0)
        self.z_spacing_spin.setSingleStep(1.0)
        self.z_spacing_spin.setValue(self.scaler.current_spacing_xyz[2])
        self.z_spacing_spin.setDecimals(1)
        self.z_spacing_spin.setSuffix(" nm")
        self.z_spacing_spin.setToolTip("Z-axis voxel spacing in nanometers")
        self.z_spacing_spin.valueChanged.connect(self._on_spacing_changed)
        z_layout.addWidget(self.z_spacing_spin)
        scaling_layout.addLayout(z_layout)
        
        # Interpolation method
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("Interpolation:"))
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["Nearest", "Linear", "Cubic"])
        self.interp_combo.setCurrentIndex(1)  # Default to linear
        self.interp_combo.setToolTip("Interpolation method for scaling")
        interp_layout.addWidget(self.interp_combo)
        scaling_layout.addLayout(interp_layout)
        
        # Auto-update checkbox
        self.auto_update_cb = QCheckBox("Auto-update on change")
        self.auto_update_cb.setChecked(False)
        self.auto_update_cb.setToolTip("Automatically apply scaling when values change")
        scaling_layout.addWidget(self.auto_update_cb)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.apply_scaling_btn = QPushButton("Apply Scaling")
        self.apply_scaling_btn.setToolTip("Apply current scaling settings to reshape the image")
        self.apply_scaling_btn.setFixedHeight(22)
        self.apply_scaling_btn.clicked.connect(self._apply_scaling)
        button_layout.addWidget(self.apply_scaling_btn)
        
        self.reset_scaling_btn = QPushButton("Reset to Original")
        self.reset_scaling_btn.setToolTip("Reset to original voxel spacing")
        self.reset_scaling_btn.setFixedHeight(22)
        self.reset_scaling_btn.clicked.connect(self._reset_scaling)
        button_layout.addWidget(self.reset_scaling_btn)
        
        scaling_layout.addLayout(button_layout)
        
        # Status info
        self.scaling_status = QLabel("Status: Original spacing")
        self.scaling_status.setWordWrap(True)
        self.scaling_status.setStyleSheet("font-weight: bold; color: #0066cc;")
        scaling_layout.addWidget(self.scaling_status)
        
        scaling_group.setLayout(scaling_layout)
        layout.addWidget(scaling_group)
        
        # Update initial status
        self._update_status_only()
    
    def _on_spacing_changed(self):
        """Handle when spacing values change"""
        if self.auto_update_cb.isChecked():
            self._apply_scaling()
        else:
            self._update_status_only()
            
    def _update_status_only(self):
        """Update status without applying scaling"""
        x_nm = self.x_spacing_spin.value()
        y_nm = self.y_spacing_spin.value() 
        z_nm = self.z_spacing_spin.value()
        
        # Calculate what the scale factors would be
        temp_scale_factors = np.array([
            self.scaler.original_spacing_xyz[2] / z_nm,  # Z
            self.scaler.original_spacing_xyz[1] / y_nm,  # Y
            self.scaler.original_spacing_xyz[0] / x_nm   # X
        ])
        
        self.scaling_status.setText(
            f"Pending: X={x_nm:.1f}, Y={y_nm:.1f}, Z={z_nm:.1f} nm\n"
            f"Scale factors (Z,Y,X): {temp_scale_factors[0]:.3f}, {temp_scale_factors[1]:.3f}, {temp_scale_factors[2]:.3f}"
        )
        
    def _apply_scaling(self):
        """Apply current scaling settings"""
        try:
            x_nm = self.x_spacing_spin.value()
            y_nm = self.y_spacing_spin.value()
            z_nm = self.z_spacing_spin.value()
            
            # Update scaler
            self.scaler.set_spacing(x_nm, y_nm, z_nm)
            
            # Get interpolation order
            interp_order = self.interp_combo.currentIndex()
            if interp_order == 0:
                order = 0  # Nearest
            elif interp_order == 1:
                order = 1  # Linear
            else:
                order = 3  # Cubic
            
            # Update status
            volume_ratio = self.scaler.get_volume_ratio()
            self.scaling_status.setText(
                f"Applied: X={x_nm:.1f}, Y={y_nm:.1f}, Z={z_nm:.1f} nm\n"
                f"Scale factors (Z,Y,X): {self.scaler.scale_factors[0]:.3f}, {self.scaler.scale_factors[1]:.3f}, {self.scaler.scale_factors[2]:.3f}\n"
                f"Volume ratio: {volume_ratio:.3f}"
            )
            
            # Call the main widget's scaling update callback
            if self.scaling_update_callback:
                self.scaling_update_callback(order)
                
            napari.utils.notifications.show_info(f"Applied anisotropic scaling: X={x_nm:.1f}, Y={y_nm:.1f}, Z={z_nm:.1f} nm")
            
        except Exception as e:
            napari.utils.notifications.show_info(f"Error applying scaling: {str(e)}")
            print(f"Scaling error: {str(e)}")
            
    def _reset_scaling(self):
        """Reset to original scaling"""
        self.scaler.reset_to_original()
        
        # Update UI
        self.x_spacing_spin.setValue(self.scaler.current_spacing_xyz[0])
        self.y_spacing_spin.setValue(self.scaler.current_spacing_xyz[1])
        self.z_spacing_spin.setValue(self.scaler.current_spacing_xyz[2])
        
        self.scaling_status.setText("Status: Reset to original spacing")
        
        # Call the main widget's scaling update callback
        if self.scaling_update_callback:
            self.scaling_update_callback(1)  # Linear interpolation for reset
            
        napari.utils.notifications.show_info("Reset to original voxel spacing")
    
    def _update_spacing_display(self):
        """Update the spacing display with current values"""
        try:
            if 'current_spacing_xyz' in self.state:
                spacing = self.state['current_spacing_xyz']
                self.spacing_info_label.setText(
                    f"Current voxel spacing: X={spacing[0]:.1f}, Y={spacing[1]:.1f}, Z={spacing[2]:.1f} nm"
                )
            else:
                self.spacing_info_label.setText("Current voxel spacing: Not configured")
        except Exception as e:
            self.spacing_info_label.setText("Current voxel spacing: Error reading values")
            print(f"Error updating spacing display: {e}")
    
    def activate_waypoints_layer(self):
        """Activate the waypoints layer for selecting points"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            self.viewer.layers.selection.active = self.state['waypoints_layer']
            self.error_status.setText("")
            self.status_label.setText("Click points on the dendrite structure")
            self._update_spacing_display()  # Update spacing display
            napari.utils.notifications.show_info("Click points on the dendrite")
        except Exception as e:
            error_msg = f"Error activating waypoints layer: {str(e)}"
            napari.utils.notifications.show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False
    
    def on_waypoints_changed(self, event=None):
        """Handle when waypoints are added or changed"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            waypoints_layer = self.state['waypoints_layer']
            if len(waypoints_layer.data) > 0:
                # Validate points are within image bounds
                valid_points = []
                for point in waypoints_layer.data:
                    valid = True
                    for i, coord in enumerate(point):
                        if coord < 0 or coord >= self.image.shape[i]:
                            valid = False
                            break
                            
                    if valid:
                        valid_points.append(point)
                    
                # Update the waypoints layer with only valid points
                if len(valid_points) != len(waypoints_layer.data):
                    waypoints_layer.data = np.array(valid_points)
                    napari.utils.notifications.show_info("Some points were outside image bounds and were removed.")
                
                # Convert to integer coordinates and store
                self.clicked_points = [point.astype(int) for point in valid_points]
                
                # Update status
                num_points = len(self.clicked_points)
                self.waypoints_status.setText(f"Status: {num_points} points selected")
                
                # Enable buttons if we have enough points
                self.find_path_btn.setEnabled(num_points >= 2)
                
                if num_points >= 2:
                    self.status_label.setText("Ready to find path using anisotropic algorithm!")
                else:
                    self.status_label.setText(f"Need at least 2 points (currently have {num_points})")
            else:
                self.clicked_points = []
                self.waypoints_status.setText("Status: Click to start selecting points")
                self.find_path_btn.setEnabled(False)
                self.status_label.setText("")
        except Exception as e:
            napari.utils.notifications.show_info(f"Error processing waypoints: {str(e)}")
            print(f"Error details: {str(e)}")
        finally:
            self.handling_event = False
    
    def find_path(self):
        """Find path using the fast waypoint A* algorithm with anisotropic smoothing"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            if len(self.clicked_points) < 2:
                napari.utils.notifications.show_info("Please select at least 2 points")
                self.error_status.setText("Error: Please select at least 2 points")
                return
            
            # Clear any previous error messages
            self.error_status.setText("")
            self.status_label.setText("Finding path using anisotropic algorithm...")
            
            # Convert clicked points to the format expected by the algorithm
            points_list = [point.tolist() for point in self.clicked_points]
            
            # Get algorithm settings
            enable_parallel = self.enable_parallel_cb.isChecked()
            weight_heuristic = self.weight_heuristic_spin.value()
            
            napari.utils.notifications.show_info("Finding brightest path with anisotropic-aware algorithm...")
            
            # Use the fast waypoint A* algorithm with weight heuristic parameter
            path = quick_accurate_optimized_search(
                image=self.image,
                points_list=points_list,
                verbose=True,
                enable_parallel=enable_parallel,
                my_weight_heuristic=weight_heuristic
            )
            
            # Check if path was found
            if path is not None and len(path) > 0:
                # Convert path to numpy array
                path_data = np.array(path)
                
                # Apply anisotropic smoothing if enabled
                if self.enable_smoothing_cb.isChecked() and len(path_data) >= 3:
                    smoothing_factor = self.smoothing_factor_spin.value()
                    
                    if smoothing_factor > 0:
                        self.status_label.setText("Applying anisotropic B-spline smoothing...")
                        
                        # Get current voxel spacing
                        spacing_xyz = self.state.get('current_spacing_xyz', (1.0, 1.0, 1.0))
                        
                        # Smooth the path using anisotropic B-spline
                        smoothed_path = self.path_smoother.smooth_path(
                            path_data, 
                            spacing_xyz=spacing_xyz,
                            smoothing_factor=smoothing_factor,
                            preserve_endpoints=True,
                            image_shape=self.image.shape
                        )
                        
                        # Update path data
                        path_data = smoothed_path
                        napari.utils.notifications.show_info("Applied anisotropic B-spline smoothing")
                
                # Generate path name
                path_name = f"Path {self.next_path_number}"
                self.next_path_number += 1
                
                # Get color for this path
                path_color = self.get_next_color()
                
                # Create a new layer for this path
                path_layer = self.viewer.add_points(
                    path_data,
                    name=path_name,
                    size=3,
                    face_color=path_color,
                    opacity=0.8
                )
                
                # Update 3D visualization if applicable
                if self.image.ndim > 2 and self.state['traced_path_layer'] is not None:
                    self._update_traced_path_visualization(path_data)
                
                # Generate a unique ID for this path
                path_id = str(uuid.uuid4())
                
                # Store the path with enhanced metadata including spacing and weight heuristic
                current_spacing = self.state.get('current_spacing_xyz', (1.0, 1.0, 1.0))
                
                self.state['paths'][path_id] = {
                    'name': path_name,
                    'data': path_data,
                    'start': self.clicked_points[0].copy(),
                    'end': self.clicked_points[-1].copy(),
                    'waypoints': [point.copy() for point in self.clicked_points[1:-1]] if len(self.clicked_points) > 2 else [],
                    'visible': True,
                    'layer': path_layer,
                    'original_clicks': [point.copy() for point in self.clicked_points],
                    'smoothed': self.enable_smoothing_cb.isChecked() and self.smoothing_factor_spin.value() > 0,
                    'algorithm': 'waypoint_astar',
                    'parallel_processing': enable_parallel,
                    'weight_heuristic': weight_heuristic,  # Store weight heuristic parameter
                    'voxel_spacing_xyz': current_spacing,  # Store voxel spacing with path
                    'anisotropic_smoothing': self.enable_smoothing_cb.isChecked()
                }
                
                # Store reference to the layer
                self.state['path_layers'][path_id] = path_layer
                
                # Update UI
                algorithm_info = f" (parallel, weight={weight_heuristic:.1f})" if enable_parallel else f" (sequential, weight={weight_heuristic:.1f})"
                smoothing_msg = " (anisotropic smoothing)" if self.state['paths'][path_id]['smoothed'] else ""
                spacing_info = f" at {current_spacing[0]:.1f}, {current_spacing[1]:.1f}, {current_spacing[2]:.1f} nm"
                
                msg = f"Anisotropic path found: {len(path_data)} points{algorithm_info}{smoothing_msg}{spacing_info}"
                napari.utils.notifications.show_info(msg)
                self.status_label.setText(f"Success: {path_name} created with anisotropic algorithm{smoothing_msg}")
                
                # Enable trace another path button
                self.trace_another_btn.setEnabled(True)
                
                # Store current path ID in state
                self.state['current_path_id'] = path_id
                
                # Emit signal that a new path was created
                self.path_created.emit(path_id, path_name, path_data)
                    
            else:
                # No path found
                msg = "Could not find a path with anisotropic algorithm"
                napari.utils.notifications.show_info(msg)
                self.error_status.setText("Error: No path found")
                self.status_label.setText("Try selecting different points or adjusting parameters")
                
        except Exception as e:
            msg = f"Error in anisotropic path finding: {e}"
            napari.utils.notifications.show_info(msg)
            self.error_status.setText(f"Error: {str(e)}")
            print(f"Anisotropic path finding error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.handling_event = False
    
    def _update_traced_path_visualization(self, path):
        """Update the 3D traced path visualization"""
        if self.state['traced_path_layer'] is None:
            return
            
        try:
            # Get the z-range of the path
            path_array = np.array(path)
            z_values = [point[0] for point in path]
            min_z = int(min(z_values))
            max_z = int(max(z_values))
            
            # Create a projection of the path onto every frame in the range
            traced_points = []
            for z in range(min_z, max_z + 1):
                for point in path:
                    new_point = point.copy()
                    new_point[0] = z
                    traced_points.append(new_point)
            
            # Update the traced path layer
            if traced_points:
                self.state['traced_path_layer'].data = np.array(traced_points)
                self.state['traced_path_layer'].visible = True
                self.viewer.dims.set_point(0, min_z)
        except Exception as e:
            print(f"Error updating traced path visualization: {e}")
    
    def trace_another_path(self):
        """Reset for tracing a new path while preserving existing paths"""
        # Clear current points
        self.clicked_points = []
        self.state['waypoints_layer'].data = np.empty((0, self.image.ndim))
        
        # Reset UI for new path
        self.waypoints_status.setText("Status: Click to start selecting points")
        self.status_label.setText("Ready for new anisotropic path - click points on dendrite")
        self.find_path_btn.setEnabled(False)
        self.trace_another_btn.setEnabled(False)
        
        # Update spacing display
        self._update_spacing_display()
        
        # Activate the waypoints layer for the new path
        self.viewer.layers.selection.active = self.state['waypoints_layer']
        napari.utils.notifications.show_info("Ready to trace a new anisotropic path. Click points on the dendrite.")
    
    def clear_points(self):
        """Clear all waypoints and paths"""
        self.clicked_points = []
        self.state['waypoints_layer'].data = np.empty((0, self.image.ndim))
        
        # Clear traced path layer if it exists
        if self.state['traced_path_layer'] is not None:
            self.state['traced_path_layer'].data = np.empty((0, self.image.ndim))
            self.state['traced_path_layer'].visible = False
            
        # Reset UI
        self.waypoints_status.setText("Status: Click to start selecting points")
        self.status_label.setText("")
        self.error_status.setText("")
        
        # Reset buttons
        self.find_path_btn.setEnabled(False)
        self.trace_another_btn.setEnabled(False)
        
        # Update spacing display
        self._update_spacing_display()
        
        napari.utils.notifications.show_info("All points cleared. Ready to start over with anisotropic scaling.")
    
    def get_next_color(self):
        """Get the next color from the predefined list"""
        colors = ['cyan', 'magenta', 'green', 'blue', 'orange', 
                  'purple', 'teal', 'coral', 'gold', 'lavender']
        
        color = colors[self.color_idx % len(colors)]
        self.color_idx += 1
        
        return color
    
    def load_path_waypoints(self, path_id):
        """Load the waypoints for a specific path"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            if path_id not in self.state['paths']:
                return
                
            path_data = self.state['paths'][path_id]
            
            # Check if this path has original clicks stored
            if ('original_clicks' in path_data and 
                len(path_data['original_clicks']) > 0):
                # Load the original clicked points
                self.clicked_points = [np.array(point) for point in path_data['original_clicks']]
                
                # Update the waypoints layer
                if self.clicked_points:
                    self.state['waypoints_layer'].data = np.array(self.clicked_points)
                    self.waypoints_status.setText(f"Status: {len(self.clicked_points)} points loaded")
            elif ('original_clicks' in path_data and 
                  len(path_data['original_clicks']) == 0):
                # This is a connected path - show a subset of the path points as waypoints
                path_points = path_data['data']
                if len(path_points) > 10:
                    # Take every nth point to get about 10 waypoints
                    step = len(path_points) // 10
                    waypoint_indices = range(0, len(path_points), step)
                    new_waypoints = [path_points[i] for i in waypoint_indices]
                else:
                    # Use all points if path is short
                    new_waypoints = path_points.copy()
                
                self.clicked_points = new_waypoints
                self.state['waypoints_layer'].data = np.array(new_waypoints)
                self.waypoints_status.setText(f"Status: {len(new_waypoints)} waypoints from connected path")
            else:
                # Fallback - reconstruct from start, waypoints, and end
                new_waypoints = []
                if 'start' in path_data and path_data['start'] is not None:
                    new_waypoints.append(path_data['start'])
                    
                if 'waypoints' in path_data and path_data['waypoints']:
                    new_waypoints.extend(path_data['waypoints'])
                    
                if 'end' in path_data and path_data['end'] is not None:
                    new_waypoints.append(path_data['end'])
                
                # Update the waypoints layer
                if new_waypoints:
                    self.state['waypoints_layer'].data = np.array(new_waypoints)
                    self.clicked_points = new_waypoints
                    self.waypoints_status.setText(f"Status: {len(new_waypoints)} points loaded")
            
            # Enable buttons
            if len(self.clicked_points) >= 2:
                self.find_path_btn.setEnabled(True)
                self.trace_another_btn.setEnabled(True)
                
            # Clear any error messages
            self.error_status.setText("")
            
            # Show path status including algorithm type, weight heuristic, and spacing info
            path_type = ""
            if path_data.get('algorithm') == 'waypoint_astar':
                path_type = " (waypoint_astar"
                if path_data.get('parallel_processing', False):
                    path_type += ", parallel"
                # Add weight heuristic info if available
                if 'weight_heuristic' in path_data:
                    path_type += f", weight={path_data['weight_heuristic']:.1f}"
                path_type += ")"
                if path_data.get('anisotropic_smoothing', False):
                    path_type += " (anisotropic smoothing)"
            elif path_data.get('smoothed', False):
                path_type = " (smoothed)"
            elif ('original_clicks' in path_data and 
                  len(path_data['original_clicks']) == 0):
                path_type = " (connected)"
            
            # Add spacing info if available
            if 'voxel_spacing_xyz' in path_data:
                spacing = path_data['voxel_spacing_xyz']
                spacing_info = f" [X={spacing[0]:.1f}, Y={spacing[1]:.1f}, Z={spacing[2]:.1f} nm]"
                path_type += spacing_info
            
            self.status_label.setText(f"Loaded path: {path_data['name']}{path_type}")
            
            # Update spacing display
            self._update_spacing_display()
            
            napari.utils.notifications.show_info(f"Loaded {path_data['name']}{path_type}")
        except Exception as e:
            error_msg = f"Error loading path waypoints: {str(e)}"
            napari.utils.notifications.show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False