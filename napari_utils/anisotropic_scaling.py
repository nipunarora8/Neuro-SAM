import numpy as np
from scipy.ndimage import zoom
import napari

class AnisotropicScaler:
    """
    Handles anisotropic scaling of 3D datasets with separate X, Y, Z spacing
    Similar to Neurotube's scaling system
    """
    
    def __init__(self, original_spacing_xyz=(94.0, 94.0, 500.0)):
        """
        Initialize with original voxel spacing in nanometers
        
        Args:
            original_spacing_xyz: Tuple of (x_spacing, y_spacing, z_spacing) in nm
        """
        self.original_spacing_xyz = np.array(original_spacing_xyz, dtype=float)
        self.current_spacing_xyz = self.original_spacing_xyz.copy()
        self.scale_factors = np.array([1.0, 1.0, 1.0])  # z, y, x order for numpy
        self.original_image = None
        self.scaled_image = None
        
    def set_spacing(self, x_nm, y_nm, z_nm):
        """
        Set new voxel spacing in nanometers
        
        Args:
            x_nm: X spacing in nanometers
            y_nm: Y spacing in nanometers  
            z_nm: Z spacing in nanometers
        """
        self.current_spacing_xyz = np.array([x_nm, y_nm, z_nm], dtype=float)
        
        # Calculate scale factors (z, y, x order for numpy arrays)
        # Scale factor = original_spacing / new_spacing
        self.scale_factors = np.array([
            self.original_spacing_xyz[2] / z_nm,  # Z scale factor
            self.original_spacing_xyz[1] / y_nm,  # Y scale factor  
            self.original_spacing_xyz[0] / x_nm   # X scale factor
        ])
        
        print(f"Updated spacing: X={x_nm:.1f}, Y={y_nm:.1f}, Z={z_nm:.1f} nm")
        print(f"Scale factors (Z,Y,X): {self.scale_factors}")
        
    def scale_image(self, image, order=1, prefilter=True):
        """
        Scale the image according to current spacing settings
        
        Args:
            image: Input 3D numpy array (Z, Y, X)
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
            prefilter: Whether to apply prefiltering for higher order interpolation
            
        Returns:
            Scaled image array
        """
        if self.original_image is None:
            self.original_image = image.copy()
            
        print(f"Scaling image from shape {image.shape} with factors {self.scale_factors}")
        
        # Apply scaling using scipy.ndimage.zoom
        scaled_image = zoom(image, self.scale_factors, order=order, prefilter=prefilter)
        
        print(f"Scaled image to shape {scaled_image.shape}")
        
        self.scaled_image = scaled_image
        return scaled_image
        
    def scale_coordinates(self, coordinates):
        """
        Scale coordinates from original space to scaled space
        
        Args:
            coordinates: Array of coordinates in (Z, Y, X) format
            
        Returns:
            Scaled coordinates
        """
        coords = np.array(coordinates)
        if coords.ndim == 1:
            # Single coordinate
            return coords * self.scale_factors
        else:
            # Multiple coordinates
            return coords * self.scale_factors[np.newaxis, :]
    
    def scale_coordinates_between_spacings(self, coordinates, from_spacing_xyz, to_spacing_xyz):
        """
        Scale coordinates from one spacing to another spacing
        
        Args:
            coordinates: Array of coordinates in (Z, Y, X) format
            from_spacing_xyz: Source spacing (x, y, z) in nm
            to_spacing_xyz: Target spacing (x, y, z) in nm
            
        Returns:
            Scaled coordinates
        """
        # Calculate scale factors between the two spacings
        # Scale factor = from_spacing / to_spacing
        scale_factors = np.array([
            from_spacing_xyz[2] / to_spacing_xyz[2],  # Z scale factor
            from_spacing_xyz[1] / to_spacing_xyz[1],  # Y scale factor  
            from_spacing_xyz[0] / to_spacing_xyz[0]   # X scale factor
        ])
        
        coords = np.array(coordinates)
        if coords.ndim == 1:
            # Single coordinate
            return coords * scale_factors
        else:
            # Multiple coordinates
            return coords * scale_factors[np.newaxis, :]
    
    def scale_mask(self, mask, target_shape, order=0):
        """
        Scale a mask to target shape using appropriate interpolation
        
        Args:
            mask: Input mask array
            target_shape: Target shape tuple
            order: Interpolation order (0 for masks to preserve binary values)
            
        Returns:
            Scaled mask
        """
        if mask.shape == target_shape:
            return mask
            
        # Calculate scale factors for this specific scaling
        scale_factors = np.array(target_shape) / np.array(mask.shape)
        
        # Use nearest neighbor for masks to preserve binary values
        scaled_mask = zoom(mask, scale_factors, order=order, prefilter=False)
        
        # Ensure binary values for segmentation masks
        if order == 0:
            scaled_mask = (scaled_mask > 0.5).astype(mask.dtype)
        
        return scaled_mask
            
    def unscale_coordinates(self, scaled_coordinates):
        """
        Convert coordinates from scaled space back to original space
        
        Args:
            scaled_coordinates: Array of coordinates in scaled space
            
        Returns:
            Original space coordinates
        """
        coords = np.array(scaled_coordinates)
        if coords.ndim == 1:
            # Single coordinate
            return coords / self.scale_factors
        else:
            # Multiple coordinates
            return coords / self.scale_factors[np.newaxis, :]
            
    def get_effective_spacing(self):
        """
        Get the effective voxel spacing in the scaled image
        
        Returns:
            Tuple of (x_spacing, y_spacing, z_spacing) in nm for the scaled image
        """
        return tuple(self.current_spacing_xyz)
        
    def get_scale_factors(self):
        """
        Get current scale factors in (Z, Y, X) order
        
        Returns:
            Array of scale factors
        """
        return self.scale_factors.copy()
        
    def reset_to_original(self):
        """Reset scaling to original spacing"""
        self.current_spacing_xyz = self.original_spacing_xyz.copy()
        self.scale_factors = np.array([1.0, 1.0, 1.0])
        
    def get_volume_ratio(self):
        """
        Get the volume ratio between scaled and original image
        
        Returns:
            Volume ratio (scaled_volume / original_volume)
        """
        return np.prod(self.scale_factors)


class ScalingWidget:
    """
    UI widget for controlling anisotropic scaling
    """
    
    def __init__(self, viewer, scaler, update_callback=None):
        """
        Initialize scaling widget
        
        Args:
            viewer: Napari viewer instance
            scaler: AnisotropicScaler instance
            update_callback: Function to call when scaling changes
        """
        self.viewer = viewer
        self.scaler = scaler
        self.update_callback = update_callback
        
    def create_scaling_controls(self, layout):
        """
        Add scaling controls to a layout
        
        Args:
            layout: QT layout to add controls to
        """
        from qtpy.QtWidgets import (QLabel, QDoubleSpinBox, QHBoxLayout, 
                                   QPushButton, QGroupBox, QVBoxLayout, QCheckBox)
        
        # Scaling section
        scaling_group = QGroupBox("Anisotropic Voxel Spacing (Neurotube-style)")
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
        from qtpy.QtWidgets import QComboBox
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
        self.apply_scaling_btn.clicked.connect(self._apply_scaling)
        button_layout.addWidget(self.apply_scaling_btn)
        
        self.reset_scaling_btn = QPushButton("Reset to Original")
        self.reset_scaling_btn.setToolTip("Reset to original voxel spacing")
        self.reset_scaling_btn.clicked.connect(self._reset_scaling)
        button_layout.addWidget(self.reset_scaling_btn)
        
        scaling_layout.addLayout(button_layout)
        
        # Status info
        self.scaling_status = QLabel("Status: Original spacing")
        self.scaling_status.setWordWrap(True)
        scaling_layout.addWidget(self.scaling_status)
        
        scaling_group.setLayout(scaling_layout)
        layout.addWidget(scaling_group)
        
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
            
            # Call update callback if provided
            if self.update_callback:
                self.update_callback(order)
                
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
        
        # Call update callback if provided
        if self.update_callback:
            self.update_callback(1)  # Linear interpolation for reset
            
        napari.utils.notifications.show_info("Reset to original voxel spacing")


# Example integration with existing NeuroSAM system
class ScaledNeuroSAMWidget:
    """
    Extended NeuroSAM widget with anisotropic scaling support
    """
    
    def __init__(self, viewer, original_image, original_spacing_xyz=(94.0, 94.0, 500.0)):
        """
        Initialize with anisotropic scaling support
        
        Args:
            viewer: Napari viewer
            original_image: Original image data
            original_spacing_xyz: Original spacing in (x, y, z) nanometers
        """
        self.viewer = viewer
        self.original_image = original_image
        self.current_image = original_image.copy()
        
        # Initialize scaler
        self.scaler = AnisotropicScaler(original_spacing_xyz)
        
        # Initialize scaling widget
        self.scaling_widget = ScalingWidget(
            viewer=self.viewer,
            scaler=self.scaler,
            update_callback=self._on_scaling_update
        )
        
        # Track the main image layer
        self.image_layer = None
        
    def _on_scaling_update(self, interpolation_order):
        """
        Handle when scaling is updated
        
        Args:
            interpolation_order: Interpolation order for scaling
        """
        try:
            # Scale the image
            scaled_image = self.scaler.scale_image(
                self.original_image, 
                order=interpolation_order
            )
            
            # Update current image
            self.current_image = scaled_image
            
            # Update the napari layer
            if self.image_layer is not None:
                # Update existing layer
                self.image_layer.data = scaled_image
                self.image_layer.name = f"Image (scaled: {self.scaler.current_spacing_xyz[0]:.1f}, {self.scaler.current_spacing_xyz[1]:.1f}, {self.scaler.current_spacing_xyz[2]:.1f} nm)"
            else:
                # Create new layer
                self.image_layer = self.viewer.add_image(
                    scaled_image,
                    name=f"Image (scaled: {self.scaler.current_spacing_xyz[0]:.1f}, {self.scaler.current_spacing_xyz[1]:.1f}, {self.scaler.current_spacing_xyz[2]:.1f} nm)",
                    colormap='gray'
                )
            
            # Clear existing paths/segmentations since they're now invalid
            self._clear_analysis_layers()
            
            print(f"Updated image to shape {scaled_image.shape} with spacing {self.scaler.current_spacing_xyz}")
            
        except Exception as e:
            napari.utils.notifications.show_info(f"Error updating scaled image: {str(e)}")
            print(f"Scaling update error: {str(e)}")
            
    def _clear_analysis_layers(self):
        """Clear path tracing and segmentation layers when scaling changes"""
        layers_to_remove = []
        
        for layer in self.viewer.layers:
            layer_name = layer.name.lower()
            if any(keyword in layer_name for keyword in [
                'path', 'segmentation', 'spine', 'waypoint', 'traced'
            ]):
                layers_to_remove.append(layer)
        
        for layer in layers_to_remove:
            self.viewer.layers.remove(layer)
            
        napari.utils.notifications.show_info("Cleared analysis layers due to scaling change")
        
    def get_current_image(self):
        """Get the currently scaled image"""
        return self.current_image
        
    def get_current_spacing(self):
        """Get current voxel spacing in (x, y, z) format"""
        return self.scaler.get_effective_spacing()
        
    def scale_coordinates_to_original(self, coordinates):
        """
        Convert coordinates from current scaled space to original image space
        Useful for saving results that reference the original image
        """
        return self.scaler.unscale_coordinates(coordinates)
        
    def scale_coordinates_from_original(self, coordinates):
        """
        Convert coordinates from original image space to current scaled space
        Useful for loading previous results
        """
        return self.scaler.scale_coordinates(coordinates)