
import os
import time
import numpy as np
import torch
import tifffile as tiff
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, 
    QDoubleSpinBox, QFileDialog, QProgressBar, QGroupBox, QFormLayout
)
from qtpy.QtCore import Qt, QThread, Signal
from scipy.ndimage import label as cc_label

import napari
from napari.qt.threading import thread_worker

# Import the model class
# Assumes installed package structure
from neuro_sam.punet.punet_inference import run_inference_volume




from neuro_sam.utils import get_weights_path

class PunetSpineSegmentationWidget(QWidget):
    """
    Widget for spine segmentation using Probabilistic U-Net.
    Replaces the old SpineDetection and SpineSegmentation widgets.
    """
    progress_signal = Signal(float)

    def __init__(self, viewer, image, state):
        super().__init__()
        self.viewer = viewer
        self.image = image  # This is the currently loaded image (could be cropped/scaled)
        self.state = state
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = get_weights_path("punet_best.pth") # Auto-download default weights
        
        # Connect custom progress signal
        self.progress_signal.connect(self._on_worker_progress)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # --- Model Section ---
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        self.lbl_model = QLabel(f"Weights: {os.path.basename(self.model_path)}")
        self.lbl_model.setWordWrap(True)
        model_layout.addWidget(self.lbl_model)
        
        btn_load_model = QPushButton("Select Weights File")
        btn_load_model.clicked.connect(self._select_model_file)
        model_layout.addWidget(btn_load_model)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # --- Parameters Section ---
        param_group = QGroupBox("Inference Parameters")
        form_layout = QFormLayout()
        
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(1, 100)
        self.spin_samples.setValue(8)
        self.spin_samples.setToolTip("Number of Monte Carlo samples per slice")
        form_layout.addRow("MC Samples:", self.spin_samples)
        
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0.1, 10.0)
        self.spin_temp.setSingleStep(0.1)
        self.spin_temp.setValue(1.4)
        self.spin_temp.setToolTip("Temperature scaling (higher = softer)")
        form_layout.addRow("Temperature:", self.spin_temp)
        
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.01, 0.99)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(0.5)
        self.spin_threshold.setToolTip("Probability threshold for binary mask")
        form_layout.addRow("Threshold:", self.spin_threshold)
        
        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(0, 1000)
        self.spin_min_size.setValue(40)
        self.spin_min_size.setToolTip("Minimum object size in voxels")
        form_layout.addRow("Min Size (vox):", self.spin_min_size)

        param_group.setLayout(form_layout)
        layout.addWidget(param_group)
        
        # --- Run Section ---
        self.btn_run = QPushButton("Run Spine Segmentation")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.btn_run.clicked.connect(self._run_segmentation)
        layout.addWidget(self.btn_run)
        
        # New Buttons Layout
        btn_layout = QHBoxLayout()
        
        self.btn_toggle_view = QPushButton("Show Full Stack")
        self.btn_toggle_view.setCheckable(True)
        self.btn_toggle_view.clicked.connect(self.toggle_view)
        self.btn_toggle_view.setEnabled(False) # Disabled until inference runs
        btn_layout.addWidget(self.btn_toggle_view)
        
        self.btn_export = QPushButton("Export Spines")
        self.btn_export.clicked.connect(self.export_spines)
        self.btn_export.setEnabled(False) # Disabled until inference runs
        btn_layout.addWidget(self.btn_export)
        
        layout.addLayout(btn_layout)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
    def _select_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Prob U-Net Weights", "", "PyTorch Models (*.pth *.pt)"
        )
        if file_path:
            self.model_path = file_path
            self.lbl_model.setText(f"Weights: {os.path.basename(file_path)}")

    def _run_segmentation(self):
        if not os.path.exists(self.model_path):
            # Check relative to current working dir
            abs_path = os.path.abspath(self.model_path)
            if not os.path.exists(abs_path):
                napari.utils.notifications.show_error(f"Model file not found: {self.model_path}")
                return
            self.model_path = abs_path
            
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0) # Indeterminate while loading model
        self.status_label.setText("Loading model...")
        
        # Parameters
        params = {
            'weights': self.model_path,
            'samples': self.spin_samples.value(),
            'temp': self.spin_temp.value(),
            'thr': self.spin_threshold.value(),
            'min_size': self.spin_min_size.value(),
            'device': self.device
        }
        
        # Image data to process
        # Ensure we use the current image from the viewer/state
        # Note: self.image passed in __init__ might be stale if updated elsewhere, 
        # but typically main_widget passes the main volume.
        # Let's ensure dimensions.
        vol = self.state.get('current_image_data', self.image) # Fallback to init image
        
        # If the viewer has a layer named "Image ...", use that data instead of init data
        # to ensure we seg on what's visible
        # (Assuming main_widget handles 'current_image' updates correctly)
        
        # In main_widget, self.current_image is updated.
        vol = self.image # Using the reference object usually works if it's mutable, but numpy arrays aren't
        # Better: get it freshly from main widget reference? 
        # Actually main_widget passed 'self.current_image' which is an array. Arrays are passed by reference?
        # No, self.current_image = image.copy() in main_widget.
        # But we can access the viewer's active image layer if needed.
        # For now, let's assume valid volume.
        
        if vol.ndim == 2:
            vol = vol[np.newaxis, ...]
        
        worker = self._segmentation_worker(vol, params)
        worker.yielded.connect(self._on_worker_progress)
        worker.returned.connect(self._on_worker_finished)
        worker.errored.connect(self._on_worker_error)
        worker.start()

    @thread_worker
    def _segmentation_worker(self, vol, params):
        import traceback
        try:
            # Import the refactored inference function from the package
            # Import the refactored inference function from the package
            try:
                from neuro_sam.punet.punet_inference import run_inference_volume
            except ImportError:
                # Fallback should not be needed with proper package execution
                raise ImportError("Could not import run_inference_volume from neuro_sam.punet.punet_inference")
            
            yield "Starting inference..."
            
            # Callback to update progress bar from worker thread
            def progress_cb(val):
                self.progress_signal.emit(val)
            
            # Call the shared inference function
            # Note: We pass verbose=False to avoid printing to stdout, 
            # but we can capture progress if we modify the library. 
            # For now, it will just run and block this thread until done.
            results = run_inference_volume(
                image_input=vol,
                weights_path=params['weights'],
                device=str(params['device']),
                samples=params['samples'],
                posterior=False,
                temperature=params['temp'],
                threshold=params['thr'],
                min_size_voxels=params['min_size'],
                verbose=True,
                progress_callback=progress_cb
            )
            
            yield "Processing results..."
            
            # The widget expects 'mask_spine' which is in the results dict
            return results
            
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def _on_worker_progress(self, data):
        if isinstance(data, str):
            self.status_label.setText(data)
            # If "Starting", switch to determinate mode
            if "Starting" in data:
                self.progress.setRange(0, 100)
                self.progress.setValue(0)
        elif isinstance(data, float):
            self.progress.setValue(int(data * 100))

    def _on_worker_finished(self, results):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("Finished.")
        self.full_spine_mask = results['mask_spine'] # Store global mask
        
        # Enable new buttons
        self.btn_toggle_view.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_toggle_view.setChecked(False) # Default to local view
        self.btn_toggle_view.setText("Show Full Stack")
        self.showing_full_stack = False
        
        # Refresh layers based on existing dendrite segmentations
        self.refresh_spine_layers()
        
        napari.utils.notifications.show_info("Spine Segmentation Complete. Updating per-path layers.")

    def toggle_view(self):
        if not hasattr(self, 'full_spine_mask') or self.full_spine_mask is None:
            return
            
        if self.btn_toggle_view.isChecked():
            # Switch to Global View
            self.btn_toggle_view.setText("Show Filtered")
            self.showing_full_stack = True
            
            # 1. Remove all local path layers
            layers_to_remove = []
            for layer in self.viewer.layers:
                if layer.name.startswith("Spine Segmentation - Path"):
                    layers_to_remove.append(layer)
            for layer in layers_to_remove:
                self.viewer.layers.remove(layer)
                
            # 2. Add Global Layer
            display_mask = self.full_spine_mask.astype(np.float32)
            display_mask[display_mask > 0] = 1.0
            
            layer = self.viewer.add_image(
                display_mask,
                name="Global Spine Segmentation",
                opacity=0.8,
                blending='additive',
                colormap='viridis'
            )
            # Custom colormap: 0=Transparent, 1=Neon Green
            custom_cmap = np.array([[0, 0, 0, 0], [0.1, 1.0, 0.1, 1.0]])
            layer.colormap = custom_cmap
            layer.contrast_limits = [0, 1]
            
        else:
            # Switch to Local View
            self.btn_toggle_view.setText("Show Full Stack")
            self.showing_full_stack = False
            
            # 1. Remove Global Layer
            for layer in self.viewer.layers:
                if layer.name == "Global Spine Segmentation":
                    self.viewer.layers.remove(layer)
                    break
            
            # 2. Restore Local Layers
            self.refresh_spine_layers()

    def export_spines(self):
        if not hasattr(self, 'full_spine_mask') or self.full_spine_mask is None:
            napari.utils.notifications.show_error("No spines to export!")
            return
            
        import tifffile
        from qtpy.QtWidgets import QFileDialog
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Spine Segmentation", "spine_segmentation.tif", 
            "TIFF Files (*.tif *.tiff);;All Files (*)", options=options
        )
        
        if file_path:
            try:
                tifffile.imwrite(file_path, self.full_spine_mask)
                napari.utils.notifications.show_info(f"Saved spine segmentation to {file_path}")
            except Exception as e:
                napari.utils.notifications.show_error(f"Failed to save file: {e}")

    def refresh_spine_layers(self):
        """
        Filter global spine mask by dendrite segments and show/update layers
        for each path that has a dendrite mask.
        """
        if not hasattr(self, 'full_spine_mask') or self.full_spine_mask is None:
            return
            
        # If we are in "Full Stack" mode, logic is paused/ignored until user switches back
        if hasattr(self, 'showing_full_stack') and self.showing_full_stack:
            return

        from scipy.ndimage import distance_transform_edt
        
        # We need to find dendrite masks for each path
        for path_id, path_data in self.state['paths'].items():
            path_name = path_data['name']
            seg_layer_name = f"Segmentation - {path_name}"
            
            # Find the dendrite layer
            dendrite_layer = None
            for layer in self.viewer.layers:
                if layer.name == seg_layer_name:
                    dendrite_layer = layer
                    break
            
            if dendrite_layer is None:
                continue # No dendrite segmentation for this path yet
                
            # Get dendrite mask (it might be float if added with add_image, usually 0 or 1)
            dendrite_data = dendrite_layer.data
            binary_dendrite = (dendrite_data > 0)
            
            # Use Distance Transform to create a broad "capture zone" around the dendrite
            # distance_transform_edt calculates distance to the nearest ZERO value.
            # So we invert the mask: Dendrite=0, Background=1.
            # Result: Distance from nearest dendrite pixel.
            dist_map = distance_transform_edt(np.logical_not(binary_dendrite))
            
            # Capture radius in pixels. 25 pixels ~ 2.5 microns (at 0.1um/px)
            # This is large enough to capture even long spines.
            capture_radius = 25 
            capture_mask = (dist_map <= capture_radius)
            
            # Mask the global spine prediction with this broad capture zone
            filtered_spine = self.full_spine_mask & capture_mask
            
            # Prepare for display (float for add_image with alpha)
            display_mask = filtered_spine.astype(np.float32)
            display_mask[display_mask > 0] = 1.0
            
            spine_layer_name = f"Spine Segmentation - {path_name}"
            
            # Remove existing spine layer if present
            existing_spine_layer = None
            for layer in self.viewer.layers:
                if layer.name == spine_layer_name:
                    existing_spine_layer = layer
                    break
            
            if existing_spine_layer:
                self.viewer.layers.remove(existing_spine_layer)
                
            # Display
            if np.any(display_mask):
                layer = self.viewer.add_image(
                    display_mask,
                    name=spine_layer_name,
                    opacity=0.8,
                    blending='additive',
                    colormap='viridis' # Dummy, overridden below
                )
                
                # Green spines (transparent background)
                color = np.array([0, 1, 0, 1])
                # Inherit or pick color? User requested neon green in prev logic
                
                custom_cmap = np.array([
                    [0, 0, 0, 0],   # Transparent
                    color           # Green
                ])
                layer.colormap = custom_cmap
                layer.contrast_limits = [0, 1]

    def _on_worker_error(self, err):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("Error occurred.")
        napari.utils.notifications.show_error(f"Segmentation failed: {err}")
