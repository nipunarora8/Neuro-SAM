import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFrame, QListWidget, QListWidgetItem,
    QProgressBar, QCheckBox, QSpinBox, QGroupBox
)
from qtpy.QtCore import Signal
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from contrasting_color_system import contrasting_color_manager
from segmentation_model import DendriteSegmenter
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, binary_closing, remove_small_objects


def fast_refine_dendrite_boundaries(image, dendrite_mask):
    """
    Fast, lightweight boundary refinement for dendrites
    Only basic morphological operations - no gradients
    """
    if not np.any(dendrite_mask):
        return dendrite_mask
    
    # Simple morphological cleanup only
    # Fill holes in dendrites (should be solid tubes)
    filled_mask = binary_fill_holes(dendrite_mask)
    
    # Light closing to connect nearby segments
    closed_mask = binary_closing(filled_mask, disk(2))
    
    # Remove very small objects
    labeled_mask, num_labels = ndimage.label(closed_mask)
    cleaned_mask = closed_mask.copy()
    
    for label_id in range(1, num_labels + 1):
        component = (labeled_mask == label_id)
        if np.sum(component) < 20:  # Small threshold for dendrites
            cleaned_mask[component] = 0
    
    return cleaned_mask.astype(np.uint8)


def fast_refine_dendrite_volume_boundaries(image_volume, mask_volume, brightest_path):
    """
    Fast dendrite boundary refinement - only process frames that have dendrite path
    """
    refined_volume = mask_volume.copy()
    
    # Only process frames that have dendrite path
    path_frames = set()
    for point in brightest_path:
        path_frames.add(int(point[0]))
    
    print(f"Fast dendrite light cleanup on {len(path_frames)} frames with dendrite...")
    
    for z in path_frames:
        if 0 <= z < image_volume.shape[0] and np.any(mask_volume[z]):
            refined_volume[z] = fast_refine_dendrite_boundaries(image_volume[z], mask_volume[z])
    
    # Report changes
    original_pixels = np.sum(mask_volume > 0)
    refined_pixels = np.sum(refined_volume > 0)
    change = refined_pixels - original_pixels
    
    print(f"Dendrite light cleanup: {original_pixels} -> {refined_pixels} pixels ({change:+d}, {change/original_pixels*100:+.1f}%)")
    
    return refined_volume.astype(np.uint8)


class SegmentationWidget(QWidget):
    """Widget for performing dendrite segmentation with boundary smoothing"""
    
    # Define signals
    segmentation_completed = Signal(str, str)  # path_id, layer_name
    
    def __init__(self, viewer, image, state):
        """Initialize the segmentation widget with boundary smoothing.
        
        Parameters:
        -----------
        viewer : napari.Viewer
            The napari viewer instance
        image : numpy.ndarray
            3D or higher-dimensional image data
        state : dict
            Shared state dictionary between modules
        """
        super().__init__()
        self.viewer = viewer
        self.image = image
        self.state = state
        
        # Initialize the segmentation model (don't load it yet)
        self.segmenter = None

        self.xy_spacing_nm = self.state.get('xy_spacing_nm', 94.0)
        
        # Flag to prevent recursive event handling
        self.handling_event = False
        
        # Setup UI
        self.setup_ui()

    def update_pixel_spacing(self, new_spacing):
        """Update pixel spacing for segmentation module"""
        self.pixel_spacing_nm = new_spacing
        print(f"Dendrite segmentation: Updated pixel spacing to {new_spacing:.1f} nm/pixel")
    
    def setup_ui(self):
        """Create the UI panel with boundary smoothing controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        
        # Model settings
        layout.addWidget(QLabel("<b>Dendrite Segmentation with Boundary Smoothing</b>"))
        layout.addWidget(QLabel("1. Load Segmentation Model\n2. Choose the path you want to segment\n3. Click on Run Segmentation to Segment"))
        layout.addWidget(QLabel("<i>Note: Uses overlapping patches + boundary smoothing to remove artifacts</i>"))
        
        # Model paths
        model_section = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(2)
        model_layout.setContentsMargins(5, 5, 5, 5)
        
        model_layout.addWidget(QLabel("Model Paths:"))
        self.model_path_edit = QLabel("SAM2 Model: checkpoints/sam2.1_hiera_small.pt")
        model_layout.addWidget(self.model_path_edit)
        
        self.config_path_edit = QLabel("Config: sam2.1_hiera_s.yaml")
        model_layout.addWidget(self.config_path_edit)
        
        self.weights_path_edit = QLabel("Weights: results/samv2_dendrite/dendrite_model.torch")
        model_layout.addWidget(self.weights_path_edit)
        
        model_section.setLayout(model_layout)
        layout.addWidget(model_section)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Path selection for segmentation
        layout.addWidget(QLabel("Select a path to segment:"))
        self.path_list = QListWidget()
        self.path_list.setFixedHeight(80)
        self.path_list.itemSelectionChanged.connect(self.on_path_selection_changed)
        layout.addWidget(self.path_list)
        
        # Segmentation parameters
        params_section = QGroupBox("Segmentation Parameters")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(2)
        params_layout.setContentsMargins(5, 5, 5, 5)
        
        params_layout.addWidget(QLabel("Segmentation Parameters:"))
        
        # Patch size
        patch_size_layout = QHBoxLayout()
        patch_size_layout.setSpacing(2)
        patch_size_layout.setContentsMargins(2, 2, 2, 2)
        patch_size_layout.addWidget(QLabel("Patch Size:"))
        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(64, 256)
        self.patch_size_spin.setSingleStep(32)
        self.patch_size_spin.setValue(128)  # Keep proven 128x128
        self.patch_size_spin.setToolTip("Size of overlapping patches (128x128 recommended)")
        patch_size_layout.addWidget(self.patch_size_spin)
        params_layout.addLayout(patch_size_layout)
        
        # Enable boundary cleanup for dendrites
        self.enable_boundary_smoothing_cb = QCheckBox("Enable Light Boundary Cleanup")
        self.enable_boundary_smoothing_cb.setChecked(False)  # Disabled by default for speed
        self.enable_boundary_smoothing_cb.setToolTip("Apply light morphological cleanup (hole filling, small object removal)")
        params_layout.addWidget(self.enable_boundary_smoothing_cb)
        
        # Dendrite structure enhancement
        self.enhance_dendrite_cb = QCheckBox("Enhance Tubular Dendrite Structure")
        self.enhance_dendrite_cb.setChecked(True)
        self.enhance_dendrite_cb.setToolTip("Apply morphological operations to connect dendrite segments and make tubular structure")
        params_layout.addWidget(self.enhance_dendrite_cb)
        
        # # Minimum dendrite size for noise removal
        # min_size_layout = QHBoxLayout()
        # min_size_layout.addWidget(QLabel("Min Dendrite Size (pixels):"))
        # self.min_dendrite_size_spin = QSpinBox()
        # self.min_dendrite_size_spin.setRange(50, 500)
        # self.min_dendrite_size_spin.setValue(100)
        # self.min_dendrite_size_spin.setToolTip("Minimum size of dendrite objects to keep (removes noise)")
        # min_size_layout.addWidget(self.min_dendrite_size_spin)
        # params_layout.addLayout(min_size_layout)
        
        # Frame range
        # self.use_full_volume_cb = QCheckBox("Process Full Volume")
        # self.use_full_volume_cb.setChecked(False)
        # self.use_full_volume_cb.setToolTip("Process entire volume instead of just path range")
        # params_layout.addWidget(self.use_full_volume_cb)
        
        # Processing method info
        # method_info = QLabel("Method: 50% overlapping patches + optional light cleanup")
        # method_info.setWordWrap(True)
        # method_info.setStyleSheet("color: #0066cc; font-style: italic;")
        # params_layout.addWidget(method_info)
        
        params_section.setLayout(params_layout)
        layout.addWidget(params_section)
        
        # Add separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)
        
        # Load model and run segmentation buttons
        self.load_model_btn = QPushButton("Load Dendrite Segmentation Model")
        self.load_model_btn.setFixedHeight(22)
        self.load_model_btn.clicked.connect(self.load_segmentation_model)
        layout.addWidget(self.load_model_btn)
        
        self.run_segmentation_btn = QPushButton("Run Dendrite Segmentation")
        self.run_segmentation_btn.setFixedHeight(22)
        self.run_segmentation_btn.clicked.connect(self.run_segmentation)
        self.run_segmentation_btn.setEnabled(False)  # Disabled until model is loaded
        layout.addWidget(self.run_segmentation_btn)
        
        # Export button
        self.export_dendrite_btn = QPushButton("Export Dendrite Masks")
        self.export_dendrite_btn.setFixedHeight(22)
        self.export_dendrite_btn.clicked.connect(self.export_dendrite_masks)
        self.export_dendrite_btn.setEnabled(False)  # Disabled until segmentation exists
        layout.addWidget(self.export_dendrite_btn)
        
        # Progress bar
        self.segmentation_progress = QProgressBar()
        self.segmentation_progress.setValue(0)
        layout.addWidget(self.segmentation_progress)
        
        # Color info display
        self.color_info_label = QLabel("")
        self.color_info_label.setWordWrap(True)
        layout.addWidget(self.color_info_label)
        
        # Status message
        self.status_label = QLabel("Status: Model not loaded")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
    
    def update_path_list(self):
        """Update the path list with current paths"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            # Clear current list
            self.path_list.clear()
            
            # Add paths to list
            for path_id, path_data in self.state['paths'].items():
                item = QListWidgetItem(path_data['name'])
                item.setData(100, path_id)  # Store path ID as custom data
                self.path_list.addItem(item)
            
            # Enable segmentation button if model is loaded and a path is selected
            self.run_segmentation_btn.setEnabled(
                self.segmenter is not None and 
                self.path_list.count() > 0 and
                self.path_list.currentRow() >= 0
            )
        except Exception as e:
            napari.utils.notifications.show_info(f"Error updating path list: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
        finally:
            self.handling_event = False
    
    def on_path_selection_changed(self):
        """Handle when path selection changes in the list"""
        # Prevent processing during updates
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            if len(selected_items) == 1:
                path_id = selected_items[0].data(100)
                if path_id in self.state['paths']:
                    # Store the selected path ID for segmentation
                    self.selected_path_id = path_id
                    path_name = self.state['paths'][path_id]['name']
                    
                    # Check if this path already has segmentation
                    seg_layer_name = f"Segmentation - {path_name}"
                    has_segmentation = False
                    for layer in self.viewer.layers:
                        if layer.name == seg_layer_name:
                            has_segmentation = True
                            break
                    
                    if has_segmentation:
                        self.status_label.setText(f"Status: Path '{path_name}' already has dendrite segmentation")
                    else:
                        self.status_label.setText(f"Status: Path '{path_name}' selected for dendrite segmentation")
                    
                    # Show color info if this path has an assigned color pair
                    color_info = contrasting_color_manager.get_pair_info(path_id)
                    if color_info:
                        self.color_info_label.setText(
                            f"Colors: Dendrite {color_info['dendrite_hex']} -> Spine {color_info['spine_hex']}"
                        )
                    else:
                        self.color_info_label.setText("Colors: Will be assigned during segmentation")
                    
                    # Enable the segmentation button if the model is loaded
                    if self.segmenter is not None:
                        self.run_segmentation_btn.setEnabled(True)
            else:
                # No path selected
                if hasattr(self, 'selected_path_id'):
                    delattr(self, 'selected_path_id')
                self.run_segmentation_btn.setEnabled(False)
                self.color_info_label.setText("")
        except Exception as e:
            napari.utils.notifications.show_info(f"Error handling dendrite segmentation path selection: {str(e)}")
        finally:
            self.handling_event = False
    
    def load_segmentation_model(self):
        """Load the segmentation model with overlapping patches"""
        try:
            # Update status
            self.status_label.setText("Status: Loading dendrite segmentation model...")
            self.load_model_btn.setEnabled(False)

            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            print(f"Device: {device}")
            
            # Initialize segmenter if not already done
            if self.segmenter is None:
                self.segmenter = DendriteSegmenter(
                    model_path="./Train-SAMv2/checkpoints/sam2.1_hiera_small.pt",
                    config_path="sam2.1_hiera_s.yaml",
                    weights_path="./Train-SAMv2/results/samv2_dendrite/dendrite_model.torch",
                    device=device
                )
            
            # Load the model
            success = self.segmenter.load_model()
            
            if success:
                self.status_label.setText("Status: Dendrite segmentation model loaded successfully!")
                self.run_segmentation_btn.setEnabled(len(self.state['paths']) > 0 and hasattr(self, 'selected_path_id'))
                napari.utils.notifications.show_info("Dendrite segmentation model loaded successfully")
            else:
                self.status_label.setText("Status: Failed to load model. Check console for errors.")
                self.load_model_btn.setEnabled(True)
                napari.utils.notifications.show_info("Failed to load dendrite segmentation model")
                
        except Exception as e:
            error_msg = f"Error loading dendrite segmentation model: {str(e)}"
            self.status_label.setText(f"Status: {error_msg}")
            self.load_model_btn.setEnabled(True)
            napari.utils.notifications.show_info(error_msg)
            print(f"Error details: {str(e)}")
    
    def run_segmentation(self):
        """Run dendrite segmentation with boundary smoothing on the selected path"""
        if self.segmenter is None:
            napari.utils.notifications.show_info("Please load the segmentation model first")
            return
        
        if len(self.state['paths']) == 0:
            napari.utils.notifications.show_info("Please create a path first")
            return
        
        try:
            # Get the selected path
            path_id = None
            if hasattr(self, 'selected_path_id'):
                path_id = self.selected_path_id
            else:
                # Fallback to the path selected in the list
                selected_items = self.path_list.selectedItems()
                if len(selected_items) == 1:
                    path_id = selected_items[0].data(100)
            
            if path_id is None or path_id not in self.state['paths']:
                napari.utils.notifications.show_info("Please select a path for segmentation")
                self.status_label.setText("Status: No path selected for segmentation")
                return
            
            # Get the path data
            path_data = self.state['paths'][path_id]
            path_name = path_data['name']
            brightest_path = path_data['data']
            
            # Get segmentation parameters
            patch_size = self.patch_size_spin.value()
            enable_boundary_smoothing = self.enable_boundary_smoothing_cb.isChecked()
            enhance_dendrite = self.enhance_dendrite_cb.isChecked()
            use_full_volume = False
            
            # Update UI
            enhancement_info = [f"{patch_size}x{patch_size} overlapping patches (50%)"]
            if enable_boundary_smoothing:
                enhancement_info.append("light cleanup")
            if enhance_dendrite:
                enhancement_info.append("tubular structure enhancement")
            
            enhancement_str = " + ".join(enhancement_info)
            
            self.status_label.setText(f"Status: Running dendrite segmentation on {path_name} with {enhancement_str}...")
            self.segmentation_progress.setValue(0)
            self.run_segmentation_btn.setEnabled(False)
            
            # Determine volume range
            if use_full_volume:
                start_frame = 0
                end_frame = len(self.image) - 1
            else:
                # Use the range from the path
                z_values = [point[0] for point in brightest_path]
                start_frame = int(min(z_values))
                end_frame = int(max(z_values))
            
            print(f"Segmenting dendrite path '{path_name}' from frame {start_frame} to {end_frame}")
            print(f"Path has {len(brightest_path)} points")
            print(f"Parameters: patch_size={patch_size}x{patch_size}, overlap=50% (stride={patch_size//2})")
            print(f"Light boundary cleanup: {enable_boundary_smoothing}")
            # print(f"Dendrite enhancement: {enhance_dendrite}, Min dendrite size: {min_dendrite_size} pixels")
            
            # Progress callback function
            def update_progress(current, total):
                if enable_boundary_smoothing:
                    progress = int((current / total) * 80)  # 0-80%
                else:
                    progress = int((current / total) * 90)  # 0-90%
                self.segmentation_progress.setValue(progress)
            
            # Try to run the segmentation with overlapping patches
            result_masks = self.segmenter.process_volume(
                image=self.image,
                brightest_path=brightest_path,
                start_frame=start_frame,
                end_frame=end_frame,
                patch_size=patch_size,
                progress_callback=update_progress
            )
            
            # Apply light boundary cleanup if requested
            if enable_boundary_smoothing and result_masks is not None:
                self.segmentation_progress.setValue(80)
                print("Applying light dendrite boundary cleanup...")
                napari.utils.notifications.show_info("Light dendrite cleanup...")
                
                refined_masks = fast_refine_dendrite_volume_boundaries(self.image, result_masks, brightest_path)
                result_masks = refined_masks
                self.segmentation_progress.setValue(90)
            
            # Process the results
            if result_masks is not None:
                # Ensure masks are binary (0 or 1)
                binary_masks = (result_masks > 0).astype(np.uint8)
                
                # Create or update the segmentation layer
                seg_layer_name = f"Segmentation - {path_name}"
                
                # Remove existing layer if it exists
                existing_layer = None
                for layer in self.viewer.layers:
                    if layer.name == seg_layer_name:
                        existing_layer = layer
                        break
                
                if existing_layer is not None:
                    print(f"Removing existing segmentation layer: {seg_layer_name}")
                    self.viewer.layers.remove(existing_layer)
                
                # Get the dendrite color from the contrasting color manager
                dendrite_color = contrasting_color_manager.get_dendrite_color(path_id)
                
                print(f"Adding new dendrite segmentation layer: {seg_layer_name}")
                print(f"Result masks shape: {binary_masks.shape}")
                print(f"Result masks type: {binary_masks.dtype}")
                print(f"Result masks min/max: {binary_masks.min()}/{binary_masks.max()}")
                print(f"Using dendrite color: {dendrite_color}")
                
                # Create the segmentation layer using add_image with proper colormap
                # Convert binary masks to float and scale to color range
                color_masks = binary_masks.astype(np.float32)
                color_masks[color_masks > 0] = 1.0  # Ensure binary values
                
                # Add as image layer
                segmentation_layer = self.viewer.add_image(
                    color_masks,
                    name=seg_layer_name,
                    opacity=0.7,
                    blending='additive',
                    colormap='viridis'  # Will be overridden
                )
                
                # Create custom colormap: [transparent, dendrite_color]
                custom_cmap = np.array([
                    [0, 0, 0, 0],  # Transparent for 0 values
                    [dendrite_color[0], dendrite_color[1], dendrite_color[2], 1]  # Color for 1 values
                ])
                
                # Apply the custom colormap
                segmentation_layer.colormap = custom_cmap
                
                # Set contrast limits to ensure proper color mapping
                segmentation_layer.contrast_limits = [0, 1]
                
                print(f"Applied custom colormap: {custom_cmap}")
                print(f"Layer contrast limits: {segmentation_layer.contrast_limits}")
                
                # Store reference in state
                self.state['segmentation_layer'] = segmentation_layer
                
                # Make sure the layer is visible
                segmentation_layer.visible = True
                
                # Update color info display
                color_info = contrasting_color_manager.get_pair_info(path_id)
                if color_info:
                    self.color_info_label.setText(
                        f"Colors: Dendrite {color_info['dendrite_hex']} -> Spine {color_info['spine_hex']}"
                    )
                
                # Enable export button
                self.export_dendrite_btn.setEnabled(True)
                
                # Update UI with segmentation information
                total_pixels = np.sum(binary_masks)
                
                result_text = f"Results: Dendrite segmentation completed - {total_pixels} pixels segmented"
                result_text += f"\nMethod: {enhancement_str}"
                result_text += f"\nOverlap: 50% (stride={patch_size//2})"
                # result_text += f"\nMin dendrite size: {min_dendrite_size} pixels"
                if enable_boundary_smoothing:
                    result_text += f"\nLight boundary cleanup applied"
                
                self.status_label.setText(result_text)
                
                napari.utils.notifications.show_info(f"Dendrite segmentation complete for {path_name}")
                
                # Emit signal that segmentation is completed
                self.segmentation_completed.emit(path_id, seg_layer_name)
            else:
                self.status_label.setText("Status: Dendrite segmentation failed. Check console for errors.")
                napari.utils.notifications.show_info("Dendrite segmentation failed")
        
        except Exception as e:
            error_msg = f"Error during dendrite segmentation: {str(e)}"
            self.status_label.setText(f"Status: {error_msg}")
            napari.utils.notifications.show_info(error_msg)
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.segmentation_progress.setValue(100)
            self.run_segmentation_btn.setEnabled(True)
    
    def export_dendrite_masks(self):
        """Export all dendrite segmentation masks"""
        from qtpy.QtWidgets import QFileDialog
        import tifffile
        import os
        from datetime import datetime
        
        try:
            # Check if there are any segmentation layers to export
            dendrite_layers = []
            for layer in self.viewer.layers:
                if hasattr(layer, 'name') and 'Segmentation -' in layer.name:
                    dendrite_layers.append(layer)
            
            if not dendrite_layers:
                napari.utils.notifications.show_info("No dendrite segmentation masks found to export")
                return
            
            # Get directory to save files
            save_dir = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Dendrite Masks", ""
            )
            
            if not save_dir:
                return
            
            # Create timestamp for this export session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            exported_count = 0
            
            for layer in dendrite_layers:
                try:
                    # Extract path name from layer name
                    path_name = layer.name.replace("Segmentation - ", "").replace(" ", "_")
                    
                    # Get the mask data
                    mask_data = layer.data
                    
                    # Convert to uint8 if needed
                    if mask_data.dtype != np.uint8:
                        mask_data = (mask_data > 0).astype(np.uint8) * 255
                    else:
                        mask_data = mask_data * 255  # Scale to 0-255 range
                    
                    # Create filename
                    filename = f"dendrite_mask_{path_name}_{timestamp}.tif"
                    filepath = os.path.join(save_dir, filename)
                    
                    # Save as TIFF
                    tifffile.imwrite(filepath, mask_data)
                    
                    exported_count += 1
                    print(f"Exported dendrite mask: {filepath}")
                    
                except Exception as e:
                    print(f"Error exporting mask for {layer.name}: {str(e)}")
                    continue
            
            if exported_count > 0:
                napari.utils.notifications.show_info(f"Successfully exported {exported_count} dendrite masks to {save_dir}")
                self.status_label.setText(f"Status: Exported {exported_count} dendrite masks")
            else:
                napari.utils.notifications.show_info("No dendrite masks were exported due to errors")
                
        except Exception as e:
            error_msg = f"Error during dendrite mask export: {str(e)}"
            napari.utils.notifications.show_info(error_msg)
            self.status_label.setText(f"Status: {error_msg}")
            print(f"Export error details: {str(e)}")
            import traceback
            traceback.print_exc()