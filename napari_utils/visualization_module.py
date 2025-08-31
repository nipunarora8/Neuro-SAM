import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFrame, QListWidget, QListWidgetItem,
    QFileDialog
)
from qtpy.QtCore import Signal

class PathVisualizationWidget(QWidget):
    """Widget for managing and visualizing multiple paths"""
    
    # Define signals
    path_selected = Signal(str)  # path_id
    path_deleted = Signal(str)   # path_id
    
    def __init__(self, viewer, image, state):
        """Initialize the path visualization widget.
        
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
        
        self.xy_spacing_nm = self.state.get('xy_spacing_nm', 94.0)

        # Flag to prevent recursive event handling
        self.handling_event = False
        
        # Setup UI
        self.setup_ui()

    def update_pixel_spacing(self, new_spacing):
        """Update pixel spacing for visualization module"""
        self.pixel_spacing_nm = new_spacing
        print(f"Visualization: Updated pixel spacing to {new_spacing:.1f} nm/pixel")
        # Visualization module typically doesn't need parameter updates
        # but spacing info could be used for distance calculations in path analysis

    
    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        
        # Path list with instructions
        layout.addWidget(QLabel("Saved Paths (select to view or manipulate):"))
        self.path_list = QListWidget()
        self.path_list.setFixedHeight(120)
        self.path_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.path_list.itemSelectionChanged.connect(self.on_path_selection_changed)
        layout.addWidget(self.path_list)
        
        # Path management buttons
        path_buttons_layout = QHBoxLayout()
        path_buttons_layout.setSpacing(2)
        path_buttons_layout.setContentsMargins(2, 2, 2, 2)
        
        self.view_path_btn = QPushButton("View Selected Path")
        self.view_path_btn.setFixedHeight(22)
        self.view_path_btn.clicked.connect(self.view_selected_path)
        self.view_path_btn.setEnabled(False)
        path_buttons_layout.addWidget(self.view_path_btn)
        
        self.delete_path_btn = QPushButton("Delete Selected Path(s)")
        self.delete_path_btn.setFixedHeight(22)
        self.delete_path_btn.clicked.connect(self.delete_selected_paths)
        self.delete_path_btn.setEnabled(False)
        path_buttons_layout.addWidget(self.delete_path_btn)
        
        layout.addLayout(path_buttons_layout)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Path connection button
        self.connect_paths_btn = QPushButton("Connect Selected Paths")
        self.connect_paths_btn.setFixedHeight(22)
        self.connect_paths_btn.setToolTip("Select exactly 2 paths to connect them")
        self.connect_paths_btn.clicked.connect(self.connect_selected_paths)
        self.connect_paths_btn.setEnabled(False)
        layout.addWidget(self.connect_paths_btn)
        
        # Path visibility options
        visibility_layout = QHBoxLayout()
        visibility_layout.setSpacing(2)
        visibility_layout.setContentsMargins(2, 2, 2, 2)
        
        self.show_all_btn = QPushButton("Show All Paths")
        self.show_all_btn.setFixedHeight(22)
        self.show_all_btn.clicked.connect(lambda: self.set_paths_visibility(True))
        visibility_layout.addWidget(self.show_all_btn)
        
        self.hide_all_btn = QPushButton("Hide All Paths")
        self.hide_all_btn.setFixedHeight(22)
        self.hide_all_btn.clicked.connect(lambda: self.set_paths_visibility(False))
        visibility_layout.addWidget(self.hide_all_btn)
        
        layout.addLayout(visibility_layout)
        
        # Export button
        self.export_all_btn = QPushButton("Export All Paths")
        self.export_all_btn.setFixedHeight(22)
        self.export_all_btn.clicked.connect(self.export_all_paths)
        self.export_all_btn.setEnabled(False)
        layout.addWidget(self.export_all_btn)
        
        # Status message
        self.status_label = QLabel("")
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
            
            # Enable/disable export button
            self.export_all_btn.setEnabled(len(self.state['paths']) > 0)
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
            num_selected = len(selected_items)
            
            # Enable/disable buttons based on selection
            self.delete_path_btn.setEnabled(num_selected > 0)
            self.view_path_btn.setEnabled(num_selected == 1)
            self.connect_paths_btn.setEnabled(num_selected == 2)
        except Exception as e:
            napari.utils.notifications.show_info(f"Error handling selection change: {str(e)}")
        finally:
            self.handling_event = False
    
    def view_selected_path(self):
        """View the selected path from the list"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            if len(selected_items) != 1:
                return
                
            item = selected_items[0]
            path_id = item.data(100)
            
            # Emit signal that path is selected
            self.path_selected.emit(path_id)
            
            # Ensure the selected path's layer is visible
            if path_id in self.state['path_layers']:
                self.state['path_layers'][path_id].visible = True
            
            napari.utils.notifications.show_info(f"Viewing {self.state['paths'][path_id]['name']}")
        except Exception as e:
            napari.utils.notifications.show_info(f"Error viewing path: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
        finally:
            self.handling_event = False
    
    def delete_selected_paths(self):
        """Delete the currently selected paths"""
        selected_items = self.path_list.selectedItems()
        if not selected_items:
            napari.utils.notifications.show_info("No paths selected")
            return
        
        paths_deleted = []
        
        for item in selected_items:
            # Get the path ID
            path_id = item.data(100)
            
            if path_id in self.state['paths']:
                path_name = self.state['paths'][path_id]['name']
                
                # Remove the path layer from viewer
                if path_id in self.state['path_layers']:
                    self.viewer.layers.remove(self.state['path_layers'][path_id])
                    del self.state['path_layers'][path_id]
                
                # Remove corresponding segmentation layer if it exists
                seg_layer_name = f"Segmentation - {path_name}"
                for layer in list(self.viewer.layers):  # Create a copy of the list to safely modify during iteration
                    if layer.name == seg_layer_name:
                        self.viewer.layers.remove(layer)
                        if (
                            'segmentation_layer' in self.state and 
                            self.state['segmentation_layer'] is not None and 
                            self.state['segmentation_layer'].name == seg_layer_name
                        ):
                            self.state['segmentation_layer'] = None
                        napari.utils.notifications.show_info(f"Removed segmentation layer for {path_name}")
                        break
                
                # Remove corresponding spine layer if it exists
                spine_layer_name = f"Spines - {path_name}"
                for layer in list(self.viewer.layers):
                    if layer.name == spine_layer_name:
                        self.viewer.layers.remove(layer)
                        if path_id in self.state['spine_layers']:
                            del self.state['spine_layers'][path_id]
                        napari.utils.notifications.show_info(f"Removed spine layer for {path_name}")
                        break
                
                # Remove from dictionary
                del self.state['paths'][path_id]
                
                # Track deleted path
                paths_deleted.append(path_id)
                
                napari.utils.notifications.show_info(f"Deleted {path_name}")
        
        # Update traced path visualization if any paths were deleted
        if paths_deleted and self.image.ndim > 2 and self.state['traced_path_layer'] is not None:
            self._update_traced_path_visualization()
        
        # Update path list
        self.update_path_list()
        
        # Emit signal for each deleted path
        for path_id in paths_deleted:
            self.path_deleted.emit(path_id)
    
    def _update_traced_path_visualization(self):
        """Update the traced path visualization to reflect current paths"""
        if 'traced_path_layer' not in self.state or self.state['traced_path_layer'] is None:
            return
            
        if not self.state['paths']:
            # If no paths remain, clear the traced path layer
            self.state['traced_path_layer'].data = np.empty((0, self.image.ndim))
            self.state['traced_path_layer'].visible = False
            return
            
        # Create a comprehensive visualization of all paths in the traced layer
        all_traced_points = []
        
        # First, determine the full z-range for all paths
        min_z = float('inf')
        max_z = float('-inf')
        
        for path_id, path_data in self.state['paths'].items():
            if len(path_data['data']) > 0 and path_data['visible']:
                z_values = [point[0] for point in path_data['data']]
                path_min_z = int(min(z_values))
                path_max_z = int(max(z_values))
                
                min_z = min(min_z, path_min_z)
                max_z = max(max_z, path_max_z)
        
        # If we have valid z-range
        if min_z != float('inf') and max_z != float('-inf'):
            # For each frame in the full range
            for z in range(min_z, max_z + 1):
                # Add all paths to this frame
                for path_id, path_data in self.state['paths'].items():
                    if path_data['visible']:
                        for point in path_data['data']:
                            # Create a new point with the current frame's z-coordinate
                            new_point = point.copy()
                            new_point[0] = z  # Set the z-coordinate to the current frame
                            all_traced_points.append(new_point)
            
            # Update the traced path layer
            if all_traced_points:
                self.state['traced_path_layer'].data = np.array(all_traced_points)
                self.state['traced_path_layer'].visible = True
            else:
                self.state['traced_path_layer'].data = np.empty((0, self.image.ndim))
                self.state['traced_path_layer'].visible = False
    
    def set_paths_visibility(self, visible):
        """Set visibility of all saved path layers and update traced path visualization"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            # Show/hide individual path layers
            for path_id, layer in self.state['path_layers'].items():
                layer.visible = visible
                self.state['paths'][path_id]['visible'] = visible
            
            # Update traced path visualization only if we have a traced path layer
            if self.image.ndim > 2 and self.state['traced_path_layer'] is not None:
                if visible:
                    self._update_traced_path_visualization()
                else:
                    # Hide traced path layer when hiding all paths
                    self.state['traced_path_layer'].data = np.empty((0, self.image.ndim))
                    self.state['traced_path_layer'].visible = False
            
            action = "shown" if visible else "hidden"
            napari.utils.notifications.show_info(f"All paths {action}")
        except Exception as e:
            napari.utils.notifications.show_info(f"Error updating path visibility: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
        finally:
            self.handling_event = False
    
    def connect_selected_paths(self):
        """Connect two selected paths"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            
            if len(selected_items) != 2:
                napari.utils.notifications.show_info("Please select exactly two paths to connect")
                return
                
            # Get the path IDs
            path_id1 = selected_items[0].data(100)
            path_id2 = selected_items[1].data(100)
            
            if path_id1 not in self.state['paths'] or path_id2 not in self.state['paths']:
                napari.utils.notifications.show_info("Invalid path selection")
                return
                
            # Get the path data
            path1 = self.state['paths'][path_id1]
            path2 = self.state['paths'][path_id2]
            
            # Check if paths have start/end points
            if path1['start'] is None or path2['end'] is None:
                napari.utils.notifications.show_info("Both paths must have start and end points to connect them")
                return
                
            # Get start of path1 and end of path2
            start_point = path1['start']
            end_point = path2['end']
            
            napari.utils.notifications.show_info(f"Connecting {path1['name']} to {path2['name']}...")
            
            # Determine if we're doing 2D or 3D search
            is_same_frame = True
            if self.image.ndim > 2:
                is_same_frame = start_point[0] == end_point[0]
                
            # Import necessary classes
            import sys
            sys.path.append('../path_tracing/brightest-path-lib')
            from brightest_path_lib.algorithm import BidirectionalAStarSearch
                
            # Prepare points format based on 2D or 3D
            if is_same_frame and self.image.ndim > 2:
                # 2D case: use [y, x] format (ignore z)
                search_start = start_point[1:3]  # [y, x]
                search_end = end_point[1:3]      # [y, x]
                search_image = self.image[int(start_point[0])]
                napari.utils.notifications.show_info(f"Using 2D path search on frame {int(start_point[0])}")
            else:
                # 3D case or already 2D image: use full coordinates
                search_start = start_point
                search_end = end_point
                search_image = self.image
                napari.utils.notifications.show_info("Using 3D path search across frames")
                
            # Search for connecting path
            search_algorithm = BidirectionalAStarSearch(
                search_image, 
                start_point=search_start, 
                goal_point=search_end
            )
            
            connecting_path = search_algorithm.search()
            
            # If path found, create combined path
            if connecting_path is not None and len(connecting_path) > 0:
                # Fix coordinates if needed (2D case)
                if is_same_frame and self.image.ndim > 2:
                    z_val = start_point[0]
                    fixed_connecting_path = []
                    for point in connecting_path:
                        if len(point) == 2:  # [y, x]
                            fixed_connecting_path.append([z_val, point[0], point[1]])
                        else:
                            fixed_connecting_path.append(point)
                    connecting_path = fixed_connecting_path
                
                # Convert to numpy arrays
                path1_data = path1['data']
                path2_data = path2['data']
                connecting_data = np.array(connecting_path)
                
                # Create combined path
                combined_path = np.vstack([path1_data, connecting_data, path2_data])
                
                # Create a name for the combined path
                combined_name = f"{path1['name']} + {path2['name']}"
                
                # Get a color
                colors = ['cyan', 'magenta', 'green', 'blue', 'orange', 
                          'purple', 'teal', 'coral', 'gold', 'lavender']
                color_idx = len(self.state['paths']) % len(colors)
                combined_color = colors[color_idx]
                
                # Create a new layer
                combined_layer = self.viewer.add_points(
                    combined_path,
                    name=combined_name,
                    size=3,
                    face_color=combined_color,
                    opacity=0.7
                )
                
                # Generate a unique ID for this path
                import uuid
                path_id = str(uuid.uuid4())
                
                # Combine waypoints from both paths
                combined_waypoints = []
                if 'waypoints' in path1 and path1['waypoints']:
                    combined_waypoints.extend(path1['waypoints'])
                if 'waypoints' in path2 and path2['waypoints']:
                    combined_waypoints.extend(path2['waypoints'])
                
                # Store the combined path
                self.state['paths'][path_id] = {
                    'name': combined_name,
                    'data': combined_path,
                    'start': path1['start'].copy(),
                    'end': path2['end'].copy(),
                    'waypoints': combined_waypoints,
                    'visible': True,
                    'layer': combined_layer,
                    'original_clicks': [],  # Empty since this is a generated path
                    'smoothed': False  # Combined paths are not smoothed
                }
                
                # Store reference to the layer
                self.state['path_layers'][path_id] = combined_layer
                
                # Update the path list
                self.update_path_list()
                
                # Update traced path visualization
                if self.image.ndim > 2 and self.state['traced_path_layer'] is not None:
                    self._update_traced_path_visualization()
                
                # Set the new combined path as the current path
                self.state['current_path_id'] = path_id
                
                # Select the new path in the list
                for i in range(self.path_list.count()):
                    item = self.path_list.item(i)
                    if item.data(100) == path_id:
                        self.path_list.setCurrentItem(item)
                        break
                
                # Update UI
                msg = f"Connected {path1['name']} to {path2['name']} successfully"
                napari.utils.notifications.show_info(msg)
                self.status_label.setText(msg)
                
                # Import the signal from the path tracing widget to properly notify other modules
                from qtpy.QtCore import QTimer
                
                # Emit path_created signal to update other modules
                def emit_delayed_signal():
                    # Find the path tracing widget and emit its signal
                    parent_widget = self.parent()
                    while parent_widget:
                        if hasattr(parent_widget, 'path_tracing_widget'):
                            parent_widget.path_tracing_widget.path_created.emit(path_id, combined_name, combined_path)
                            break
                        parent_widget = parent_widget.parent()
                    
                    # Also emit our own signal
                    self.path_selected.emit(path_id)
                
                # Use a timer to emit the signal after the UI has updated
                QTimer.singleShot(100, emit_delayed_signal)
            else:
                error_msg = f"Could not find a path connecting {path1['name']} to {path2['name']}"
                napari.utils.notifications.show_info(error_msg)
                self.status_label.setText(error_msg)
        except Exception as e:
            error_msg = f"Error connecting paths: {str(e)}"
            napari.utils.notifications.show_info(error_msg)
            self.status_label.setText(error_msg)
            print(f"Error details: {str(e)}")
        finally:
            self.handling_event = False
            
    def export_all_paths(self):
        """Export all paths to a file"""
        if not self.state['paths']:
            napari.utils.notifications.show_info("No paths to export")
            return
        
        # Get path to save file
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save All Paths", "", "NumPy Files (*.npz)"
        )
        
        if not filepath:
            return
        
        try:
            # Prepare data for export
            path_data = {}
            for path_id, path_info in self.state['paths'].items():
                export_data = {
                    'points': path_info['data'],
                    'start': path_info['start'] if 'start' in path_info and path_info['start'] is not None else np.array([]),
                    'end': path_info['end'] if 'end' in path_info and path_info['end'] is not None else np.array([]),
                }
                
                # Include waypoints if available
                if 'waypoints' in path_info and path_info['waypoints']:
                    export_data['waypoints'] = np.array(path_info['waypoints'])
                
                path_data[path_info['name']] = export_data
            
            # Save as NumPy archive
            np.savez(filepath, paths=path_data)
            
            napari.utils.notifications.show_info(f"All paths saved to {filepath}")
            self.status_label.setText(f"Paths exported to {filepath}")
            
        except Exception as e:
            napari.utils.notifications.show_info(f"Error saving paths: {e}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def update_path_visualization(self):
        """Update the visualization of paths after modifications"""
        if self.state['traced_path_layer'] is not None:
            self._update_traced_path_visualization()