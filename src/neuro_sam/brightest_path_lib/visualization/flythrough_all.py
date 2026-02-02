import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from neuro_sam.brightest_path_lib.algorithm import WaypointBidirectionalAStarSearch

def create_integrated_flythrough(image, start_point, goal_point, waypoints=None, 
                               output_file='integrated_flythrough.mp4', fps=15,
                               zoom_size=50, field_of_view=50, view_distance=5, 
                               reference_image=None, reference_cmap='viridis', 
                               reference_alpha=0.7):
    """
    Create an integrated fly-through visualization along the brightest path in a 3D image
    with full slice on top and zoomed/tube views below, optionally using a reference image
    
    Parameters:
    -----------
    image : numpy.ndarray
        The 3D image data (z, y, x) used to find the path
    start_point : array-like
        Starting coordinates [z, y, x]
    goal_point : array-like
        Goal coordinates [z, y, x]
    waypoints : list of array-like, optional
        List of waypoints to include in the path
    output_file : str
        Filename for the output animation
    fps : int
        Frames per second for the animation
    zoom_size : int
        Size of the zoomed patch in pixels
    field_of_view : int
        Width of the field of view in degrees for the tube view
    view_distance : int
        How far ahead to look along the path for the tube view
    reference_image : numpy.ndarray, optional
        Optional reference image (can be 3D, same size as image, or 3D with 3 channels)
        to overlay or use for visualization while showing the path from the main image
    reference_cmap : str, optional
        Colormap to use for the reference image
    reference_alpha : float, optional
        Alpha (transparency) value for the reference image overlay
    """
    # Validate and process reference image if provided
    if reference_image is not None:
        # Check if reference image has same z,y,x dimensions as main image
        if reference_image.shape[0] != image.shape[0]:
            raise ValueError("Reference image must have same number of z-slices as main image")
        
        # Handle multi-channel reference images
        is_multichannel = False
        if len(reference_image.shape) == 4:  # [z, y, x, channels]
            is_multichannel = True
            if reference_image.shape[1:3] != image.shape[1:3]:
                raise ValueError("Reference image must have same y,x dimensions as main image")
        elif reference_image.shape != image.shape:
            raise ValueError("Reference image must have same dimensions as main image")
    
    # Run the brightest path algorithm
    astar = WaypointBidirectionalAStarSearch(
        image=image,
        start_point=np.array(start_point),
        goal_point=np.array(goal_point),
        waypoints=waypoints if waypoints else None
    )
    
    path = astar.search(verbose=True)
    
    if not astar.found_path:
        raise ValueError("Could not find a path through the image")
    
    # Convert path to numpy array for easier manipulation
    path_array = np.array(path)
    
    # Pre-compute tangent vectors (direction of travel) for each point in the path
    tangent_vectors = []
    window_size = view_distance  # Points to look ahead/behind
    
    for i in range(len(path)):
        # Get window of points around current position
        start_idx = max(0, i - window_size)
        end_idx = min(len(path), i + window_size + 1)
        
        if end_idx - start_idx < 2:  # Need at least 2 points
            # If at the start/end, use the direction to the next/previous point
            if i == 0:
                tangent = path[1] - path[0]
            else:
                tangent = path[i] - path[i-1]
        else:
            # Fit a line to the window of points and use its direction
            window_points = path_array[start_idx:end_idx]
            
            # Simple approach: use the direction from first to last point in window
            tangent = window_points[-1] - window_points[0]
            
        # Normalize the tangent vector
        norm = np.linalg.norm(tangent)
        if norm > 0:
            tangent = tangent / norm
        
        tangent_vectors.append(tangent)
    
    # Create figure for visualization with a top row and bottom row
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 2], width_ratios=[1, 2])
    
    # Create three subplots
    ax_slice = fig.add_subplot(gs[0, :])  # Top row: Full slice view (spans both columns)
    ax_zoom = fig.add_subplot(gs[1, 0])   # Bottom left: Zoomed patch view
    ax_tube = fig.add_subplot(gs[1, 1])   # Bottom right: Tube/first-person view
    
    # Function to update the plot for each frame
    def update(frame_idx):
        # Clear all axes
        ax_slice.clear()
        ax_zoom.clear()
        ax_tube.clear()
        
        # Get current position in the path
        current_idx = min(frame_idx, len(path) - 1)
        current_point = path[current_idx]
        current_tangent = tangent_vectors[current_idx]
        
        # Convert to integers for indexing
        z, y, x = np.round(current_point).astype(int)
        
        # Ensure we're within image bounds
        z = np.clip(z, 0, image.shape[0] - 1)
        y = np.clip(y, 0, image.shape[1] - 1)
        x = np.clip(x, 0, image.shape[2] - 1)
        
        #------------------ Full Slice View (Top) ------------------#
        # Display the current slice from the image stack
        ax_slice.imshow(image[z], cmap='gray')
        
        # Overlay reference image if provided
        if reference_image is not None:
            if is_multichannel:
                # For multi-channel reference, create an RGB overlay
                ref_slice = reference_image[z]
                # Normalize for RGB display if needed
                if ref_slice.max() > 1:
                    ref_slice = ref_slice / 255.0
                ax_slice.imshow(ref_slice, alpha=reference_alpha)
            else:
                # For single-channel reference, use the provided colormap
                ax_slice.imshow(reference_image[z], cmap=reference_cmap, alpha=reference_alpha)
        
        # Plot the path projection on this slice
        # Find all path points on the current z-slice
        slice_points = path_array[path_array[:, 0] == z]
        if len(slice_points) > 0:
            ax_slice.plot(slice_points[:, 2], slice_points[:, 1], 'r-', linewidth=2)
        
        # Mark the current position
        ax_slice.scatter(x, y, c='red', s=100, marker='o')
        
        # Plot "shadows" of the path from other slices (lighter color)
        # Points from slices above current position
        above_points = path_array[path_array[:, 0] < z]
        if len(above_points) > 0:
            ax_slice.plot(above_points[:, 2], above_points[:, 1], 'r-', alpha=0.3, linewidth=1)
            
        # Points from slices below current position
        below_points = path_array[path_array[:, 0] > z]
        if len(below_points) > 0:
            ax_slice.plot(below_points[:, 2], below_points[:, 1], 'r-', alpha=0.3, linewidth=1)
        
        # Add a rectangle showing the zoomed area
        half_size = zoom_size // 2
        zoom_rect = plt.Rectangle((x - half_size, y - half_size), 
                                zoom_size, zoom_size,
                                fill=False, edgecolor='yellow', linewidth=2)
        ax_slice.add_patch(zoom_rect)
        
        ax_slice.set_title(f'Slice Z={z} - Frame {frame_idx+1}/{len(path)}')
        ax_slice.set_xlabel('X')
        ax_slice.set_ylabel('Y')
        
        #------------------ Zoomed Patch View (Bottom Left) ------------------#
        # Get coordinates for the zoom window, handling edges of the image
        y_min = max(0, y - half_size)
        y_max = min(image.shape[1], y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(image.shape[2], x + half_size)
        
        # Extract the patch from the main image
        zoom_patch = image[z, y_min:y_max, x_min:x_max]
        
        # Display zoomed patch
        ax_zoom.imshow(zoom_patch, cmap='gray')
        
        # Overlay reference image in zoomed view if provided
        if reference_image is not None:
            if is_multichannel:
                # Extract the patch from the reference image (RGB)
                ref_zoom_patch = reference_image[z, y_min:y_max, x_min:x_max]
                # Normalize for RGB display if needed
                if ref_zoom_patch.max() > 1:
                    ref_zoom_patch = ref_zoom_patch / 255.0
                ax_zoom.imshow(ref_zoom_patch, alpha=reference_alpha)
            else:
                # Extract the patch from the reference image (single channel)
                ref_zoom_patch = reference_image[z, y_min:y_max, x_min:x_max]
                ax_zoom.imshow(ref_zoom_patch, cmap=reference_cmap, alpha=reference_alpha)
        
        # Find path points within this zoomed patch and transform coordinates
        patch_slice_points = slice_points[
            (slice_points[:, 1] >= y_min) & (slice_points[:, 1] < y_max) &
            (slice_points[:, 2] >= x_min) & (slice_points[:, 2] < x_max)
        ] if len(slice_points) > 0 else np.array([])
        
        if len(patch_slice_points) > 0:
            # Transform coordinates to patch space
            patch_path_y = patch_slice_points[:, 1] - y_min
            patch_path_x = patch_slice_points[:, 2] - x_min
            ax_zoom.plot(patch_path_x, patch_path_y, 'r-', linewidth=3)
        
        # Mark current position in zoomed view
        if (y >= y_min and y < y_max and x >= x_min and x < x_max):
            ax_zoom.scatter(x - x_min, y - y_min, c='red', s=150, marker='o')
        
        ax_zoom.set_title(f'Zoomed View')
        ax_zoom.axis('off')  # Hide axes for cleaner look
        
        #------------------ Tube/First-Person View (Bottom Right) ------------------#
        # Get tangent direction (normalized)
        forward = current_tangent
        
        # Create an orthogonal basis for the viewing plane
        # Find the least aligned axis to create a truly orthogonal basis
        axis_alignments = np.abs(forward)
        least_aligned_idx = np.argmin(axis_alignments)
        
        # Create a reference vector along that axis
        reference = np.zeros(3)
        reference[least_aligned_idx] = 1.0
        
        # Compute right vector (orthogonal to forward)
        right = np.cross(forward, reference)
        right = right / np.linalg.norm(right)
        
        # Compute up vector (orthogonal to forward and right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Generate the viewing plane
        # This is a plane perpendicular to the tangent direction
        plane_size = field_of_view // 2
        plane_points_y = []
        plane_points_x = []
        plane_values = []
        plane_ref_values = []  # For reference image values
        
        # Sample points on the viewing plane
        for i in range(-plane_size, plane_size + 1):
            for j in range(-plane_size, plane_size + 1):
                # Calculate the point in 3D space
                point = current_point + i * up + j * right
                
                # Convert to integers for indexing
                pz, py, px = np.round(point).astype(int)
                
                # Check if the point is within the image bounds
                if (0 <= pz < image.shape[0] and 
                    0 <= py < image.shape[1] and 
                    0 <= px < image.shape[2]):
                    
                    # Store the point coordinates in the viewing plane
                    plane_points_y.append(i + plane_size)
                    plane_points_x.append(j + plane_size)
                    
                    # Get the image value at this point
                    plane_values.append(image[pz, py, px])
                    
                    # Get reference image value at this point if available
                    if reference_image is not None:
                        if is_multichannel:
                            # For RGB reference, we'll store the indices to access later
                            plane_ref_values.append((pz, py, px))
                        else:
                            plane_ref_values.append(reference_image[pz, py, px])
        
        # Create a 2D array for the viewing plane
        if plane_points_y:  # Check if we have any valid points
            # Create a blank viewing plane
            plane_image = np.zeros((plane_size * 2 + 1, plane_size * 2 + 1))
            
            # Fill in the values we sampled
            for py, px, val in zip(plane_points_y, plane_points_x, plane_values):
                plane_image[py, px] = val
            
            # Display the viewing plane
            ax_tube.imshow(plane_image, cmap='gray')
            
            # Overlay reference image in tube view if provided
            if reference_image is not None:
                if is_multichannel:
                    # For RGB reference, create a blank RGBA image
                    ref_plane_image = np.zeros((plane_size * 2 + 1, plane_size * 2 + 1, 4))
                    # Set alpha channel to transparent by default
                    ref_plane_image[:, :, 3] = 0
                    
                    # Fill in RGB values from reference image
                    for py, px, indices in zip(plane_points_y, plane_points_x, plane_ref_values):
                        pz, py_ref, px_ref = indices
                        # Get RGB values from reference image
                        rgb = reference_image[pz, py_ref, px_ref]
                        # Normalize if needed
                        if rgb.max() > 1:
                            rgb = rgb / 255.0
                        # Set RGB values and alpha
                        ref_plane_image[py, px, :3] = rgb
                        ref_plane_image[py, px, 3] = reference_alpha
                    
                    ax_tube.imshow(ref_plane_image)
                else:
                    # For single channel reference, create a blank plane
                    ref_plane_image = np.zeros((plane_size * 2 + 1, plane_size * 2 + 1))
                    
                    # Fill in the values from reference image
                    for py, px, val in zip(plane_points_y, plane_points_x, plane_ref_values):
                        ref_plane_image[py, px] = val
                    
                    ax_tube.imshow(ref_plane_image, cmap=reference_cmap, alpha=reference_alpha)
            
            # Show the path ahead
            # Project the next several points onto the viewing plane
            look_ahead = view_distance  # How many points to look ahead
            ahead_points_y = []
            ahead_points_x = []
            
            for i in range(1, look_ahead + 1):
                next_idx = min(current_idx + i, len(path) - 1)
                if next_idx == current_idx:
                    break
                    
                # Vector from current point to next point
                next_point = path[next_idx]
                vector = next_point - current_point
                
                # Project this vector onto the viewing plane
                # First, find the distance along the forward direction
                forward_dist = np.dot(vector, forward)
                
                # Only show points that are ahead of us
                if forward_dist > 0:
                    # Find the components along the up and right vectors
                    up_component = np.dot(vector, up)
                    right_component = np.dot(vector, right)
                    
                    # Convert to viewing plane coordinates
                    view_y = plane_size + int(up_component)
                    view_x = plane_size + int(right_component)
                    
                    # Check if the point is within the viewing plane
                    if (0 <= view_y < plane_size * 2 + 1 and 
                        0 <= view_x < plane_size * 2 + 1):
                        ahead_points_y.append(view_y)
                        ahead_points_x.append(view_x)
            
            # Plot the path ahead as a red line if desired
            if len(ahead_points_y) > 1:
                ax_tube.plot(ahead_points_x, ahead_points_y, 'r-', linewidth=2)
                
                # Mark the next immediate point with a larger marker
                if ahead_points_x:
                    ax_tube.scatter(ahead_points_x[0], ahead_points_y[0], 
                                  c='red', s=100, marker='o', alpha=0.4)
            
            # Show a "target" reticle at the center if desired
            center = plane_size
            ax_tube.axhline(center, color='yellow', alpha=0.5)
            ax_tube.axvline(center, color='yellow', alpha=0.5)
            
            ax_tube.set_title(f"In-Tube View (Forward Direction)")
        else:
            ax_tube.text(0.5, 0.5, "Out of bounds", ha='center', va='center', transform=ax_tube.transAxes)
        
        # Add a super title for the whole figure
        plt.suptitle(f"Brightest Path Flythrough - Position: Z={z}, Y={y}, X={x}", fontsize=14)
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(path), interval=1000/fps)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    # Save animation
    anim.save(output_file, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    
    print(f"Integrated flythrough animation saved to {output_file}")
    return output_file