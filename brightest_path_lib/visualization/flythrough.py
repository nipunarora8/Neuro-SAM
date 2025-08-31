import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from brightest_path_lib.algorithm import WaypointBidirectionalAStarSearch

def create_path_flythrough(image, start_point, goal_point, waypoints=None, 
                          output_file='flythrough.mp4', fps=15, zoom_size=50):
    """
    Create a simple fly-through visualization along the brightest path in a 3D image stack
    with a zoomed view of the current position
    
    Parameters:
    -----------
    image : numpy.ndarray
        The 3D image data (z, y, x)
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
    """
    # Run the brightest path algorithm
    astar = WaypointBidirectionalAStarSearch(
        image=image,
        start_point=np.array(start_point),
        goal_point=np.array(goal_point),
        waypoints=waypoints if waypoints else None
    )
    
    path = astar.search()
    
    if not astar.found_path:
        raise ValueError("Could not find a path through the image")
    
    # Convert path to numpy array
    path_array = np.array(path)
    
    # Create figure for animation with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), 
                                 gridspec_kw={'width_ratios': [3, 1]})
    
    # Function to update the plot for each frame
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        # Get current position in the path
        current_idx = min(frame_idx, len(path) - 1)
        current_point = path[current_idx]
        z, y, x = current_point
        
        # Display the current slice from the image stack
        ax1.imshow(image[z], cmap='gray')
        
        # Plot the path projection on this slice
        # Find all path points on the current z-slice
        slice_points = path_array[path_array[:, 0] == z]
        if len(slice_points) > 0:
            ax1.plot(slice_points[:, 2], slice_points[:, 1], 'r-', linewidth=2)
        
        # Mark the current position
        ax1.scatter(x, y, c='red', s=100, marker='o')
        
        # Plot "shadows" of the path from other slices (lighter color)
        # Points from slices above current position
        above_points = path_array[path_array[:, 0] < z]
        if len(above_points) > 0:
            ax1.plot(above_points[:, 2], above_points[:, 1], 'r-', alpha=0.3, linewidth=1)
            
        # Points from slices below current position
        below_points = path_array[path_array[:, 0] > z]
        if len(below_points) > 0:
            ax1.plot(below_points[:, 2], below_points[:, 1], 'r-', alpha=0.3, linewidth=1)
        
        # Add a rectangle showing the zoomed area
        half_size = zoom_size // 2
        zoom_rect = plt.Rectangle((x - half_size, y - half_size), 
                                zoom_size, zoom_size,
                                fill=False, edgecolor='yellow', linewidth=2)
        ax1.add_patch(zoom_rect)
        
        ax1.set_title(f'Slice Z={z} - Frame {frame_idx+1}/{len(path)}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # Create zoomed view
        # Get coordinates for the zoom window, handling edges of the image
        y_min = max(0, y - half_size)
        y_max = min(image.shape[1], y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(image.shape[2], x + half_size)
        
        # Extract the patch
        zoom_patch = image[z, y_min:y_max, x_min:x_max]
        
        # Display zoomed patch
        ax2.imshow(zoom_patch, cmap='gray')
        
        # Find path points within this zoomed patch and transform coordinates
        patch_slice_points = slice_points[
            (slice_points[:, 1] >= y_min) & (slice_points[:, 1] < y_max) &
            (slice_points[:, 2] >= x_min) & (slice_points[:, 2] < x_max)
        ]
        
        if len(patch_slice_points) > 0:
            # Transform coordinates to patch space
            patch_path_y = patch_slice_points[:, 1] - y_min
            patch_path_x = patch_slice_points[:, 2] - x_min
            ax2.plot(patch_path_x, patch_path_y, 'r-', linewidth=3)
        
        # Mark current position in zoomed view
        if (y >= y_min and y < y_max and x >= x_min and x < x_max):
            ax2.scatter(x - x_min, y - y_min, c='red', s=150, marker='o')
        
        ax2.set_title(f'Zoomed View')
        ax2.axis('off')  # Hide axes for cleaner look
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(path), interval=1000/fps)
    
    # Save animation
    anim.save(output_file, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    
    print(f"Flythrough animation saved to {output_file}")
    return output_file