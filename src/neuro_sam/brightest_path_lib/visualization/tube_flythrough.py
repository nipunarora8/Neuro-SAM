import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from neuro_sam.brightest_path_lib.algorithm import WaypointBidirectionalAStarSearch
from mpl_toolkits.mplot3d import Axes3D

def create_tube_flythrough(image, start_point, goal_point, waypoints=None, 
                                   output_file='tube_flythrough.mp4', fps=15,
                                   view_distance=5, field_of_view=40):
    """
    Create a first-person fly-through visualization along the brightest path in a 3D image
    
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
    view_distance : int
        How far ahead to look along the path
    field_of_view : int
        Width of the field of view in degrees
    """
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
    # For smoother movement, calculate using a window of surrounding points
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
    
    # Create figure for visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Function to update the plot for each frame
    def update(frame_idx):
        ax.clear()
        
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
        
        # Get tangent direction (normalized)
        direction = current_tangent
        
        # Create an orthogonal basis for the viewing plane
        # First normalize the tangent vector (it should already be normalized)
        forward = direction / np.linalg.norm(direction)
        
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
        # Calculate points on the plane based on field of view
        plane_size = field_of_view // 2
        plane_points_y = []
        plane_points_x = []
        plane_values = []
        
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
        
        # Create a 2D array for the viewing plane
        if plane_points_y:  # Check if we have any valid points
            # Create a blank viewing plane
            plane_image = np.zeros((plane_size * 2 + 1, plane_size * 2 + 1))
            
            # Fill in the values we sampled
            for y, x, val in zip(plane_points_y, plane_points_x, plane_values):
                plane_image[y, x] = val
            
            # Display the viewing plane
            ax.imshow(plane_image, cmap='gray')
            
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
            
            # Plot the path ahead as a red line
            if len(ahead_points_y) > 1:
                ax.plot(ahead_points_x, ahead_points_y, 'r-', linewidth=2)
                
                # Mark the next immediate point with a larger marker
                if ahead_points_x:
                    ax.scatter(ahead_points_x[0], ahead_points_y[0], 
                              c='red', s=100, marker='o')
            
            # Show a "target" reticle at the center
            center = plane_size
            ax.axhline(center, color='yellow', alpha=0.5)
            ax.axvline(center, color='yellow', alpha=0.5)
            
            # Add position information
            ax.set_title(f"Position: Z={z}, Y={y}, X={x}\nFrame {frame_idx+1}/{len(path)}")
            ax.set_xlabel("View X")
            ax.set_ylabel("View Y")
        else:
            ax.text(0.5, 0.5, "Out of bounds", ha='center', va='center', transform=ax.transAxes)
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(path), interval=1000/fps)
    
    # Save animation
    anim.save(output_file, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    
    print(f"Tube fly-through animation saved to {output_file}")
    return output_file