import cc3d
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from matplotlib.colors import ListedColormap
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def analyze_dendrite_stack(image_path, delta_value=10, connectivity=26):
    """
    Analyze 3D dendrite TIFF stack using connected-components-3d with delta parameter
    
    Parameters:
    - image_path: Path to the TIFF stack
    - delta_value: Delta parameter for cc3d (controls intensity similarity threshold)
    - connectivity: Neighborhood connectivity (6/18/26 for 3D)
    """
    # Load the image stack
    print(f"Loading 3D TIFF stack from {image_path}")
    image = io.imread(image_path)
    
    # Print image info
    print(f"Image stack shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Value range: {image.min()} to {image.max()}")
    
    # Apply connected components with delta parameter
    print(f"Running connected components with delta={delta_value}, connectivity={connectivity}")
    labels_out = cc3d.connected_components(image, delta=delta_value, connectivity=connectivity)
    
    # Get number of unique components (excluding background 0)
    unique_labels = np.unique(labels_out)
    num_components = len(unique_labels) - (1 if 0 in unique_labels else 0)
    print(f"Found {num_components} connected components")
    
    # Get component statistics
    stats = cc3d.statistics(labels_out)
    
    # Sort components by size (voxel count) to identify main dendrites vs spines
    component_sizes = [(label, stats["voxel_counts"][label]) for label in stats["voxel_counts"]]
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Print sizes of largest components
    print("\nLargest components (possibly main dendrites):")
    for i, (label, size) in enumerate(component_sizes[:5]):
        if i >= len(component_sizes):
            break
        print(f"Component {label}: {size} voxels")
    
    # Print sizes of some medium components (might be spines)
    if len(component_sizes) > 10:
        print("\nMedium-sized components (possibly spines):")
        middle_idx = len(component_sizes) // 2
        for i, (label, size) in enumerate(component_sizes[middle_idx:middle_idx+5]):
            if middle_idx + i >= len(component_sizes):
                break
            print(f"Component {label}: {size} voxels")
    
    # Visualize several slices from the stack
    n_slices = min(4, image.shape[0])
    slice_indices = np.linspace(0, image.shape[0]-1, n_slices, dtype=int)
    
    # Make a random colormap for visualization
    np.random.seed(42)  # For reproducible colors
    colors = np.random.rand(num_components+1, 3)
    colors[0] = [0, 0, 0]  # background black
    cmap = ListedColormap(colors)
    
    fig, axs = plt.subplots(n_slices, 2, figsize=(12, 4*n_slices))
    
    if n_slices == 1:
        axs = np.array([axs])  # Make it 2D for consistent indexing
    
    for i, slice_idx in enumerate(slice_indices):
        # Original image slice
        im1 = axs[i, 0].imshow(image[slice_idx], cmap='gray')
        axs[i, 0].set_title(f'Original Image (Slice {slice_idx})')
        divider = make_axes_locatable(axs[i, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        # Connected components slice
        im2 = axs[i, 1].imshow(labels_out[slice_idx], cmap=cmap)
        axs[i, 1].set_title(f'Connected Components (Slice {slice_idx})')
        divider = make_axes_locatable(axs[i, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    # Optional: Show 3D maximum intensity projection
    if image.shape[0] > 1:  # Only if it's truly 3D
        print("\nCreating maximum intensity projections...")
        
        # Create maximum intensity projections
        orig_z_proj = np.max(image, axis=0)
        labels_z_proj = np.max(labels_out, axis=0)
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        
        # Original MIP
        im1 = axs[0].imshow(orig_z_proj, cmap='gray')
        axs[0].set_title('Original Image (Z-Max Projection)')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        # Labels MIP
        im2 = axs[1].imshow(labels_z_proj, cmap=cmap)
        axs[1].set_title('Connected Components (Z-Max Projection)')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
        
        plt.tight_layout()
        plt.show()
    
    return labels_out, stats

def delta_comparison_for_dendrites(image_path, delta_values=[5, 15, 30, 50], connectivity=26):
    """
    Compare different delta values for dendrite/spine segmentation
    """
    # Load the 3D stack
    print(f"Loading 3D TIFF stack from {image_path}")
    image = io.imread(image_path)
    
    print(f"Image stack shape: {image.shape}")
    
    # Choose a representative slice in the middle
    if len(image.shape) >= 3:
        middle_slice = image.shape[0] // 2
    else:
        middle_slice = 0
        print("Warning: Image appears to be 2D, not a stack")
    
    # Determine number of rows (original + each delta value)
    n_rows = len(delta_values) + 1
    
    fig, axs = plt.subplots(n_rows, 1, figsize=(10, 5*n_rows))
    
    # Show original image
    im0 = axs[0].imshow(image[middle_slice], cmap='gray')
    axs[0].set_title(f'Original Image (Slice {middle_slice})')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    # Process with different delta values
    results = []
    
    for i, delta in enumerate(delta_values):
        print(f"Processing with delta={delta}...")
        # Apply connected components
        labels = cc3d.connected_components(image, delta=delta, connectivity=connectivity)
        results.append(labels)
        
        # Get component count
        num_components = len(np.unique(labels)) - 1
        
        # Make a random colormap for visualization
        np.random.seed(i)  # Different seed for each delta
        colors = np.random.rand(num_components+1, 3)
        colors[0] = [0, 0, 0]  # background black
        cmap = ListedColormap(colors)
        
        # Display the result
        im = axs[i+1].imshow(labels[middle_slice], cmap=cmap)
        axs[i+1].set_title(f'Connected Components with delta={delta} ({num_components} components)')
        plt.colorbar(im, ax=axs[i+1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # Also show Z-projections for each delta value
    if len(image.shape) >= 3 and image.shape[0] > 1:
        print("\nCreating maximum intensity projections for each delta value...")
        
        fig, axs = plt.subplots(1, len(delta_values) + 1, figsize=(5*(len(delta_values) + 1), 5))
        
        # Original MIP
        orig_z_proj = np.max(image, axis=0)
        im0 = axs[0].imshow(orig_z_proj, cmap='gray')
        axs[0].set_title('Original (Z-Max Projection)')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        
        # Show MIP for each delta result
        for i, (delta, labels) in enumerate(zip(delta_values, results)):
            labels_z_proj = np.max(labels, axis=0)
            
            # Get component count
            num_components = len(np.unique(labels)) - 1
            
            # Make colormap
            np.random.seed(i)
            colors = np.random.rand(10000, 3)  # Large number to accommodate many labels
            colors[0] = [0, 0, 0]
            cmap = ListedColormap(colors[:num_components+1])
            
            im = axs[i+1].imshow(labels_z_proj, cmap=cmap)
            axs[i+1].set_title(f'delta={delta} ({num_components} components)')
            plt.colorbar(im, ax=axs[i+1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    return results

def visualize_specific_components(image, labels, component_ids, slice_idx=None):
    """
    Visualize specific components (useful for inspecting dendrites vs spines)
    
    Parameters:
    - image: Original image data
    - labels: Connected components labeled image
    - component_ids: List of component IDs to visualize
    - slice_idx: Slice to visualize (if None, uses middle slice)
    """
    if slice_idx is None:
        slice_idx = image.shape[0] // 2
    
    # Create a mask for the selected components
    mask = np.zeros_like(labels, dtype=bool)
    for cid in component_ids:
        mask = np.logical_or(mask, labels == cid)
    
    # Display the results
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    im0 = axs[0].imshow(image[slice_idx], cmap='gray')
    axs[0].set_title(f'Original Image (Slice {slice_idx})')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    # All components
    # Random colormap
    num_components = len(np.unique(labels)) - 1
    np.random.seed(42)
    colors = np.random.rand(num_components+1, 3)
    colors[0] = [0, 0, 0]
    cmap = ListedColormap(colors)
    
    im1 = axs[1].imshow(labels[slice_idx], cmap=cmap)
    axs[1].set_title(f'All Components (Slice {slice_idx})')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    # Selected components
    selected = np.zeros_like(labels[slice_idx])
    for i, cid in enumerate(component_ids):
        selected[labels[slice_idx] == cid] = i + 1
    
    # Colormap for selected components
    n_selected = len(component_ids)
    np.random.seed(100)
    sel_colors = np.random.rand(n_selected+1, 3)
    sel_colors[0] = [0, 0, 0]
    sel_cmap = ListedColormap(sel_colors)
    
    im2 = axs[2].imshow(selected, cmap=sel_cmap)
    axs[2].set_title(f'Selected Components {component_ids} (Slice {slice_idx})')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

    # Also show 3D projections of selected components
    if image.shape[0] > 1:
        # Create maximum intensity projections
        orig_z_proj = np.max(image, axis=0)
        
        # Project selected components
        selected_3d = np.zeros_like(labels)
        for i, cid in enumerate(component_ids):
            selected_3d[labels == cid] = i + 1
        
        selected_z_proj = np.max(selected_3d, axis=0)
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        
        # Original MIP
        im1 = axs[0].imshow(orig_z_proj, cmap='gray')
        axs[0].set_title('Original Image (Z-Max Projection)')
        plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
        
        # Selected components MIP
        im2 = axs[1].imshow(selected_z_proj, cmap=sel_cmap)
        axs[1].set_title(f'Selected Components {component_ids} (Z-Max Projection)')
        plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your dendrite TIFF stack path
    image_path = r'DeepD3_Benchmark.tif'  # Update this with your actual file path
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
    else:
        # Option 1: Compare different delta values to find the best one
        results = delta_comparison_for_dendrites(
            image_path, 
            delta_values=[5, 10, 20, 50], 
            connectivity=26
        )
        
        # Option 2: Analyze with a specific delta value
        # labels, stats = analyze_dendrite_stack(
        #     image_path, 
        #     delta_value=20,  # Adjust based on Option 1 results
        #     connectivity=26
        # )
        
        # Option 3: Visualize specific components (e.g., to examine spines vs. dendrites)
        # Uncomment after running Option 2
        # # Visualize the 3 largest components (likely main dendrites)
        # large_components = [comp_id for comp_id, _ in sorted(
        #     stats["voxel_counts"].items(), 
        #     key=lambda x: x[1], 
        #     reverse=True
        # )[:3]]
        # visualize_specific_components(io.imread(image_path), labels, large_components)
        
        # # Visualize some medium-sized components (likely spines)
        # sorted_components = sorted(stats["voxel_counts"].items(), key=lambda x: x[1], reverse=True)
        # medium_idx = len(sorted_components) // 2
        # medium_components = [comp_id for comp_id, _ in sorted_components[medium_idx:medium_idx+3]]
        # visualize_specific_components(io.imread(image_path), labels, medium_components)