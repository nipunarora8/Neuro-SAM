import napari
import numpy as np
import imageio.v2 as io
from napari_utils.main_widget import NeuroSAMWidget  # Updated with anisotropic scaling


def pad_image_for_patches(image, patch_size=128, pad_value=0):
    """
    Pad the image so that its height and width are multiples of patch_size.
    Handles various image dimensions including stacks of colored images.
    
    Parameters:
    -----------
    image (np.ndarray): Input image array:
        - 2D: (H x W)
        - 3D: (C x H x W) for grayscale stacks or (H x W x C) for colored image
        - 4D: (Z x H x W x C) for stacks of colored images
    patch_size (int): The patch size to pad to, default is 128.
    pad_value (int or tuple): The constant value(s) for padding.
    
    Returns:
    --------
    padded_image (np.ndarray): The padded image.
    padding_amounts (tuple): The amount of padding applied (pad_h, pad_w).
    original_dims (tuple): The original dimensions (h, w).
    """
    # Determine the image format and dimensions
    if image.ndim == 2:
        # 2D grayscale image (H x W)
        h, w = image.shape
        is_color = False
        is_stack = False
    elif image.ndim == 3:
        # This could be either:
        # - A stack of 2D grayscale images (Z x H x W)
        # - A single color image (H x W x C)
        # We'll check the third dimension to decide
        if image.shape[2] <= 4:  # Assuming color channels ≤ 4 (RGB, RGBA)
            # Single color image (H x W x C)
            h, w, c = image.shape
            is_color = True
            is_stack = False
        else:
            # Stack of grayscale images (Z x H x W)
            z, h, w = image.shape
            is_color = False
            is_stack = True
    elif image.ndim == 4:
        # Stack of color images (Z x H x W x C)
        z, h, w, c = image.shape
        is_color = True
        is_stack = True
    else:
        raise ValueError(f"Unsupported image dimension: {image.ndim}")
    
    # Compute necessary padding for height and width
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    # Pad the image based on its format
    if not is_stack and not is_color:
        # 2D grayscale image
        padding = ((0, pad_h), (0, pad_w))
        padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    
    elif is_stack and not is_color:
        # Stack of grayscale images (Z x H x W)
        padding = ((0, 0), (0, pad_h), (0, pad_w))
        padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    
    elif not is_stack and is_color:
        # Single color image (H x W x C)
        padding = ((0, pad_h), (0, pad_w), (0, 0))
        padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    
    elif is_stack and is_color:
        # Stack of color images (Z x H x W x C)
        padding = ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
        padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    
    return padded_image, (pad_h, pad_w), (h, w)


def run_neuro_sam(image=None, image_path=None, spacing_xyz=(94.0, 94.0, 500.0)):
    """
    Launch the NeuroSAM plugin with anisotropic scaling support
    
    Parameters:
    -----------
    image : numpy.ndarray, optional
        3D or higher-dimensional image data. If None, image_path must be provided.
    image_path : str, optional
        Path to image file to load. If None, image must be provided.
    spacing_xyz : tuple, optional
        Original voxel spacing in (x, y, z) nanometers. 
        Default: (94.0, 94.0, 500.0) - typical for two photon microscopy
    
    Returns:
    --------
    viewer : napari.Viewer
        The napari viewer instance
    """
    # Validate inputs
    if image is None and image_path is None:
        raise ValueError("Either image or image_path must be provided")
    
    # Load image if path provided
    if image is None:
        try:
            image = np.asarray(io.imread(image_path))
            print(f"Loaded image from {image_path}")
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {str(e)}")
    
    # Normalize image to 0-1 range for better visualization
    if image.max() > 1:
        image = image.astype(np.float32)
        image_min, image_max = image.min(), image.max()
        image = (image - image_min) / (image_max - image_min)
        print(f"Normalized image from range [{image_min:.2f}, {image_max:.2f}] to [0, 1]")

    # Pad image for patch-based processing
    image, padding_amounts, original_dims = pad_image_for_patches(image, patch_size=128, pad_value=0)
    if padding_amounts[0] > 0 or padding_amounts[1] > 0:
        print(f"Padded image by {padding_amounts} pixels to be divisible by 128")
        print(f"New image shape: {image.shape}")
    
    # Create a viewer
    viewer = napari.Viewer()
    
    # Display spacing information
    print(f"Original voxel spacing: X={spacing_xyz[0]:.1f}, Y={spacing_xyz[1]:.1f}, Z={spacing_xyz[2]:.1f} nm")
    
    # Create and add our widget with anisotropic scaling capabilities
    neuro_sam_widget = NeuroSAMWidget(
        viewer=viewer, 
        image=image, 
        original_spacing_xyz=spacing_xyz
    )
    
    viewer.window.add_dock_widget(
        neuro_sam_widget, name="Neuro-SAM", area="right"
    )
    
    # Set initial view
    if image.ndim >= 3:
        # Start with a mid-slice view for 3D+ images
        mid_slice = image.shape[0] // 2
        viewer.dims.set_point(0, mid_slice)
    
    # Display startup information
    napari.utils.notifications.show_info(
        f"NeuroSAM loaded! Image shape: {image.shape}. "
        f"Configure voxel spacing in the 'Path Tracing' tab first."
    )
    
    return viewer


def run_neuro_sam_with_metadata(image_path, metadata=None):
    """
    Launch NeuroSAM with metadata-derived spacing information
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    metadata : dict, optional
        Metadata dictionary with spacing information.
        Expected keys: 'spacing_x_nm', 'spacing_y_nm', 'spacing_z_nm'
        
    Returns:
    --------
    viewer : napari.Viewer
        The napari viewer instance
    """
    # Default spacing values
    default_spacing = (94.0, 94.0, 500.0)  # (x, y, z) in nm
    
    if metadata is not None:
        try:
            # Extract spacing from metadata
            x_spacing = metadata.get('spacing_x_nm', default_spacing[0])
            y_spacing = metadata.get('spacing_y_nm', default_spacing[1])
            z_spacing = metadata.get('spacing_z_nm', default_spacing[2])
            
            spacing_xyz = (float(x_spacing), float(y_spacing), float(z_spacing))
            print(f"Using metadata-derived spacing: X={spacing_xyz[0]:.1f}, Y={spacing_xyz[1]:.1f}, Z={spacing_xyz[2]:.1f} nm")
        except (ValueError, TypeError) as e:
            print(f"Error parsing metadata spacing, using defaults: {e}")
            spacing_xyz = default_spacing
    else:
        spacing_xyz = default_spacing
        print(f"No metadata provided, using default spacing: X={spacing_xyz[0]:.1f}, Y={spacing_xyz[1]:.1f}, Z={spacing_xyz[2]:.1f} nm")
    
    return run_neuro_sam(image_path=image_path, spacing_xyz=spacing_xyz)


def load_ome_tiff_with_spacing(image_path):
    """
    Load OME-TIFF file and extract voxel spacing from metadata
    
    Parameters:
    -----------
    image_path : str
        Path to OME-TIFF file
        
    Returns:
    --------
    tuple : (image, spacing_xyz)
        Image array and spacing tuple
    """
    image = np.asarray(io.imread(image_path))
    return image, (94.0, 94.0, 500.0)


# For direct execution from command line
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch NeuroSAM with anisotropic scaling support")
    parser.add_argument("--image_path", nargs="?", help="Path to image file")
    parser.add_argument("--x-spacing", type=float, default=94.0, help="X voxel spacing in nm (default: 94.0)")
    parser.add_argument("--y-spacing", type=float, default=94.0, help="Y voxel spacing in nm (default: 94.0)")
    parser.add_argument("--z-spacing", type=float, default=500.0, help="Z voxel spacing in nm (default: 500.0)")
    parser.add_argument("--ome", action="store_true", help="Try to extract spacing from OME-TIFF metadata")
    
    args = parser.parse_args()
    
    if args.image_path:
        if args.ome:
            # Try to load OME-TIFF with metadata
            try:
                image, spacing_xyz = load_ome_tiff_with_spacing(args.image_path)
                viewer = run_neuro_sam(image=image, spacing_xyz=spacing_xyz)
            except Exception as e:
                print(f"Error loading OME-TIFF: {e}")
                print("Falling back to standard loading...")
                spacing_xyz = (args.x_spacing, args.y_spacing, args.z_spacing)
                viewer = run_neuro_sam(image_path=args.image_path, spacing_xyz=spacing_xyz)
        else:
            # Use command line spacing arguments
            spacing_xyz = (args.x_spacing, args.y_spacing, args.z_spacing)
            viewer = run_neuro_sam(image_path=args.image_path, spacing_xyz=spacing_xyz)
    else:
        # Try to load a default benchmark image
        try:
            default_path = './dataset/DeepD3_Benchmark.tif'
            print(f"No image path provided, trying to load default: {default_path}")
            spacing_xyz = (args.x_spacing, args.y_spacing, args.z_spacing)
            viewer = run_neuro_sam(image_path=default_path, spacing_xyz=spacing_xyz)
        except FileNotFoundError:
            sys.exit(1)
    
    print("\nStarted NeuroSAM with anisotropic scaling support!")
    print("Configure voxel spacing in the 'Path Tracing' tab before starting analysis.")
    napari.run()  # Start the Napari event loop