import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import maximum_filter
from skimage.segmentation import watershed
from skimage.morphology import disk, closing
import matplotlib.pyplot as plt
import glob

def find_local_maxima(image, min_distance, threshold):
    """Find local maxima in an image"""
    # Create neighborhood for local maximum detection
    neighborhood_size = 2 * min_distance + 1
    local_max = (maximum_filter(image, size=neighborhood_size) == image)
    
    # Apply threshold
    above_threshold = image > threshold
    peaks = local_max & above_threshold
    
    # Get coordinates
    coords = np.where(peaks)
    return np.column_stack((coords[0], coords[1]))

def detect_spine_centers_from_array(img_array, min_distance=3, visualize=False):
    """
    Detect spine centers from a numpy array
    
    Parameters:
    img_array: numpy array (grayscale image)
    min_distance: minimum distance between centers
    visualize: whether to show visualization
    
    Returns:
    centers: numpy array of (row, col) coordinates
    labels: watershed labeled image
    """
    # Ensure it's grayscale
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array.copy()
    
    # Create binary mask
    _, binary_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Simple cleanup - just fill small holes
    kernel = disk(1)
    binary_mask = closing(binary_mask, kernel)
    
    # Calculate distance transform
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Simple threshold - 30% of maximum distance
    max_distance = np.max(distance)
    threshold = max_distance * 0.3
    
    # Find spine centers
    centers = find_local_maxima(distance, min_distance, threshold)
    
    # Create markers for watershed
    markers = np.zeros_like(distance, dtype=int)
    for i, center in enumerate(centers):
        markers[center[0], center[1]] = i + 1
    
    # Apply watershed segmentation
    labels = watershed(-distance, markers, mask=binary_mask)
    
    if visualize:
        visualize_results(img_array, centers, labels, distance, binary_mask)
    
    return centers, labels

def visualize_results(original_img, centers, labels, distance_transform, binary_mask):
    """Show the results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Binary mask
    axes[0, 1].imshow(binary_mask, cmap='gray')
    axes[0, 1].set_title('Binary Mask')
    axes[0, 1].axis('off')
    
    # Distance transform
    axes[0, 2].imshow(distance_transform, cmap='hot')
    axes[0, 2].set_title('Distance Transform')
    axes[0, 2].axis('off')
    
    # Watershed labels
    axes[1, 0].imshow(labels, cmap='nipy_spectral')
    axes[1, 0].set_title('Watershed Labels')
    axes[1, 0].axis('off')
    
    # Centers on original
    axes[1, 1].imshow(original_img, cmap='gray')
    if len(centers) > 0:
        axes[1, 1].plot(centers[:, 1], centers[:, 0], 'r+', markersize=10, markeredgewidth=2)
    axes[1, 1].set_title(f'Detected Centers ({len(centers)})')
    axes[1, 1].axis('off')
    
    # Centers on distance transform
    axes[1, 2].imshow(distance_transform, cmap='hot')
    if len(centers) > 0:
        axes[1, 2].plot(centers[:, 1], centers[:, 0], 'b+', markersize=10, markeredgewidth=2)
    axes[1, 2].set_title('Centers on Distance Transform')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
# if __name__ == "__main__":
    # For numpy array input (MAIN USE CASE)
    # your_frame = cv2.imread(glob.glob('Spines/*')[0], cv2.IMREAD_GRAYSCALE)  # or any numpy array
    # centers, labels = detect_spine_centers_from_array(your_frame, min_distance= 3, visualize=True)
    # print(f"Found {len(centers)} spine centers at coordinates:")
    # print(centers)