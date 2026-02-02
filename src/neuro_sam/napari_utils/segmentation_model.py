import numpy as np
import torch
import cv2
from scipy.ndimage import zoom
from skimage.morphology import binary_closing, binary_opening, remove_small_objects
from scipy.ndimage import binary_fill_holes as ndimage_fill_holes
from scipy.ndimage import label
from matplotlib.path import Path


from neuro_sam.utils import get_weights_path

class DendriteSegmenter:
    """Class for segmenting dendrites from 3D image volumes using SAM2 with overlapping patches"""
    
    def __init__(self, model_path=None, config_path="sam2.1_hiera_s.yaml", weights_path=None, device="cuda"):
        """
        Initialize the dendrite segmenter with overlapping patches.
        
        Args:
            model_path: Path to SAM2 model checkpoint (auto-downloaded if None)
            config_path: Path to model configuration
            weights_path: Path to trained weights (auto-downloaded if None)
            device: Device to run the model on (cpu or cuda)
        """
        if model_path is None:
            self.model_path = get_weights_path("sam2.1_hiera_small.pt")
        else:
            self.model_path = model_path
            
        self.config_path = config_path
        
        if weights_path is None:
            self.weights_path = get_weights_path("dendrite_model.torch")
        else:
            self.weights_path = weights_path
            
        self.device = device
        self.predictor = None

    def load_model(self):
        """Load the segmentation model with improved error reporting and path handling"""
        try:
            print(f"Loading dendrite model with overlapping patches from {self.model_path} with config {self.config_path}")
            print(f"Using weights from {self.weights_path}")
            
            # Try importing first to catch import errors
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                print("Successfully imported SAM2 modules")
            except ImportError as ie:
                print(f"Failed to import SAM2 modules: {str(ie)}")
                print("Make sure the SAM2 package is installed and in the Python path")
                return False
            
            # Use bfloat16 for memory efficiency
            torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

            # Build model and load weights
            print("Building SAM2 model...")
            sam2_model = build_sam2(self.config_path, self.model_path, device=self.device)
            print("Creating SAM2 image predictor...")
            self.predictor = SAM2ImagePredictor(sam2_model)
            print("Loading model weights...")
            self.predictor.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            print("Dendrite model with overlapping patches loaded successfully")
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def predict_mask(self, img, positive_points, negative_points):
        """
        Predict mask using SAM with positive and negative prompt points (original working method)
        
        Args:
            img: Input image (2D)
            positive_points: List of (x, y) foreground points
            negative_points: List of (x, y) background points
            
        Returns:
            Binary mask
        """
        if self.predictor is None:
            print("Model not loaded. Call load_model() first.")
            return None
            
        # Convert to RGB format for SAM
        image_rgb = cv2.merge([img, img, img]).astype(np.float32)
        
        # Prepare points and labels
        points = np.array(positive_points + negative_points, dtype=np.float32)
        labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)
        
        # Run prediction
        try:
            with torch.no_grad():
                self.predictor.set_image(image_rgb)
                pred, _, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels
                )
            
            # Return the predicted mask
            return pred[0]
            
        except Exception as e:
            print(f"Error in predict_mask: {e}")
            import traceback
            traceback.print_exc()
            # Return an empty mask as fallback
            return np.zeros_like(img, dtype=np.uint8)

    def generate_overlapping_patches(self, image_shape, patch_size=128, stride=64):
        """
        Generate overlapping patch coordinates with 50% overlap for dendrites
        
        Args:
            image_shape: (height, width) of the image
            patch_size: Size of square patches
            stride: Step size between patches (64 for 50% overlap)
            
        Returns:
            List of (y_start, y_end, x_start, x_end) coordinates
        """
        height, width = image_shape
        patches = []
        
        # Generate patches with overlapping
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                y_end = min(y + patch_size, height)
                x_end = min(x + patch_size, width)
                
                # Only use patches that are close to full size
                if (y_end - y) >= patch_size * 0.8 and (x_end - x) >= patch_size * 0.8:
                    patches.append((y, y_end, x, x_end))
        
        # Handle edge cases - add patches for remaining borders
        # Right edge
        if width % stride != 0:
            for y in range(0, height - patch_size + 1, stride):
                x_start = width - patch_size
                if x_start >= 0:
                    patches.append((y, y + patch_size, x_start, width))
        
        # Bottom edge  
        if height % stride != 0:
            for x in range(0, width - patch_size + 1, stride):
                y_start = height - patch_size
                if y_start >= 0:
                    patches.append((y_start, height, x, x + patch_size))
        
        # Bottom-right corner
        if height % stride != 0 and width % stride != 0:
            y_start = height - patch_size
            x_start = width - patch_size
            if y_start >= 0 and x_start >= 0:
                patches.append((y_start, height, x_start, width))
        
        print(f"Generated {len(patches)} overlapping patches for dendrite segmentation on image shape {image_shape}")
        return patches

    def merge_overlapping_dendrite_predictions(self, all_patch_masks, patch_coords, image_shape, brightest_path):
        """
        Merge overlapping patch predictions with dendrite-specific enhancements
        
        Args:
            all_patch_masks: List of prediction masks from each patch
            patch_coords: List of (y_start, y_end, x_start, x_end) for each patch
            image_shape: (height, width) of full image
            brightest_path: List of path points for validation
            
        Returns:
            Final merged mask with enhanced dendrite structure
        """
        height, width = image_shape
        
        # Create accumulation arrays
        prediction_sum = np.zeros((height, width), dtype=np.float32)
        prediction_count = np.zeros((height, width), dtype=np.int32)
        
        # Accumulate predictions from all patches
        for mask, (y_start, y_end, x_start, x_end) in zip(all_patch_masks, patch_coords):
            if mask is not None and np.any(mask):
                # Add this patch's prediction to the accumulation
                mask_region = prediction_sum[y_start:y_end, x_start:x_end]
                count_region = prediction_count[y_start:y_end, x_start:x_end]
                
                # Ensure shapes match
                if mask.shape == mask_region.shape:
                    prediction_sum[y_start:y_end, x_start:x_end] += mask.astype(np.float32)
                    prediction_count[y_start:y_end, x_start:x_end] += 1
        
        # Avoid division by zero
        prediction_count[prediction_count == 0] = 1
        
        # Create average prediction
        averaged_prediction = prediction_sum / prediction_count
        
        # Apply adaptive threshold based on overlap (more conservative for dendrites)
        # Areas with more overlap get higher confidence
        adaptive_threshold = np.zeros_like(averaged_prediction)
        adaptive_threshold[prediction_count == 1] = 0.6  # Single patch areas (higher threshold)
        adaptive_threshold[prediction_count == 2] = 0.5  # 2x overlap areas  
        adaptive_threshold[prediction_count == 3] = 0.45 # 3x overlap areas
        adaptive_threshold[prediction_count >= 4] = 0.4  # 4+ overlap areas (most confident)
        
        # Create initial binary mask
        binary_mask = (averaged_prediction > adaptive_threshold).astype(np.uint8)
        
        # Apply dendrite-specific morphological operations
        enhanced_mask = self.enhance_dendrite_structure(binary_mask, brightest_path)
        
        print(f"Merged {len(all_patch_masks)} overlapping patches for dendrite")
        print(f"Max overlap count: {np.max(prediction_count)}")
        print(f"Final dendrite mask pixels: {np.sum(enhanced_mask)}")
        
        return enhanced_mask

    def enhance_dendrite_structure(self, binary_mask, brightest_path, min_dendrite_size=100):
        """
        Enhance dendrite structure to be more connected and tubular
        
        Args:
            binary_mask: Initial binary segmentation mask
            brightest_path: List of path points for validation
            min_dendrite_size: Minimum size of dendrite objects to keep
            
        Returns:
            Enhanced binary mask with better dendrite structure
        """
        if not np.any(binary_mask):
            return binary_mask
        
        enhanced_mask = binary_mask.copy()
        
        # Step 1: Fill small holes inside dendrites (dendrites should be solid tubes)
        enhanced_mask = ndimage_fill_holes(enhanced_mask).astype(np.uint8)
        
        # Step 2: Apply morphological closing to connect nearby dendrite segments
        # Use larger structuring element for dendrites compared to spines
        from skimage.morphology import footprint_rectangle
        # Horizontal closing to connect dendrite segments
        horizontal_element = footprint_rectangle((3, 7))  # Wider horizontal connectivity
        enhanced_mask = binary_closing(enhanced_mask, horizontal_element).astype(np.uint8)
        
        # Vertical closing to connect dendrite segments
        vertical_element = footprint_rectangle((7, 3))  # Taller vertical connectivity
        enhanced_mask = binary_closing(enhanced_mask, vertical_element).astype(np.uint8)
        
        # Step 3: Remove small noise objects (much larger threshold for dendrites)
        enhanced_mask = remove_small_objects(enhanced_mask.astype(bool), min_size=min_dendrite_size).astype(np.uint8)
        
        # Step 4: Final hole filling after connectivity enhancement
        enhanced_mask = ndimage_fill_holes(enhanced_mask).astype(np.uint8)
        
        # Step 5: Validate enhanced regions against path points
        if brightest_path is not None and len(brightest_path) > 0:
            enhanced_mask = self.validate_dendrite_against_path(enhanced_mask, brightest_path)
        
        print(f"Enhanced dendrite structure: {np.sum(binary_mask)} -> {np.sum(enhanced_mask)} pixels")
        
        return enhanced_mask

    def validate_dendrite_against_path(self, mask, brightest_path, max_distance=20):
        """
        Validate segmented regions against the brightest path, remove disconnected noise
        
        Args:
            mask: Binary segmentation mask
            brightest_path: List of (z, y, x) path points
            max_distance: Maximum distance from path to keep a region
            
        Returns:
            Validated mask with disconnected noise removed
        """
        if not np.any(mask):
            return mask
        
        # Label connected components
        labeled_mask, num_features = label(mask)
        
        if num_features == 0:
            return mask
        
        # Create validated mask
        validated_mask = np.zeros_like(mask)
        
        # Check each connected component
        for region_label in range(1, num_features + 1):
            region_mask = (labeled_mask == region_label)
            
            # Check if this region is close to the brightest path
            is_valid = False
            
            # Sample some points from this region
            region_coords = np.where(region_mask)
            if len(region_coords[0]) == 0:
                continue
            
            # Sample up to 20 points from the region for efficiency
            sample_size = min(20, len(region_coords[0]))
            sample_indices = np.random.choice(len(region_coords[0]), sample_size, replace=False)
            
            for idx in sample_indices:
                region_y = region_coords[0][idx]
                region_x = region_coords[1][idx]
                
                # Check distance to any path point (in 2D, ignoring z)
                for path_point in brightest_path:
                    path_y, path_x = path_point[1], path_point[2]  # [z, y, x] format
                    
                    distance = np.sqrt((region_y - path_y)**2 + (region_x - path_x)**2)
                    
                    if distance <= max_distance:
                        is_valid = True
                        break
                
                if is_valid:
                    break
            
            # Keep this region if it's valid
            if is_valid:
                validated_mask[region_mask] = 1
        
        removed_pixels = np.sum(mask) - np.sum(validated_mask)
        if removed_pixels > 0:
            print(f"Removed {removed_pixels} dendrite noise pixels through path validation")
        
        return validated_mask
    
    def create_boundary_around_path(self, path_points, min_distance=5, max_distance=15):
        """
        Create boundary around path points without spline interpolation (original working method)
        
        Args:
            path_points: List of (x, y) path coordinates
            min_distance: Inner boundary distance
            max_distance: Outer boundary distance
            
        Returns:
            inner_path, outer_path: Path objects for boundary checking
        """
        if len(path_points) < 2:
            return None, None
            
        points = np.array(path_points)
        
        # Create simplified boundary by expanding each point
        inner_points = []
        outer_points = []
        
        for i, (x, y) in enumerate(points):
            # Calculate local direction (simplified approach)
            if i == 0 and len(points) > 1:
                # Use direction to next point
                dx, dy = points[i+1] - points[i]
            elif i == len(points) - 1:
                # Use direction from previous point
                dx, dy = points[i] - points[i-1]
            else:
                # Use average direction
                dx1, dy1 = points[i] - points[i-1]
                dx2, dy2 = points[i+1] - points[i]
                dx, dy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
            
            # Normalize direction
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx / length, dy / length
            else:
                dx, dy = 1, 0
                
            # Perpendicular directions
            nx, ny = -dy, dx
            
            # Create boundary points
            inner_points.extend([
                (x + nx * min_distance, y + ny * min_distance),
                (x - nx * min_distance, y - ny * min_distance)
            ])
            
            outer_points.extend([
                (x + nx * max_distance, y + ny * max_distance),
                (x - nx * max_distance, y - ny * max_distance)
            ])
        
        # Create convex hull for smoother boundaries
        from scipy.spatial import ConvexHull
        
        try:
            if len(inner_points) >= 3:
                inner_hull = ConvexHull(inner_points)
                inner_boundary = np.array(inner_points)[inner_hull.vertices]
                inner_path = Path(inner_boundary)
            else:
                inner_path = None
                
            if len(outer_points) >= 3:
                outer_hull = ConvexHull(outer_points)
                outer_boundary = np.array(outer_points)[outer_hull.vertices]
                outer_path = Path(outer_boundary)
            else:
                outer_path = None
                
            return inner_path, outer_path
            
        except Exception as e:
            print(f"Boundary creation failed: {e}")
            return None, None
    
    def process_frame_overlapping_patches(self, frame_idx, image, brightest_path, patch_size=128):
        """
        Process a single frame with overlapping patches for dendrite segmentation
        
        Args:
            frame_idx: Index of the frame to process
            image: Input image volume
            brightest_path: List of path points [z, y, x]
            patch_size: Size of patches to process
            
        Returns:
            Predicted mask for the frame
        """
        if self.predictor is None:
            print("Model not loaded. Call load_model() first.")
            return None
            
        height, width = image[frame_idx].shape
        
        # Generate overlapping patches with 50% overlap
        stride = patch_size // 2  # 50% overlap
        patch_coords = self.generate_overlapping_patches((height, width), patch_size, stride)
        
        all_patch_masks = []
        patches_processed = 0
        patches_with_segmentation = 0
        
        # Process each overlapping patch
        for y_start, y_end, x_start, x_end in patch_coords:
            # Extract patch
            patch = image[frame_idx][y_start:y_end, x_start:x_end]
            
            # Skip if patch is too small
            if patch.shape[0] < patch_size * 0.8 or patch.shape[1] < patch_size * 0.8:
                all_patch_masks.append(None)
                continue
            
            # Find points in the current patch - using the ORIGINAL coordinates
            current_frame_points = []
            for f_idx in range(len(brightest_path)):
                f = brightest_path[f_idx]
                if f[0] == frame_idx and y_start <= f[1] < y_end and x_start <= f[2] < x_end:
                    current_frame_points.append(f)
                    
            total_frames_in_path = [i[0] for i in brightest_path] 
            frame_min, frame_max = int(min(total_frames_in_path)), int(max(total_frames_in_path))

            # Find points in nearby frames
            nearby_frame_points = []
            frame_range = 4
            for f_idx in range(len(brightest_path)):
                f = brightest_path[f_idx]
                if (frame_idx - frame_range <= f[0] <= frame_idx + frame_range and
                    y_start <= f[1] < y_end and x_start <= f[2] < x_end):
                    intensity = image[round(f[0]), round(f[1]), round(f[2])]
                    if intensity > 0.1:
                        nearby_frame_points.append(f)
                    
            # Combine unique points
            all_points = current_frame_points.copy()
            for point in nearby_frame_points:
                spatial_match = False
                for current_point in current_frame_points:
                    if point[1] == current_point[1] and point[2] == current_point[2]:
                        spatial_match = True
                        break
                if not spatial_match:
                    all_points.append(point)
            
            # Only process if we have enough points
            if len(all_points) >= 3:
                patches_processed += 1
                
                # Get original patch shape before potential resizing
                original_patch_shape = patch.shape
                
                # Resize patch to exact patch_size if needed
                if patch.shape != (patch_size, patch_size):
                    patch_resized = cv2.resize(patch, (patch_size, patch_size), cv2.INTER_LINEAR)
                    patch = patch_resized
                
                # Define range parameters for negative points
                min_distance = 5   # Minimum distance from positive points
                max_distance = 15  # Maximum distance from positive points
                
                # Sort points
                sorted_points = sorted(all_points, key=lambda p: (p[1], p[2]))
                
                # Convert coordinates to patch local coordinates
                # Subtract the patch origin and then apply scaling for resize
                path_y = [(p[1] - y_start) * (patch_size / original_patch_shape[0]) for p in sorted_points]
                path_x = [(p[2] - x_start) * (patch_size / original_patch_shape[1]) for p in sorted_points]
                
                # Set up containers for SAM points
                positive_points = []  # Points on the path (foreground)
                negative_points = []  # Points in the boundary region (background)
                
                if len(path_x) >= 3:
                    # Convert to (x, y) format for boundary creation
                    path_points_xy = list(zip(path_x, path_y))
                    
                    # Sample positive points along the path
                    if len(path_points_xy) <= 20:
                        positive_points = path_points_xy
                    else:
                        indices = np.linspace(0, len(path_points_xy)-1, 20, dtype=int)
                        positive_points = [path_points_xy[i] for i in indices]
                    
                    # Create boundary around the path (original working method)
                    inner_path, outer_path = self.create_boundary_around_path(
                        path_points_xy, min_distance, max_distance
                    )
                    
                    # Generate negative points
                    if inner_path is not None and outer_path is not None:
                        # Find bounding box of outer boundary
                        outer_vertices = outer_path.vertices
                        min_x, max_x = np.min(outer_vertices[:, 0]), np.max(outer_vertices[:, 0])
                        min_y, max_y = np.min(outer_vertices[:, 1]), np.max(outer_vertices[:, 1])
                        
                        # Generate random points in the boundary region
                        neg_count = 0
                        max_attempts = 1000
                        attempts = 0
                        
                        while neg_count < 10 and attempts < max_attempts:
                            rand_x = np.random.uniform(min_x, max_x)
                            rand_y = np.random.uniform(min_y, max_y)
                            
                            # Check bounds
                            if rand_x < 1 or rand_y < 1 or rand_x > patch_size-1 or rand_y > patch_size-1:
                                attempts += 1
                                continue
                            
                            # Check if point is between boundaries
                            if (outer_path.contains_point((rand_x, rand_y)) and 
                                not inner_path.contains_point((rand_x, rand_y))):
                                negative_points.append((rand_x, rand_y))
                                neg_count += 1
                            
                            attempts += 1
                    else:
                        # Fallback: generate negative points around positive points
                        for _ in range(10):
                            idx = np.random.randint(0, len(positive_points))
                            px, py = positive_points[idx]
                            
                            angle = np.random.uniform(0, 2*np.pi)
                            radius = np.random.uniform(min_distance, max_distance)
                            nx = px + radius * np.cos(angle)
                            ny = py + radius * np.sin(angle)
                            
                            nx = max(0, min(nx, patch_size-1))
                            ny = max(0, min(ny, patch_size-1))
                            
                            negative_points.append((nx, ny))
                
                # Generate prediction mask if we have points (original working method!)
                if positive_points and negative_points:
                    prediction_mask = self.predict_mask(patch, positive_points, negative_points)
                    
                    # Check if mask is valid and contains segmentation
                    if prediction_mask is not None:
                        # Ensure we have a binary mask (0 or 1)
                        binary_mask = (prediction_mask > 0).astype(np.uint8)
                        
                        # Resize prediction mask back to original patch size if needed
                        if binary_mask.shape != (y_end - y_start, x_end - x_start):
                            binary_mask = cv2.resize(
                                binary_mask, (x_end - x_start, y_end - y_start), 
                                interpolation=cv2.INTER_NEAREST
                            )
                        
                        # Check if there's actually segmentation in the mask
                        if np.sum(binary_mask) > 0:
                            patches_with_segmentation += 1
                            all_patch_masks.append(binary_mask)
                        else:
                            all_patch_masks.append(None)
                    else:
                        all_patch_masks.append(None)
                else:
                    all_patch_masks.append(None)
            else:
                all_patch_masks.append(None)
        
        # Merge overlapping predictions with dendrite-specific enhancement
        final_mask = self.merge_overlapping_dendrite_predictions(
            all_patch_masks, patch_coords, (height, width), brightest_path
        )
        
        print(f"Dendrite frame {frame_idx}: Processed {patches_processed} patches, {patches_with_segmentation} with segmentation")
        print(f"Overlapping patches: {len(patch_coords)} total patches")
        print(f"Final dendrite frame mask sum: {np.sum(final_mask)}")
        
        return final_mask
    
    def process_volume(self, image, brightest_path, start_frame=None, end_frame=None, patch_size=128, progress_callback=None):
        """
        Process a volume of frames using overlapping patches
        
        Args:
            image: Input image volume
            brightest_path: List of path points [z, y, x]
            start_frame: First frame to process (default: min z in path)
            end_frame: Last frame to process (default: max z in path)
            patch_size: Size of patches to process
            progress_callback: Optional callback function to report progress
            
        Returns:
            Predicted mask volume
        """
        if self.predictor is None:
            print("Model not loaded. Call load_model() first.")
            return None
            
        # Convert brightest_path to list if it's a numpy array
        if isinstance(brightest_path, np.ndarray):
            brightest_path = brightest_path.tolist()
            
        # Determine frame range from path if not provided
        if start_frame is None or end_frame is None:
            z_values = [point[0] for point in brightest_path]
            if start_frame is None:
                start_frame = int(min(z_values))
            if end_frame is None:
                end_frame = int(max(z_values))
        
        # Initialize output mask volume
        pred_masks = np.zeros((len(image), image[0].shape[0], image[0].shape[1]), dtype=np.uint8)

        # Process each frame
        total_frames = end_frame - start_frame + 1
        
        print(f"Processing dendrite segmentation with overlapping patches from frame {start_frame} to {end_frame}")
        print(f"Patch size: {patch_size}x{patch_size}, Overlap: 50% (stride={patch_size//2})")
        
        for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            if progress_callback:
                progress_callback(i, total_frames)
            else:
                print(f"Processing dendrite overlapping patches frame {frame_idx}/{end_frame} ({i+1}/{total_frames})")
                
            # Process the frame with overlapping patches
            frame_mask = self.process_frame_overlapping_patches(
                frame_idx, image, brightest_path, patch_size=patch_size
            )
            
            # Add the frame mask to the output volume
            pred_masks[frame_idx] = frame_mask
            
        # Check if we have any segmentation
        total_segmentation = np.sum(pred_masks)
        print(f"Total dendrite segmentation volume with overlapping patches: {total_segmentation} pixels")
        
        if total_segmentation == 0:
            print("WARNING: No dendrite segmentation found in any frame!")
            
        return pred_masks

    def pad_image_for_patches(self, image, patch_size=128, pad_value=0):
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