import numpy as np
import torch
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import disk, binary_closing, binary_opening, remove_small_objects
from scipy.ndimage import binary_fill_holes as ndimage_fill_holes
from scipy.ndimage import label


class SpineSegmenter:
    """Class for segmenting individual spines using SAM2 with overlapping patches"""
    
    def __init__(self, model_path="checkpoints/sam2.1_hiera_small.pt", 
                 config_path="sam2.1_hiera_s.yaml", 
                 weights_path="results/samv2_spines/spine_model.torch", 
                 device="cuda"):
        """
        Initialize the spine segmenter with overlapping patches.
        
        Args:
            model_path: Path to SAM2 model checkpoint
            config_path: Path to model configuration
            weights_path: Path to trained spine weights
            device: Device to run the model on
        """
        self.model_path = model_path
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self.predictor = None

    def load_model(self):
        """Load the spine segmentation model"""
        try:
            print(f"Loading spine segmentation model with overlapping patches from {self.model_path}")
            import sys
            sys.path.append('./Train-SAMv2')
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Use bfloat16 for memory efficiency
            torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

            # Build model and load weights
            sam2_model = build_sam2(self.config_path, self.model_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.predictor.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            
            print("Spine segmentation model with overlapping patches loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading spine segmentation model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def predict_spine_mask(self, image_patch, spine_points, dendrite_mask_patch=None):
        """
        Predict spine mask using SAM with spine center points (original working method)
        
        Args:
            image_patch: Input image patch (2D)
            spine_points: List of (x, y) spine center coordinates
            dendrite_mask_patch: Optional dendrite mask to suppress dendrite signal
            
        Returns:
            Binary mask of segmented spines
        """
        if self.predictor is None:
            print("Spine segmentation model not loaded. Call load_model() first.")
            return None
            
        if not spine_points:
            return np.zeros_like(image_patch, dtype=np.uint8)
        
        # Apply dendrite mask overlay if provided (img[mask] = 0)
        processed_patch = image_patch.copy().astype(np.float32)
        if dendrite_mask_patch is not None:
            # Set dendrite regions to 0 (suppressing dendrite signal)
            processed_patch[dendrite_mask_patch > 0] = 0
        
        # Convert to RGB format for SAM
        image_rgb = cv2.merge([processed_patch, processed_patch, processed_patch]).astype(np.float32)
        
        # Prepare points and labels (all spine centers are positive - keep it simple!)
        points = np.array(spine_points, dtype=np.float32)
        labels = np.array([1] * len(spine_points), dtype=np.int32)
        
        try:
            with torch.no_grad():
                self.predictor.set_image(image_rgb)
                pred, _, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels
                )
            
            return pred[0].astype(np.uint8)
            
        except Exception as e:
            print(f"Error in spine prediction: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros_like(image_patch, dtype=np.uint8)

    def generate_overlapping_patches(self, image_shape, patch_size=128, stride=64):
        """
        Generate overlapping patch coordinates with 50% overlap
        
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
        
        print(f"Generated {len(patches)} overlapping patches for image shape {image_shape}")
        return patches

    def merge_overlapping_predictions(self, all_patch_masks, patch_coords, image_shape, spine_positions):
        """
        Merge overlapping patch predictions with circular shape enhancement and noise removal
        
        Args:
            all_patch_masks: List of prediction masks from each patch
            patch_coords: List of (y_start, y_end, x_start, x_end) for each patch
            image_shape: (height, width) of full image
            spine_positions: List of spine positions for validation
            
        Returns:
            Final merged mask with circular spines and reduced noise
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
        
        # Apply adaptive threshold based on overlap
        # Areas with more overlap get higher threshold (more confident)
        adaptive_threshold = np.zeros_like(averaged_prediction)
        adaptive_threshold[prediction_count == 1] = 0.5  # Single patch areas
        adaptive_threshold[prediction_count == 2] = 0.4  # 2x overlap areas  
        adaptive_threshold[prediction_count == 3] = 0.35 # 3x overlap areas
        adaptive_threshold[prediction_count >= 4] = 0.3  # 4+ overlap areas (most confident)
        
        # Create initial binary mask
        binary_mask = (averaged_prediction > adaptive_threshold).astype(np.uint8)
        
        # Apply morphological operations to create circular, well-filled spines
        enhanced_mask = self.enhance_spine_shapes(binary_mask, spine_positions)
        
        print(f"Merged {len(all_patch_masks)} overlapping patches")
        print(f"Max overlap count: {np.max(prediction_count)}")
        print(f"Final mask pixels: {np.sum(enhanced_mask)}")
        
        return enhanced_mask

    def enhance_spine_shapes(self, binary_mask, spine_positions, min_spine_size=10):
        """
        Enhance spine shapes to be more circular and well-filled, remove noise
        
        Args:
            binary_mask: Initial binary segmentation mask
            spine_positions: List of spine center positions for validation
            min_spine_size: Minimum size of spine objects to keep
            
        Returns:
            Enhanced binary mask with circular spines
        """
        if not np.any(binary_mask):
            return binary_mask
        
        enhanced_mask = binary_mask.copy()
        
        # Step 1: Fill small holes inside spines
        enhanced_mask = ndimage_fill_holes(enhanced_mask).astype(np.uint8)
        
        # Step 2: Remove small noise objects
        enhanced_mask = remove_small_objects(enhanced_mask.astype(bool), min_size=min_spine_size).astype(np.uint8)
        
        # Step 3: Apply morphological closing to make spines more circular
        # Use small circular structuring element
        circular_element = disk(2)  # Small disk for gentle rounding
        enhanced_mask = binary_closing(enhanced_mask, circular_element).astype(np.uint8)
        
        # Step 4: Apply gentle opening to separate merged spines
        opening_element = disk(1)  # Very small disk to avoid over-erosion
        enhanced_mask = binary_opening(enhanced_mask, opening_element).astype(np.uint8)
        
        # Step 5: Final hole filling
        enhanced_mask = ndimage_fill_holes(enhanced_mask).astype(np.uint8)
        
        # Step 6: Validate enhanced regions against spine positions
        if spine_positions is not None and len(spine_positions) > 0:
            enhanced_mask = self.validate_spines_against_positions(enhanced_mask, spine_positions)
        
        print(f"Enhanced spine shapes: {np.sum(binary_mask)} -> {np.sum(enhanced_mask)} pixels")
        
        return enhanced_mask

    def validate_spines_against_positions(self, mask, spine_positions, max_distance=15):
        """
        Validate segmented regions against known spine positions, remove isolated noise
        
        Args:
            mask: Binary segmentation mask
            spine_positions: List of (z, y, x) spine positions
            max_distance: Maximum distance from spine position to keep a region
            
        Returns:
            Validated mask with noise removed
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
            
            # Find centroid of this region
            region_coords = np.where(region_mask)
            if len(region_coords[0]) == 0:
                continue
                
            region_centroid_y = np.mean(region_coords[0])
            region_centroid_x = np.mean(region_coords[1])
            
            # Check if this region is close to any spine position
            is_valid = False
            for spine_pos in spine_positions:
                spine_y, spine_x = spine_pos[1], spine_pos[2]  # [z, y, x] format
                
                distance = np.sqrt((region_centroid_y - spine_y)**2 + (region_centroid_x - spine_x)**2)
                
                if distance <= max_distance:
                    is_valid = True
                    break
            
            # Keep this region if it's valid
            if is_valid:
                validated_mask[region_mask] = 1
        
        removed_pixels = np.sum(mask) - np.sum(validated_mask)
        if removed_pixels > 0:
            print(f"Removed {removed_pixels} noise pixels through position validation")
        
        return validated_mask

    def process_frame_overlapping_patches(self, frame_idx, image, spine_positions, dendrite_mask=None, patch_size=128):
        """
        Process a single frame using overlapping patches with 50% overlap
        
        Args:
            frame_idx: Index of the frame to process
            image: Input image volume
            spine_positions: List of spine positions for this frame
            dendrite_mask: Optional dendrite segmentation mask volume
            patch_size: Size of patches to process (128x128)
            
        Returns:
            Predicted spine mask for the frame
        """
        if self.predictor is None:
            print("Model not loaded. Call load_model() first.")
            return None
            
        # Get frame-specific spine positions
        frame_spines = [pos for pos in spine_positions if int(pos[0]) == frame_idx]
        
        if not frame_spines:
            return np.zeros_like(image[frame_idx], dtype=np.uint8)
            
        height, width = image[frame_idx].shape
        
        # Get frame image and normalize
        frame_img = (image[frame_idx] - image[frame_idx].min()) / (image[frame_idx].max() - image[frame_idx].min())
        
        # Get dendrite mask for this frame if available
        frame_dendrite_mask = None
        if dendrite_mask is not None and frame_idx < len(dendrite_mask):
            frame_dendrite_mask = dendrite_mask[frame_idx]
        
        # Generate overlapping patches with 50% overlap (stride = 64)
        stride = patch_size // 2  # 50% overlap
        patch_coords = self.generate_overlapping_patches((height, width), patch_size, stride)
        
        all_patch_masks = []
        patches_processed = 0
        patches_with_segmentation = 0
        
        # Process each overlapping patch
        for y_start, y_end, x_start, x_end in patch_coords:
            # Extract patch
            patch = frame_img[y_start:y_end, x_start:x_end]
            
            # Skip if patch is too small
            if patch.shape[0] < patch_size * 0.8 or patch.shape[1] < patch_size * 0.8:
                all_patch_masks.append(None)
                continue
            
            # Extract dendrite mask patch if available
            dendrite_patch = None
            if frame_dendrite_mask is not None:
                dendrite_patch = frame_dendrite_mask[y_start:y_end, x_start:x_end]
            
            # Find spine positions in this patch
            patch_spine_points = []
            for spine_pos in frame_spines:
                spine_y, spine_x = spine_pos[1], spine_pos[2]
                
                # Check if spine is in this patch
                if y_start <= spine_y < y_end and x_start <= spine_x < x_end:
                    # Convert to patch-local coordinates
                    local_y = spine_y - y_start
                    local_x = spine_x - x_start
                    patch_spine_points.append((local_x, local_y))
            
            # Process patch if it contains spines
            if patch_spine_points:
                patches_processed += 1
                
                # Resize patch to exact patch_size if needed
                if patch.shape != (patch_size, patch_size):
                    original_shape = patch.shape
                    patch_resized = cv2.resize(patch, (patch_size, patch_size), cv2.INTER_LINEAR)
                    
                    # Resize dendrite patch if needed
                    dendrite_patch_resized = None
                    if dendrite_patch is not None:
                        dendrite_patch_resized = cv2.resize(
                            dendrite_patch.astype(np.uint8), (patch_size, patch_size), cv2.INTER_NEAREST
                        )
                    
                    # Scale spine coordinates
                    scaled_spine_points = []
                    for px, py in patch_spine_points:
                        scaled_x = px * (patch_size / original_shape[1])
                        scaled_y = py * (patch_size / original_shape[0])
                        scaled_spine_points.append((scaled_x, scaled_y))
                    
                    patch = patch_resized
                    dendrite_patch = dendrite_patch_resized
                    patch_spine_points = scaled_spine_points
                
                # Predict spine mask for this patch (original working method!)
                patch_mask = self.predict_spine_mask(patch, patch_spine_points, dendrite_patch)
                
                if patch_mask is not None and np.sum(patch_mask) > 0:
                    patches_with_segmentation += 1
                    
                    # Resize back to original patch size if needed
                    if patch_mask.shape != (y_end - y_start, x_end - x_start):
                        target_h = y_end - y_start
                        target_w = x_end - x_start
                        patch_mask = cv2.resize(patch_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    
                    all_patch_masks.append(patch_mask)
                else:
                    all_patch_masks.append(None)
            else:
                all_patch_masks.append(None)
        
        # Merge overlapping predictions with circular enhancement
        final_mask = self.merge_overlapping_predictions(
            all_patch_masks, patch_coords, (height, width), frame_spines
        )
        
        print(f"Frame {frame_idx}: Processed {patches_processed} patches, {patches_with_segmentation} with segmentation")
        print(f"Overlapping patches: {len(patch_coords)} total patches")
        print(f"Final frame mask sum: {np.sum(final_mask)}")
        
        return final_mask

    def process_volume_spines(self, image, spine_positions, dendrite_mask=None, patch_size=128, progress_callback=None):
        """
        Process a volume for spine segmentation using overlapping patches
        
        Args:
            image: Input image volume
            spine_positions: List of spine positions [z, y, x]
            dendrite_mask: Optional dendrite segmentation mask volume
            patch_size: Size of patches to process (128x128)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Predicted spine mask volume
        """
        if self.predictor is None:
            print("Model not loaded. Call load_model() first.")
            return None
        
        # Convert spine positions to numpy array if needed
        if isinstance(spine_positions, list):
            spine_positions = np.array(spine_positions)
        
        # Determine frame range from spine positions
        if len(spine_positions) == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        z_values = spine_positions[:, 0].astype(int)
        start_frame = max(0, int(np.min(z_values)))
        end_frame = min(len(image) - 1, int(np.max(z_values)))
        
        # Initialize output mask volume
        pred_masks = np.zeros_like(image, dtype=np.uint8)
        
        total_frames = end_frame - start_frame + 1
        
        print(f"Processing spine segmentation with overlapping patches from frame {start_frame} to {end_frame}")
        print(f"Patch size: {patch_size}x{patch_size}, Overlap: 50% (stride={patch_size//2})")
        if dendrite_mask is not None:
            print(f"Using dendrite mask overlay: dendrite signal will be suppressed")
        
        # Process each frame
        for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            if progress_callback:
                progress_callback(i, total_frames)
            else:
                print(f"Processing overlapping patches frame {frame_idx}/{end_frame} ({i+1}/{total_frames})")
            
            # Process the frame with overlapping patches
            frame_mask = self.process_frame_overlapping_patches(
                frame_idx, image, spine_positions, dendrite_mask, patch_size
            )
            
            if frame_mask is not None:
                pred_masks[frame_idx] = frame_mask
        
        total_segmentation = np.sum(pred_masks)
        print(f"Total spine segmentation volume with overlapping patches: {total_segmentation} pixels")
        
        return pred_masks

    # Keep compatibility methods
    def detect_spine_centers_from_array(self, mask, min_distance=3):
        """Detect spine centers from binary mask"""
        from scipy.ndimage import label, center_of_mass
        from scipy.spatial.distance import pdist, squareform
        
        # Label connected components
        labeled_mask, num_features = label(mask)
        
        if num_features == 0:
            return [], []
        
        # Get centers of mass
        centers = center_of_mass(mask, labeled_mask, range(1, num_features + 1))
        centers = np.array(centers)
        
        if len(centers) == 0:
            return [], []
        
        # Filter centers based on minimum distance
        if len(centers) > 1:
            distances = pdist(centers)
            distance_matrix = squareform(distances)
            
            # Keep only centers that are far enough apart
            keep_indices = []
            for i in range(len(centers)):
                if i == 0:
                    keep_indices.append(i)
                else:
                    min_dist_to_kept = min([distance_matrix[i, j] for j in keep_indices])
                    if min_dist_to_kept >= min_distance:
                        keep_indices.append(i)
            
            centers = centers[keep_indices]
        
        # Convert to (x, y) format and create labels
        spine_points = [(center[1], center[0]) for center in centers]  # (x, y)
        spine_labels = [1] * len(spine_points)  # All positive points
        
        return spine_points, spine_labels

    def pad_image_for_patches(self, image, patch_size=128, pad_value=0):
        """Pad image to be divisible by patch_size"""
        h, w = image.shape[:2]
        
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        
        if image.ndim == 3:
            padding = ((0, pad_h), (0, pad_w), (0, 0))
        else:
            padding = ((0, pad_h), (0, pad_w))
        
        padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
        return padded_image, (pad_h, pad_w), (h, w)