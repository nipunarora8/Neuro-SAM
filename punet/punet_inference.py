"""
Run inference with a trained Probabilistic U-Net model for dendrite and spine segmentation.
Refactored for library usage.
"""
import argparse
from pathlib import Path
from typing import Union, Dict, Tuple, Optional, Callable
import numpy as np
import torch
import tifffile as tiff
from tqdm import tqdm
from scipy.ndimage import label as cc_label

from prob_unet_with_tversky import ProbabilisticUnetDualLatent

# --- Helper Functions ---

def filter_small_objects_3d(binary_mask: np.ndarray, min_size_voxels: int = 40) -> np.ndarray:
    """
    Remove small connected components from a 3D binary mask.
    """
    labeled, num_features = cc_label(binary_mask)
    
    if num_features == 0:
        return binary_mask
    
    sizes = np.bincount(labeled.ravel())[1:]  # Skip background
    keep_labels = np.where(sizes >= min_size_voxels)[0] + 1
    
    mask_filtered = np.isin(labeled, keep_labels)
    return mask_filtered.astype(binary_mask.dtype)


def minmax01(im: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    vmin, vmax = float(im.min()), float(im.max())
    if vmax > vmin:
        im = (im - vmin) / (vmax - vmin)
    else:
        im = np.zeros_like(im, dtype=np.float32)
    return im.astype(np.float32)


def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_stack(path: Path, arr: np.ndarray, dtype):
    tiff.imwrite(path.as_posix(), arr.astype(dtype))


@torch.no_grad()
def infer_logits_mean(model, device, img_2d: np.ndarray, mc_samples: int, 
                      use_posterior: bool, temperature: float = 1.0):
    """
    Return mean logits over multiple samples for a single 2D slice.
    """
    H, W = img_2d.shape
    pad_h = (32 - (H % 32)) % 32
    pad_w = (32 - (W % 32)) % 32
    
    if pad_h or pad_w:
        x_np = np.pad(img_2d, ((0, pad_h), (0, pad_w)), mode="reflect")
    else:
        x_np = img_2d

    x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
    # Forward pass to encode
    model.forward(x, training=False)

    d_logits_list, s_logits_list = [], []
    S = max(1, mc_samples)
    
    for _ in range(S):
        ld, ls = model.sample(testing=True, use_posterior=use_posterior)
        d_logits_list.append(ld)
        s_logits_list.append(ls)

    d_stack = torch.stack(d_logits_list, 0)
    s_stack = torch.stack(s_logits_list, 0)
    
    # Apply temperature scaling before averaging
    d_mean = (d_stack.mean(0) / temperature).squeeze(0).squeeze(0)
    s_mean = (s_stack.mean(0) / temperature).squeeze(0).squeeze(0)

    if pad_h or pad_w:
        d_mean = d_mean[:H, :W]
        s_mean = s_mean[:H, :W]
    
    return d_mean, s_mean


# --- Main Inference Function ---

def run_inference_volume(
    image_input: Union[str, Path, np.ndarray],
    weights_path: Union[str, Path],
    device: Optional[str] = None,
    samples: int = 24,
    posterior: bool = False,
    temperature: float = 1.4,
    threshold: float = 0.5,
    min_size_voxels: int = 40,
    verbose: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Dict[str, np.ndarray]:
    """
    Run inference on a 3D volume (or 2D image) using the Probabilistic U-Net.

    Args:
        image_input: Path to TIF file or numpy array (Z, H, W).
        weights_path: Path to .pth model checkpoint.
        device: 'cuda' or 'cpu'. If None, auto-detects.
        samples: Number of Monte Carlo samples per slice.
        posterior: Whether to use posterior sampling (requires truth, usually False for inference).
        temperature: Temperature scaling factor (higher = softer probs).
        threshold: Binarization threshold for masks.
        min_size_voxels: Minimum object size for filtering.
        verbose: Print progress bars and info.
        progress_callback: Optional function(float) called with progress 0.0-1.0.

    Returns:
        Dictionary containing:
        - 'prob_dendrite': (Z, H, W) float32 [0-1]
        - 'prob_spine': (Z, H, W) float32 [0-1]
        - 'mask_dendrite': (Z, H, W) uint8 {0, 1} (filtered)
        - 'mask_spine': (Z, H, W) uint8 {0, 1} (filtered)
    """
    
    # 1. Setup Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
        
    if verbose:
        print(f"--- Inference Settings ---")
        print(f"Device: {device}")
        print(f"Weights: {weights_path}")
        print(f"Samples: {samples} | Temp: {temperature} | Thr: {threshold}")

    # 2. Load Data
    if isinstance(image_input, (str, Path)):
        if verbose: print(f"Loading image from {image_input}...")
        vol = tiff.imread(str(image_input))
    else:
        vol = image_input

    # Ensure 3D (Z, H, W)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    
    Z, H, W = vol.shape

    # 3. Load Model
    # Note: Architecture arguments must match training. 
    # If these vary, you might need to pass them in or load from a config.
    model = ProbabilisticUnetDualLatent(
        input_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        latent_dim_dendrite=12,
        latent_dim_spine=12,
        no_convs_fcomb=4,
        recon_loss="tversky", # Loss params don't matter for inference, but init requires them
        tversky_alpha=0.3, 
        tversky_beta=0.7,
        tversky_gamma=1.0,
        beta_dendrite=1.0,
        beta_spine=1.0,
        loss_weight_dendrite=1.0,
        loss_weight_spine=1.0,
    ).to(device)
    
    model.eval()

    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)

    # 4. Inference Loop
    prob_d = np.zeros((Z, H, W), dtype=np.float32)
    prob_s = np.zeros((Z, H, W), dtype=np.float32)
    
    iterator = range(Z)
    if verbose:
        iterator = tqdm(iterator, desc="Inferring Slices")

    for z in iterator:
        img_slice = minmax01(vol[z].astype(np.float32))
        
        d_logits, s_logits = infer_logits_mean(
            model, device, img_slice, samples, posterior, temperature
        )

        prob_d[z] = torch.sigmoid(d_logits).float().cpu().numpy()
        prob_s[z] = torch.sigmoid(s_logits).float().cpu().numpy()
        
        if progress_callback is not None:
            progress_callback((z + 1) / Z)

    # 5. Post-processing (Thresholding & Filtering)
    if verbose: print("Post-processing masks...")
    
    # Binarize
    mask_d = (prob_d >= threshold)
    mask_s = (prob_s >= threshold)

    # Filter small objects
    mask_d = filter_small_objects_3d(mask_d, min_size_voxels)
    mask_s = filter_small_objects_3d(mask_s, min_size_voxels)

    return {
        "prob_dendrite": prob_d,
        "prob_spine": prob_s,
        "mask_dendrite": mask_d.astype(np.uint8),
        "mask_spine": mask_s.astype(np.uint8)
    }


# --- CLI Wrapper ---

def main():
    ap = argparse.ArgumentParser(description="Run inference via CLI")
    
    ap.add_argument("--weights", required=True, help="Model checkpoint path")
    ap.add_argument("--tif", required=True, help="Input TIF (Z,H,W) or (H,W)")
    ap.add_argument("--out", default="inference_results_punet/", help="Output directory")
    
    ap.add_argument("--posterior", action="store_true", help="Use posterior sampling")
    ap.add_argument("--samples", type=int, default=24, help="MC samples")
    ap.add_argument("--temp", type=float, default=1.4, help="Temperature scaling")
    ap.add_argument("--thr", type=float, default=0.50, help="Prediction threshold")
    ap.add_argument("--min_size", type=int, default=40, help="Minimum object size (voxels)")
    
    args = ap.parse_args()

    outdir = Path(args.out)
    ensure_outdir(outdir)

    # Call the reusable function
    results = run_inference_volume(
        image_input=args.tif,
        weights_path=args.weights,
        samples=args.samples,
        posterior=args.posterior,
        temperature=args.temp,
        threshold=args.thr,
        min_size_voxels=args.min_size,
        verbose=True
    )

    print("\nSaving results...")
    
    # Save Probabilities
    save_stack(outdir / "prob_dendrite.tif", results['prob_dendrite'], np.float32)
    save_stack(outdir / "prob_spine.tif", results['prob_spine'], np.float32)
    
    # Save Masks
    tag = f"{'post' if args.posterior else 'prior'}_S{args.samples}_T{args.temp:.2f}_size{args.min_size}"
    
    save_stack(
        outdir / f"mask_spine_{tag}_thr{args.thr:.2f}.tif", 
        results['mask_spine'] * 255, 
        np.uint8
    )
    save_stack(
        outdir / f"mask_dendrite_{tag}_thr{args.thr:.2f}.tif", 
        results['mask_dendrite'] * 255, 
        np.uint8
    )
    
    print(f"Done! Results saved to: {outdir}")

if __name__ == "__main__":
    main()