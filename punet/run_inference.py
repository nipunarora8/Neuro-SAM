
import argparse
from pathlib import Path

import numpy as np
import torch
import tifffile as tiff
from tqdm import tqdm

from prob_unet_with_tversky import ProbabilisticUnetDualLatent


def pad_to_multiple(img: np.ndarray, multiple: int = 32):
    """Pad HxW to next multiple with reflect to keep context."""
    H, W = img.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    img_p = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
    return img_p, (pad_h, pad_w)


@torch.no_grad()
def infer_slice(model, device, img_2d: np.ndarray, mc_samples: int = 8):
    """
    img_2d: float32 in [0,1], shape HxW
    returns: (prob_dend, prob_spine) each HxW float32
    """
    # pad for UNet down/upsampling safety
    x_np, (ph, pw) = pad_to_multiple(img_2d, multiple=32)
    # to tensor [B,C,H,W] = [1,1,H,W]
    x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)


    model.forward(x, training=False)

    # Multi-sample averaging from prior
    pd_list, ps_list = [], []
    for _ in range(max(1, mc_samples)):
        ld, ls = model.sample(testing=True, use_posterior=False)
        pd_list.append(torch.sigmoid(ld))
        ps_list.append(torch.sigmoid(ls))

    pd = torch.stack(pd_list, 0).mean(0)  # [1,1,H,W]
    ps = torch.stack(ps_list, 0).mean(0)

    # back to numpy, remove padding
    pd_np = pd.squeeze().float().cpu().numpy()
    ps_np = ps.squeeze().float().cpu().numpy()
    if ph or pw:
        pd_np = pd_np[: pd_np.shape[0] - ph, : pd_np.shape[1] - pw]
        ps_np = ps_np[: ps_np.shape[0] - ph, : ps_np.shape[1] - pw]
    return pd_np.astype(np.float32), ps_np.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Inference on DeepD3_Benchmark.tif with Dual-Latent Prob-UNet")
    ap.add_argument("--weights", required=True, help="Path to checkpoint .pth (with model_state_dict)")
    ap.add_argument("--tif", required=True, help="Path to DeepD3_Benchmark.tif")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--samples", type=int, default=16, help="MC samples per slice (default: 8)")
    ap.add_argument("--thr_d", type=float, default=0.5, help="Threshold for dendrite mask save")
    ap.add_argument("--thr_s", type=float, default=0.5, help="Threshold for spine mask save")
    ap.add_argument("--save_bin", action="store_true", help="Also save thresholded uint8 masks")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = ProbabilisticUnetDualLatent(
        input_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        latent_dim_dendrite=12,
        latent_dim_spine=12,
        no_convs_fcomb=4,
        recon_loss="tversky",        
        tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=1.0,
        beta_dendrite=1.0, beta_spine=1.0,
        loss_weight_dendrite=1.0, loss_weight_spine=1.0,
    ).to(device)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) 
    model.load_state_dict(state, strict=True)

    print(f"Reading: {args.tif}")
    vol = tiff.imread(args.tif)  # shape: (Z,H,W) or (H,W)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    Z, H, W = vol.shape
    print(f"Volume shape: Z={Z}, H={H}, W={W}")

    # Output arrays (float32)
    prob_d = np.zeros((Z, H, W), dtype=np.float32)
    prob_s = np.zeros((Z, H, W), dtype=np.float32)

    # ----- Run inference per slice -----
    for z in tqdm(range(Z), desc="Inferring"):
        img = vol[z].astype(np.float32)
        # per-slice min-max normalize , avoid div by zero
        vmin, vmax = float(img.min()), float(img.max())
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        pd, ps = infer_slice(model, device, img, mc_samples=args.samples)
        prob_d[z] = pd
        prob_s[z] = ps

    prob_d_path = outdir / "DeepD3_Benchmark_prob_dendrite.tif"
    prob_s_path = outdir / "DeepD3_Benchmark_prob_spine.tif"
    tiff.imwrite(prob_d_path.as_posix(), prob_d, dtype=np.float32)
    tiff.imwrite(prob_s_path.as_posix(), prob_s, dtype=np.float32)
    print(f"Saved: {prob_d_path}")
    print(f"Saved: {prob_s_path}")

    if args.save_bin:
        bin_d = (prob_d >= args.thr_d).astype(np.uint8) * 255
        bin_s = (prob_s >= args.thr_s).astype(np.uint8) * 255
        bin_d_path = outdir / f"DeepD3_Benchmark_mask_dendrite_thr{args.thr_d:.2f}.tif"
        bin_s_path = outdir / f"DeepD3_Benchmark_mask_spine_thr{args.thr_s:.2f}.tif"
        tiff.imwrite(bin_d_path.as_posix(), bin_d, dtype=np.uint8)
        tiff.imwrite(bin_s_path.as_posix(), bin_s, dtype=np.uint8)
        print(f"Saved: {bin_d_path}")
        print(f"Saved: {bin_s_path}")

    print("Done.")


if __name__ == "__main__":
    main()
