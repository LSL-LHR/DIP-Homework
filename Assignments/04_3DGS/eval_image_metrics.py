import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ssim(pred, gt):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores = []
    for channel in range(3):
        x = pred[..., channel]
        y = gt[..., channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        scores.append(float(np.mean(ssim_map)))
    return float(np.mean(scores))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image quality metrics for matched image folders.")
    parser.add_argument("--renders", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument(
        "--normalize_background",
        choices=["none", "black", "white"],
        default="none",
        help="Use non-black GT pixels as a foreground mask and set background to a fixed color before scoring.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.02,
        help="Foreground threshold on max GT RGB value when --normalize_background is enabled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    render_dir = Path(args.renders)
    gt_dir = Path(args.gt)

    render_paths = sorted(render_dir.glob("*.png"))
    if not render_paths:
        raise RuntimeError(f"No PNG files found in {render_dir}")

    rows = []
    for render_path in render_paths:
        gt_path = gt_dir / render_path.name
        if not gt_path.exists():
            print(f"Skipping {render_path.name}: missing GT")
            continue

        pred = read_rgb(render_path)
        gt = read_rgb(gt_path)
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)

        if args.normalize_background != "none":
            mask = np.max(gt, axis=-1, keepdims=True) > args.mask_threshold
            bg = 1.0 if args.normalize_background == "white" else 0.0
            pred = np.where(mask, pred, bg)
            gt = np.where(mask, gt, bg)

        rows.append(
            {
                "name": render_path.name,
                "psnr": psnr(pred, gt),
                "ssim": ssim(pred, gt),
                "l1": float(np.mean(np.abs(pred - gt))),
            }
        )

    if not rows:
        raise RuntimeError("No matched render/GT image pairs found.")

    mean_psnr = float(np.mean([row["psnr"] for row in rows]))
    mean_ssim = float(np.mean([row["ssim"] for row in rows]))
    mean_l1 = float(np.mean([row["l1"] for row in rows]))

    print(f"Pairs: {len(rows)}")
    print(f"PSNR: {mean_psnr:.4f} dB")
    print(f"SSIM: {mean_ssim:.4f}")
    print(f"L1:   {mean_l1:.6f}")


if __name__ == "__main__":
    main()
