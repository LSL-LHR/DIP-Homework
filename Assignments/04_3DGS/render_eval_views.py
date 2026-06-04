import argparse
import os
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from data_utils import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer


def save_rgb(path, image):
    image = (image * 255.0).clip(0, 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the simplified 3DGS checkpoint at original COLMAP camera views."
    )
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval/simple")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_views", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    render_dir = output_dir / "renders"
    gt_dir = output_dir / "gt"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    dataset = ColmapDataset(args.colmap_dir)
    sample = dataset[0]
    height, width = sample["image"].shape[:2]

    model = GaussianModel(dataset.points3D_xyz, dataset.points3D_rgb).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    renderer = GaussianRenderer(height, width).to(device)
    with torch.no_grad():
        gaussian_params = model()

    num_views = len(dataset) if args.max_views is None else min(args.max_views, len(dataset))
    for idx in tqdm(range(num_views), desc="Rendering eval views"):
        item = dataset[idx]
        name = Path(item["image_path"]).stem + ".png"

        K = item["K"].to(device)
        R = item["R"].to(device)
        t = item["t"].reshape(3).to(device)

        with torch.no_grad():
            rendered = renderer(
                means3D=gaussian_params["positions"],
                covs3d=gaussian_params["covariance"],
                colors=gaussian_params["colors"],
                opacities=gaussian_params["opacities"],
                K=K,
                R=R,
                t=t,
            )

        save_rgb(render_dir / name, rendered.cpu().numpy())
        save_rgb(gt_dir / name, item["image"].cpu().numpy())

    print(f"Saved renders to: {render_dir}")
    print(f"Saved GT images to: {gt_dir}")


if __name__ == "__main__":
    main()
