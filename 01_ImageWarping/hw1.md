# DIP-Teaching: Digital Image Processing Assignments

This repository contains implementations of classic digital image processing and deep learning-based image synthesis techniques, including:
- Global Image Transformation (Scale, Rotation, Translation)
- Point-based Image Deformation (MLS)
- Poisson Image Editing
- Pix2Pix for Image-to-Image Translation

## Requirements

### Environment Setup (Recommended: Conda)

```bash
# 1. Create and activate Python 3.10 environment
conda create -n dip_env python=3.10 -y
conda activate dip_env

# 2. Install dependencies
pip install torch torchvision opencv-python gradio pillow numpy

