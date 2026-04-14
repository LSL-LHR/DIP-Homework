## # Poisson Image Editing and Pix2Pix Implementation

This repository contains two parts:

1. **Poisson Image Editing (PyTorch + Gradio)**
2. **Pix2Pix Image-to-Image Translation (Fully Convolutional Network)**

---

## ## Overview

### 🔹 Part 1: Poisson Image Editing

We implement **Poisson Image Blending** using PyTorch optimization.

Key components:

* Polygon-based mask generation
* Laplacian constraint formulation
* Gradient-domain blending via optimization

You completed:

* `create_mask_from_points` (Polygon → Mask)
* `cal_laplacian_loss` (Laplacian consistency loss)

---

### 🔹 Part 2: Pix2Pix Implementation

We implement a **Fully Convolutional Encoder–Decoder Network** for image-to-image translation.

Key features:

* Downsampling via Conv layers
* Upsampling via ConvTranspose layers
* Output normalized to ([-1,1]) using `Tanh`

---

## ## Requirements

```bash
pip install torch torchvision numpy pillow gradio opencv-python
```

---

## ## Running Poisson Image Editing

```bash
python run_blending_gradio.py
```

Then open the Gradio interface in browser.

### Usage:

1. Upload foreground image
2. Click to define polygon
3. Close polygon
4. Upload background image
5. Adjust (dx, dy)
6. Click **Blend Images**

---

## ## Method Details

### 🔹 Mask Generation

Polygon is converted into a binary mask:

* Inside polygon → 255
* Outside → 0

Implemented using OpenCV:

```python
cv2.fillPoly(mask, [pts], 255)
```

---

### 🔹 Laplacian Loss

We enforce **gradient consistency** between foreground and blended image.

Using discrete Laplacian:

\nabla^2 I = I_{x+1,y} + I_{x-1,y} + I_{x,y+1} + I_{x,y-1} - 4I_{x,y}

In code:

* Implemented via `torch.nn.functional.conv2d`
* Applied channel-wise (groups=3)
* Loss:

[
\mathcal{L} = | \nabla^2 I_{fg} - \nabla^2 I_{blend} |^2
]

Only computed inside mask.

---

### 🔹 Optimization

* Variable: blended image
* Optimizer: Adam
* Iterations: 5000
* Learning rate decay at 2/3 steps

---

## ## Pix2Pix Model

### Architecture

Encoder:

* Conv → BN → ReLU
* Channels: 3 → 256
* Downsampling ×6

Decoder:

* ConvTranspose → BN → ReLU
* Channels: 256 → 3

Final activation:

```python
Tanh()
```

---

### Forward Pass

```python
x → Encoder → Bottleneck → Decoder → Output
```

No skip connections (simplified Pix2Pix).

---

## ## Training (Pix2Pix)

```bash
python train.py
```

（如果你没写 train.py，可以写成说明性内容）

Typical settings:

* Loss: L1 / MSE
* Optimizer: Adam
* Learning rate: 1e-4

---

## ## Results

### Poisson Blending

* Smooth transition at boundary
* Gradient consistency preserved
* No visible seams

### Pix2Pix

| Model               | Output                              |
| ------------------- | ----------------------------------- |
| FCN Encoder-Decoder | Reasonable structure reconstruction |

（你可以放截图）

---

## ## Key Insights

### Poisson Editing

* Works in **gradient domain**
* Avoids intensity discontinuity
* Optimization-based (slow but accurate)

### Pix2Pix

* Learns **mapping instead of solving PDE**
* Faster inference
* Depends on training data

---





