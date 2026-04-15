## # Poisson Image Editing and Pix2Pix Implementation

This repository contains two parts:

1. **Implement Poisson Image Editing with PyTorch.**
2. **Pix2Pix Implementation**

---

## ## Requirements

To install a virtual environment:

```bash
conda create -n dip_env python=3.11
conda activate dip_env
```

To install requirements:

```bash
pip install torch torchvision numpy pillow gradio opencv-python
```

To contain more images for better generalization on the validation set:

```bash
bash download_facades_dataset.sh
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
* Iterations(**iter_num**): 5000
* Learning rate decay at 2/3 steps

---

###   🔹 Cases

![alt](pics\monolisa.png)

 ![alt](pics\water.png)

 ![alt](pics\equation.png)

---



## ## Pix2Pix Model

### Train the model on the [pix2pix](https://github.com/phillipi/pix2pix#datasets).

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

---

## ## Training (Pix2Pix) and Evaluation

**Train on a larger Pix2Pix dataset and generate 'val_results'.**

```bash
python train.py
```

Typical settings:

* Loss: L1 / MSE
* Optimizer: Adam
* Learning rate: 1e-3

---