## Problem

The goal of this implementation is to create a **LeNet-inspired sliding-window convolution** that manually applies a **3x3 kernel** to a 2D image. This helps in understanding spatial feature extraction, which is fundamental in convolutional neural networks (CNNs). The implementation uses **loops** instead of built-in convolution functions to reinforce the concept of local receptive fields and feature detection.

---

## Knowledge

### 1. **LeNet Architecture**

LeNet is a pioneering CNN architecture introduced by Yann LeCun in 1989. It consists of:
- **Convolutional Layers** for feature extraction using small filters.
- **Activation Functions** (ReLU or Tanh) to introduce non-linearity.
- **Pooling Layers** for dimensionality reduction.
- **Fully Connected Layers** for classification.

In this implementation, we focus on the first step: **convolution** using a manually implemented sliding-window approach.

---

### 2. **Mathematical Formulation**

#### **Convolution Operation:**
A 2D convolution applies a kernel \( K \) to an image \( I \), producing an output \( O \):

\[
O(i, j) = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I(i+m, j+n) \cdot K(m, n)
\]

where:
- \( k_h \), \( k_w \) are the height and width of the kernel.
- \( i, j \) are the coordinates of the sliding window.
- \( I(i+m, j+n) \) represents the pixel values within the receptive field.
- \( K(m, n) \) represents the weights of the kernel.

#### **Edge Detection Example:**
A **Sobel X filter** for edge detection is defined as:

\[
K = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}
\]

When applied to an image, it highlights vertical edges by computing differences in pixel intensity.
