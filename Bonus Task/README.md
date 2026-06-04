# Exprement-in-AUTOENCODER
For your GitHub README, I'd focus on the **experimentation journey**, not just "I built an autoencoder."

# Fashion-MNIST Autoencoder: Exploring Decoder Activations and Compression

## Overview

This project implements an Autoencoder using PyTorch on the Fashion-MNIST dataset.

The primary objective was to compress input images into a low-dimensional latent representation and reconstruct them back while preserving as much visual information as possible.

Beyond implementation, this project evolved into a series of experiments investigating:

* Decoder output activation functions
* Reconstruction quality
* Compression vs reconstruction trade-offs
* Bottleneck design choices
* Latent space representation

---

## Dataset

Dataset Used:

* Fashion-MNIST
* 60,000 training images
* 10,000 test images
* Grayscale images
* Image size: 28 × 28

Categories include:

* T-shirts
* Shirts
* Pullovers
* Dresses
* Coats
* Sandals
* Sneakers
* Bags
* Ankle Boots

---

## Model Architecture

### Encoder

```text
784
↓
256
↓
128
↓
32
```

The encoder compresses the original image into a 32-dimensional latent representation.

### Decoder

```text
32
↓
128
↓
256
↓
784
```

The decoder reconstructs the image from the compressed latent representation.

---

## Compression Analysis

The original image contains:

```text
28 × 28 = 784 pixels
```

The bottleneck contains:

```text
32 latent features
```

Compression ratio:

```text
784 / 32 ≈ 24.5x
```

This means the model learns to represent an image using only a small fraction of the original dimensionality.

---

## Experiments Performed

### Experiment 1: Sigmoid Output

Baseline decoder output:

```python
nn.Sigmoid()
```

Result:

```text
Loss = 0.0102
```

---

### Experiment 2: Tanh + LeakyReLU

Hypothesis:

* Tanh can represent positive and negative feature signals
* LeakyReLU preserves gradient flow

Decoder output:

```python
nn.Tanh()
nn.LeakyReLU()
```

Result:

```text
Loss = 0.0094
```

Observation:

* Better reconstruction than Sigmoid
* Cleaner output images

---

### Experiment 3: LeakyReLU Only

Decoder output:

```python
nn.LeakyReLU()
```

Result:

```text
Loss = 0.0091
```

Observation:

* Best reconstruction loss among all tested activations
* Cleaner backgrounds
* Slightly sharper reconstructed images

This result contradicted the initial hypothesis that Tanh + LeakyReLU would perform best.

---

### Experiment 4: No Output Activation

Decoder output:

```python
nn.Linear(...)
```

Result:

```text
Loss = 0.0100
```

Observation:

* Slightly better than Sigmoid
* Demonstrated that MSE loss alone can guide outputs toward the correct image range

---

## Key Findings

| Output Activation | Reconstruction Loss |
| ----------------- | ------------------- |
| Sigmoid           | 0.0102              |
| No Activation     | 0.0100              |
| Tanh + LeakyReLU  | 0.0094              |
| LeakyReLU         | 0.0091              |

Best Performing Configuration:

```text
LeakyReLU Output Layer
```

---

## What I Learned

This project helped me understand:

* Autoencoders as compression systems
* Latent space representations
* Feature extraction and reconstruction
* The role of activation functions in decoder design
* Why experiments are often more reliable than assumptions
* Trade-offs between compression and reconstruction quality

One of the most surprising outcomes was that the commonly used Sigmoid output layer was not the best-performing configuration in this architecture.
---

## Technologies Used

* Python
* PyTorch
* Torchvision
* Matplotlib
* Google Colab

---

This project started as an implementation exercise and evolved into an exploration of how architectural decisions influence reconstruction quality and learned representations in autoencoders. 🚀

This is the kind of README that makes recruiters or other students immediately understand **what you explored, what you found, and what you learned**, instead of just seeing a notebook dump.
