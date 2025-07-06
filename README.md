
# Swin Transformer for Parameter Regression: Model Architecture Explained

This document provides a detailed explanation of the Swin Transformer architecture as implemented in this project for the task of regressing six physical parameters (`Rc`, `H30`, `incl`, `mDisk`, `gamma`, `psi`) from input images.

## 1. High-Level Overview

The model is a **Vision Transformer**, specifically a **Swin Transformer**, which is adapted for a **regression task**. Instead of classifying an image into a category, it predicts six continuous values.

The core idea is to:
1.  Divide the input image into small, non-overlapping **patches**.
2.  Treat these patches like "words" in a sentence and process them with a powerful **Transformer** architecture.
3.  Use a hierarchical approach that merges patches to create representations at different scales, similar to how a Convolutional Neural Network (CNN) has pooling layers.
4.  Finally, take the learned features and pass them through specialized **prediction heads** to get the six output values.

---

## 2. Detailed Architecture Breakdown

The model can be broken down into three main parts:
1.  **Patch Embedding**: The initial processing of the input image.
2.  **Swin Transformer Backbone**: The main feature extractor.
3.  **Regression Head**: The final output stage.

### 2.1. Patch Embedding (`PatchEmbed` class)

This is the first layer of the model. Its job is to convert the input image into a sequence of embeddings (vectors) that the Transformer can understand.

-   **Input**: An image tensor of shape `(B, C, H, W)`, where `B` is the batch size, `C` is the number of input channels (e.g., 2), `H` is the height, and `W` is the width.
-   **Process**:
    1.  A **Convolutional Layer** (`nn.Conv2d`) with a kernel size and stride equal to the `patch_size` (e.g., 4x4) is applied to the image.
    2.  This effectively divides the image into a grid of patches and, for each patch, creates an embedding vector of dimension `embed_dim`.
-   **Output**: A tensor of shape `(B, num_patches, embed_dim)`, where `num_patches` is the total number of patches (e.g., `(H/patch_size) * (W/patch_size)`). This sequence of vectors is now ready for the Transformer.

### 2.2. Swin Transformer Backbone (`BasicLayer` and `SwinTransformerBlock`)

This is the core of the model where the most complex feature learning happens. It's composed of several "stages," and each stage consists of multiple `SwinTransformerBlock`s.

#### Key Concept: Windowed Self-Attention

A standard Vision Transformer calculates self-attention across all patches, which is computationally expensive for high-resolution images. The Swin Transformer's key innovation is to compute attention only within small, local **windows**.

-   **W-MSA (Windowed Multi-Head Self-Attention)**: In the first block of a stage, the patches are divided into a grid of windows (e.g., 8x8 patches per window). Self-attention is calculated independently within each window. This is efficient but limits the model's receptive field.

-   **SW-MSA (Shifted Window Multi-Head Self-Attention)**: To allow information to flow between windows, the next block uses a **shifted window** configuration. The window grid is shifted (e.g., by half a window size). This causes the new windows to capture different sets of patches, effectively creating connections between the original, un-shifted windows.

A `SwinTransformerBlock` contains:
1.  A W-MSA or SW-MSA module.
2.  A standard MLP (or Feed-Forward Network) consisting of two linear layers and a GELU activation.
3.  Layer Normalization and residual connections ("shortcuts"), which are crucial for stable training.

#### Hierarchical Structure and `PatchMerging`

The backbone creates a hierarchical representation of the image. After each stage, a `PatchMerging` layer is applied:

-   **Process**: It takes groups of neighboring patches (e.g., 2x2) and concatenates their features. A linear layer then reduces the feature dimension.
-   **Effect**: This reduces the number of patches (halving the spatial resolution) while increasing the embedding dimension (doubling the number of channels). This is analogous to a pooling layer in a CNN and allows the model to learn features at different scales.

The data flows through the stages like this:
-   **Stage 1**: Processes the initial patches.
-   `PatchMerging`: Reduces resolution, increases channels.
-   **Stage 2**: Processes the merged patches with more channels.
-   `PatchMerging`: Reduces resolution again, increases channels again.
-   ... and so on for all stages.

### 2.3. Regression Head (`SwinTransformer` class `forward` method)

After the final stage of the backbone, we have a set of highly informative feature vectors. The regression head's job is to convert these features into the final six parameter predictions.

1.  **Pooling**: An `AdaptiveAvgPool1d` layer is applied. It averages the features across all the patches, resulting in a single feature vector of dimension `num_features` for each image in the batch.
2.  **Flattening**: The vector is flattened to a shape of `(B, num_features)`.
3.  **Multi-Head Prediction**: Instead of using a single linear layer to predict all six parameters at once, the model uses a `nn.ModuleDict`. This creates **six independent prediction heads**, one for each parameter.
    -   Each head is a small MLP (`nn.Sequential`) that takes the `num_features` vector as input.
    -   It has a hidden layer, a GELU activation, a Dropout layer for regularization, and a final linear layer that outputs a single value.
4.  **Activation**: A `Sigmoid` activation is applied to the output of each head. This squashes the predicted value to be within the range **[0, 1]**. This is critical and assumes that your target values in the dataset have also been normalized to this range.
5.  **Concatenation**: The outputs from the six individual heads are concatenated into a final tensor of shape `(B, 6)`. The order of concatenation is fixed (`Rc`, `H30`, `incl`, `mDisk`, `psi`, `gamma`) to ensure the loss is calculated correctly against the ground truth labels.

---

## 3. Summary of Data Flow

1.  **Input**: Image `(B, 2, 256, 256)`
2.  **PatchEmbed**: Converted to `(B, 4096, 96)` (assuming 256x256 image, 4x4 patches, embed_dim 96)
3.  **Stage 1**: Processes `(B, 4096, 96)` -> `(B, 4096, 96)`
4.  **PatchMerging**: Downsamples to `(B, 1024, 192)`
5.  **Stage 2**: Processes `(B, 1024, 192)` -> `(B, 1024, 192)`
6.  **PatchMerging**: Downsamples to `(B, 256, 384)`
7.  **Stage 3**: Processes `(B, 256, 384)` -> `(B, 256, 384)`
8.  **PatchMerging**: Downsamples to `(B, 64, 768)`
9.  **Stage 4**: Processes `(B, 64, 768)` -> `(B, 64, 768)`
10. **Pooling & Flattening**: Averaged and flattened to `(B, 768)`
11. **Regression Heads**: The `(B, 768)` vector is fed into 6 separate MLPs.
12. **Output**: The 6 outputs are concatenated to produce the final tensor `(B, 6)`.

This hierarchical design, combined with the efficiency of windowed attention and the specificity of multiple regression heads, makes the Swin Transformer a powerful and well-suited model for this parameter estimation task.
