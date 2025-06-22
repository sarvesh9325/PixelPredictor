# Pixel Predictor

## Table of Contents
1. [Introduction](#introduction)
2. [Assumptions](#assumptions)
3. [Network Model](#network-model)
4. [Use of Convolutional Autoencoders](#use-of-convolutional-autoencoders)
5. [Attributes and Parameters](#attributes-and-parameters)
6. [Simulation Algorithm](#simulation-algorithm)
7. [Performance Analysis](#performance-analysis)
8. [Simulation Analysis](#simulation-analysis)

---

## Introduction <a name="introduction"></a>
This project, **Pixel Predictor**, is a deep learning-based approach to predict and restore missing or corrupted pixels in RGB images. The model is built using a convolutional autoencoder architecture, which effectively extracts features, processes them, and recontructs the image. 
A convolutional autoencoders is a type of neural network that uses convolutional layers to learn efficient image representations and recontruct input data. The project is implemented using PyTorch and is designed to handle RGB images with missing or corrupted pixel data.

---

## Assumptions <a name="assumptions"></a>
- The input images are RGB images with three channels.
- The missing or corrupted pixels are represented as zero values in the image tensor.
- The model assumes that the missing pixels are randomly distributed across the image.
- The patch size for training and prediction is fixed and provided as an input parameter.

---

## Network Model <a name="network-model"></a>
The network model consists of three main components:
1. **Head**: A **Convolutional Neural Network (CNN)** that extracts features from the input image.
2. **Neck**: A **Multi-Layer Perceptron (MLP)** that processes the extracted features.
3. **Bottom**: A **Transposed Convolutional Network** that reconstructs the image from the processed features.

The model is trained using a **Mean Squared Error (MSE)** loss function and optimized using the **Adam optimizer**.

---

## Use of Convolutional Autoencoders <a name="use-of-convolutional-autoencoders"></a>

### **Convolutional Neural Network (CNN)**
- **Role**: The CNN is used in the **Head** of the model to extract spatial features from the input image. It captures local patterns, edges, and textures, which are essential for understanding the structure of the image.
- **Architecture**:
  - **Layer 1**: A convolutional layer with 32 filters, kernel size 3x3, stride 1, and padding 1, followed by a ReLU activation.
  - **Layer 2**: A convolutional layer with 64 filters, kernel size 3x3, stride 1, and padding 1, followed by a ReLU activation and max pooling.
  - **Layer 3**: A convolutional layer with 128 filters, kernel size 3x3, stride 1, and padding 1, followed by a ReLU activation and adaptive average pooling.
- **Output**: The CNN outputs a feature map that is passed to the MLP for further processing.

### **Multi-Layer Perceptron (MLP)**
- **Role**: The MLP is used in the **Neck** of the model to process the high-dimensional feature maps extracted by the CNN. It learns non-linear relationships between the features and prepares them for image reconstruction.
- **Architecture**:
  - **Layer 1**: A fully connected layer with 128 units, followed by a ReLU activation and dropout (0.4).
  - **Layer 2**: A fully connected layer with 256 units, followed by a ReLU activation and dropout (0.4).
  - **Layer 3**: A fully connected layer with 512 units, followed by a ReLU activation and dropout (0.4).
  - **Layer 4**: A fully connected layer that maps the features back to the original dimensionality of the feature map.
- **Output**: The MLP outputs a processed feature map that is passed to the transposed convolutional network for image reconstruction.

---

## Attributes and Parameters <a name="attributes-and-parameters"></a>
- **Input Image**: The image with missing or corrupted pixels.
- **Patch Size**: The size of the image patches used for training and prediction.
- **Output Image**: The reconstructed image with predicted pixels.
- **Mask**: A binary mask indicating the locations of missing or corrupted pixels.
- **Learning Rate**: The learning rate for the Adam optimizer.
- **Epochs**: The number of training epochs.

---

## Simulation Algorithm <a name="simulation-algorithm"></a>
1. **Masking**: Randomly mask a percentage of pixels in the input image to simulate missing or corrupted data.
2. **Scaling**: Normalize the input image to a range suitable for training.
3. **Training**: Train the model on the masked image to predict the missing pixels.
4. **Prediction**: Use the trained model to predict the missing pixels in the input image.
5. **Reconstruction**: Rescale the predicted image to its original range and combine it with the original image to produce the final output.

---

## Performance Analysis <a name="performance-analysis"></a>
- **Loss Function**: The model is evaluated using the **Mean Squared Error (MSE)** loss function.
- **Training Time**: The time taken to train the model on a given dataset.
- **Reconstruction Quality**: The quality of the reconstructed image is assessed visually and using quantitative metrics such as **PSNR (Peak Signal-to-Noise Ratio)**.
  
---

## Simulation Analysis <a name="simulation-analysis"></a>
- **Effect of Patch Size**: Analyze how different patch sizes affect the model's performance.
- **Effect of Masking Percentage**: Study the impact of varying the percentage of masked pixels on the reconstruction quality.
- **Effect of Learning Rate**: Investigate how different learning rates influence the training process and final model performance.

---
