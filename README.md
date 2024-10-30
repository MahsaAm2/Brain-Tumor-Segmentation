# Tumor Segmentation with 2D U-Net on BraTS 2021 Dataset

This repository contains the implementation of a 2D U-Net model for tumor segmentation using the BraTS 2021 dataset.

## Project Overview
This project focuses on segmenting brain tumors using MRI images from the BraTS 2021 dataset. The pipeline involves three main stages:

1. **Pre-processing**: Includes data normalization, resizing, and skull-stripping.
2. **Model Implementation**: Development and training of a 2D U-Net model optimized for MRI images.
3. **Evaluation**: Assessment of model performance using metrics such as Dice Score and IoU.

## Dataset
The BraTS 2021 dataset provides four MRI modalities (T1, T2, FLAIR, T1c) and a skull-stripped mask, with each modality containing 155 slices at 240x240 resolution. The labels in the dataset are:
- **Label 0**: Unlabeled data
- **Label 1**: Necrotic and non-enhancing tumor
- **Label 2**: Peritumoral edema
- **Label 3**: GD-enhancing tumor

## Processing

1. **Scaling**: Normalized the data to a range between 0 and 1.
2. **Filtering**: Removed data with less than 1% of their volume in unhealthy regions.
3. **Slice Selection**: Reduced the number of slices to 128 by selecting slices 13 through 141.
4. **Resizing**: Scaled each slice from 240x240 to 128x128 pixels.
5. **Channel Combination**: Merged the T2, T1c, and FLAIR series, creating final images with dimensions 128x128x3.
   
![image](https://github.com/user-attachments/assets/3f8e9aa5-732c-4353-b1f8-d56ffdd939ed)

## Data Split
- **Training Dataset**: 75%
- **Validation Dataset**: 25%

## Model Architecture
The model follows a U-Net structure with two main paths:

### Contraction Path
The contraction path includes five blocks, with the final block omitting Max Pooling. Each block consists of:
- Conv2D
- Dropout
- Conv2D
- Max Pooling (except the last block)

### Expansive Path
The expansive path contains four blocks, structured as follows:
- Conv2DTranspose
- Concatenate
- Conv2D
- Dropout
- Conv2D

## Loss Function
The error function is a combination of the following two loss functions:
1. **Dice Loss**
2. **Focal Loss**

## Evaluation Metrics
The evaluation criteria include:
1. **Accuracy**
2. **IoU Score**

Below are the training results for model performance:

### Accuracy
![Accuracy](https://github.com/user-attachments/assets/fe4dcc42-91d5-4cac-91c7-7e25a7e014cd)


### IoU Score
![IoU Score](https://github.com/user-attachments/assets/ce1693b0-12c7-435d-b99f-58457195b272)


## Segmentation Results

Below are sample segmentation results from the model:

### Example 1
![example](https://github.com/user-attachments/assets/da97eb5a-29dd-4072-b6ed-836c37ebb0d7)

### Example 2
![example2](https://github.com/user-attachments/assets/e02844c4-0783-4dfa-959f-ae725b1f2801)

### Example 3
![example3](https://github.com/user-attachments/assets/4c60e963-ec08-4432-94ef-0503508b6e0e)

### Example 4
![example4](https://github.com/user-attachments/assets/801e37a2-8da4-4bbc-a356-d658f2b61f6b)








