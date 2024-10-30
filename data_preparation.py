import numpy as np
import pandas as pd
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler
import random
import splitfolders
import SimpleITK as sitk

### display a sample 
_flair = 'BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_flair.nii.gz'
_t1 = 'BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_t1.nii.gz'
_t1c = 'BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz'
_t2 = 'BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_t2.nii.gz'
_label = 'BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_seg.nii.gz'
sample =  {"t1":_t1, "t2":_t2, "t1c":_t1c,"flair":_flair, "label":_label}
depth = random.randint(50,100)

i = 1
fig = plt.figure(figsize=(20, 20))
for j in sample:
    img = sitk.GetArrayFromImage(sitk.ReadImage(sample[j]))
    fig.add_subplot(1,5,i)
    plt.imshow(img[depth,:,:],interpolation='nearest', cmap='gray')
    plt.title("mod:{}\n {}".format(j,img.shape[1:]))
    plt.xticks([])
    plt.yticks([])
    i = i + 1

t1_list = sorted(glob.glob('BraTS2021_Training_Data/*/*t1.nii.gz'))
t2_list = sorted(glob.glob('BraTS2021_Training_Data/*/*t2.nii.gz'))
t2_list = sorted(glob.glob('BraTS2021_Training_Data/*/*t2.nii.gz'))
t1ce_list = sorted(glob.glob('BraTS2021_Training_Data/*/*t1ce.nii.gz'))
flair_list = sorted(glob.glob('BraTS2021_Training_Data/*/*flair.nii.gz'))
mask_list = sorted(glob.glob('BraTS2021_Training_Data/*/*seg.nii.gz'))

# Image Pre-processing Steps:
# 1. Scaling: Normalized the data to a range between 0 and 1.
# 2. Filtering: Removed data with less than 1% of their volume in unhealthy regions.
# 3. Slice Selection: Reduced the number of slices to 128 by selecting slices 13 through 141.
# 4. Resizing: Scaled each slice from 240x240 to 128x128 pixels.
# 5. Channel Combination: Merged the T2, T1c, and FLAIR series to create final images with dimensions 128x128x3.


scaler = MinMaxScaler()
slice_b = 13
slice_t = 141
num_data = len(t1_list)

## Save as a slice

for img in range(num_data):

    image_t2=nib.load(t2_list[img]).get_fdata()
    image_t2=scaler.fit_transform(image_t2.reshape(-1, image_t2.shape[-1])).reshape(image_t2.shape)
 
 
    image_t1ce= nib.load(t1ce_list[img]).get_fdata()
    image_t1ce= scaler.fit_transform(image_t1ce.reshape(-1, image_t1ce.shape[-1])).reshape(image_t1ce.shape)

    image_flair= nib.load(flair_list[img]).get_fdata()
    image_flair= scaler.fit_transform(image_flair.reshape(-1, image_flair.shape[-1])).reshape(image_flair.shape)
 
    mask= nib.load(mask_list[img]).get_fdata()
    mask= mask.astype(np.uint8)
    
    val, counts = np.unique(mask, return_counts=True)

   
    mask[mask==4] = 3

    
    if (1 - (counts[0]/counts.sum())) > 0.01:
    
        for s in range(slice_b,slice_t):


            temp_combined_images = np.stack([image_flair[:,:,s], image_t1ce[:,:,s], image_t2[:,:,s]], axis=2)
            temp_combined_images=temp_combined_images[56:184, 56:184]


            temp_mask = mask[:,:,s]
            temp_mask=temp_mask[56:184, 56:184] 
            temp_mask= to_categorical(temp_mask, num_classes=4)


            np.save('Sliced-Dataset-2D/images/image_'+str(img)+'_'+str(s)+'.npy', temp_combined_images)
            np.save('Sliced-Dataset-2D/masks/mask_'+str(img)+'_'+str(s)+'.npy', temp_mask)

### Divide the dataset into training and validation


input_folder = 'Sliced-Dataset-2D/'
output_folder = 'Sliced-Dataset-128/'
# Split with a ratio.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None) # default values

