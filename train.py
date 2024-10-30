import os
import numpy as np
import pandas as pd
from data_generator import imageLoader
import keras
from matplotlib import pyplot as plt
import glob
import random
import segmentation_models_3D as sm
from  2d_unet import simple_unet_model
from keras.models import load_model


train_img_dir = "Sliced-Dataset-128/train/images/"
train_mask_dir = "Sliced-Dataset-128/train/masks/"

val_img_dir = "Sliced-Dataset-128/val/images/"
val_mask_dir = "Sliced-Dataset-128/val/masks/"


train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list=os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

#Define loss, metrics and optimizer to be used for training

wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]
batch_size = 64
LR = 0.0001
optim = keras.optimizers.Adam(LR)

################################################

train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size


model = simple_unet_model(IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_CHANNELS=3,
                          num_classes=4)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)

###################################################
# fit model

history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=20,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )

model.save('brats_2d_v1.hdf5')

######################################
# test model
my_model = load_model('brats_2d_v1.hdf5', compile=False)

batch_size=64 
test_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)


test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

def plot_result(img_num, slice_num):
    test_img = np.load("Sliced-Dataset-128/val/images/image_"+str(img_num)+'_'+str(slice_num)+".npy")

    test_mask = np.load("Sliced-Dataset-128/val/masks/mask_"+str(img_num)+'_'+str(slice_num)+".npy")
    test_mask_argmax=np.argmax(test_mask, axis=2)

    test_img_input = np.expand_dims(test_img, axis=0)
    test_prediction = my_model.predict(test_img_input)
    test_prediction_argmax=np.argmax(test_prediction, axis=3)[0,:,:]
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,1], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(test_mask_argmax[:,:])
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction_argmax[:,:])
    plt.show()



