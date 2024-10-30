import os
import numpy as np

def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):
        if (image_name.split('.')[1] == 'npy'):

            image = np.load(img_dir+image_name)

            images.append(image)
    images = np.array(images)

    return(images)
    

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size

# test the data generator

train_img_dir = "Sliced-Dataset-128/train/images/"
train_mask_dir = "Sliced-Dataset-128/train/masks/"
train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 64

train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)

img, msk = train_img_datagen.__next__()

test_img=img[2]
test_mask=msk[2]
test_mask=np.argmax(test_mask, axis=2)



fig = plt.figure(figsize=(20, 20))
fig.add_subplot(1,4,1)
plt.imshow(test_img[:,:, 0],interpolation='nearest', cmap='gray')
plt.title('Image flair')
fig.add_subplot(1,4,2)
plt.imshow(test_img[:,:, 1],interpolation='nearest', cmap='gray')
plt.title('Image t1ce')
fig.add_subplot(1,4,3)
plt.imshow(test_img[:,:,2],interpolation='nearest', cmap='gray')
plt.title('Image t2')
fig.add_subplot(1,4,4)
plt.imshow(test_mask, interpolation='nearest', cmap='gray')
plt.title('Mask')
plt.show()

