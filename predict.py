from sklearn.model_selection import train_test_split
import os
import glob
import cv2
import numpy as np # linear algebra
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util import montage
import gc; gc.enable() # memory is tight
from skimage.morphology import label
from utils import *
import random
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from PIL import Image

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
data_dir = 'data'
train_image_dir = os.path.join(data_dir, 'train')
test_image_dir = os.path.join(data_dir, 'test')

BATCH_SIZE = 4
EDGE_CROP = 2
NB_EPOCHS = 25
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'DECONV'
# downsampling inside the network
NET_SCALING = None
# downsampling in preprocessing
IMG_SCALING = (1, 1)
# number of validation images to use
VALID_IMG_COUNT = 20
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 200
AUGMENT_BRIGHTNESS = False


def get_all_imgs():
    img_path = os.path.join(test_image_dir)
    images = glob.glob(os.path.join(img_path,'*.*'))
    return [os.path.basename(image) for image in images]

TEST_IMGS = get_all_imgs()

def make_image_gen(img_file_list=TRAIN_IMGS, batch_size = BATCH_SIZE):
    all_batches = img_file_list
    out_rgb = []
    out_mask = []
    img_path = os.path.join(test_image_dir)
    while True:
        np.random.shuffle(all_batches)
        for c_img_id in all_batches:
            c_img = imread(os.path.join(img_path,c_img_id))
            c_img = cv2_brightness_augment(c_img)
            if IMG_SCALING is not None:
                c_img = cv2.resize(c_img,(256,256),interpolation = cv2.INTER_AREA)
                
            out_rgb += [c_img]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0
                out_rgb=[]





#MAKE VALIDATION SET
valid_x = next(make_image_gen(TEST_IMGS,len(TEST_IMGS)))
print(valid_x.shape)



def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)

if UPSAMPLE_MODE=='DECONV':
    upsample=upsample_conv
else:
    upsample=upsample_simple
    
input_img = layers.Input(valid_x.shape[1:], name = 'RGB_Input')
pp_in_layer = input_img
print("Shape of pp_in_layer is",pp_in_layer.shape)
if NET_SCALING is not None:
    pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)
    
pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
pp_in_layer = layers.BatchNormalization()(pp_in_layer)

c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (pp_in_layer)
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c1)
p1 = layers.MaxPooling2D((2, 2)) (c1)

c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p1)
c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c2)
p2 = layers.MaxPooling2D((2, 2)) (c2)

c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p2)
c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c3)
p3 = layers.MaxPooling2D((2, 2)) (c3)

c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p3)
c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)


c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (p4)
c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (c5)

u6 = upsample(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (u6)
c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c6)

u7 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u7)
c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c7)

u8 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u8)
c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c8)

u9 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u9)
c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c9)

d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
if NET_SCALING is not None:
    d = layers.UpSampling2D(NET_SCALING)(d)

seg_model = models.Model(inputs=[input_img], outputs=[d])
seg_model.summary()


def dice_coef(y_true, y_pred, smooth=1):
    print(type(y_pred))
    print("the type of y true is ",type(y_true))
    print("the shape of y_true is",y_true.shape)
    print("the shape of y_pred is",y_pred.shape)
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    print("the shape of in_gt is",in_gt.shape)
    print("the shape of in_pred is",in_pred.shape)
    
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    print("the shape of y_true in true_positive_rate is",y_true.shape)
    print("the shape of y_pred in true_positive_rate is",y_pred.shape)
    
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])


weight_path="{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)

seg_model.load_weights(weight_path)






in_files=[]
folder_path="data/test/"
for filename in os.listdir("data/test/"):
    in_files.append(folder_path+filename)
    print(folder_path+filename)
    TEST_IMGS=[filename]
    valid_x = next(make_image_gen(TEST_IMGS,len(TEST_IMGS)))
    pred_y = seg_model.predict(valid_x)
    print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())
    output_dir="test_predictions/"
    data=pred_y
    data=data*(255.0)
    for i in range(data.shape[0]):
        # Create a PIL image object from the grayscale data
        img = Image.fromarray(data[i, :, :, 0].astype(np.uint8), mode='L')

        # Define the filename for the JPEG file
        filename2 = folder_path+filename[:-3]+"OUT"+".jpg"
        filename2="test_predictions/"+filename[:-3]+"OUT"+".jpg"
        # Save the image to the specified filename
        img.save(filename2)

        # Print the filename to confirm that the image was saved
        print(f'Saved {filename2}')


"""
pred_y = seg_model.predict(valid_x)
print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())
output_dir="test_predictions/"
data=pred_y
data=data*(255.0)
for i in range(data.shape[0]):
    # Create a PIL image object from the grayscale data
    img = Image.fromarray(data[i, :, :, 0].astype(np.uint8), mode='L')

    # Define the filename for the JPEG file
    filename = f'image_{i}.jpg'

    # Save the image to the specified filename
    img.save(output_dir+filename)

    # Print the filename to confirm that the image was saved
    print(f'Saved {filename}')
"""