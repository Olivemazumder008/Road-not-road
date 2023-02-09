# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:13:43 2022

@author: sas11
"""
import tensorflow as tf # Imports tensorflow
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,Embedding
from tensorflow.keras.layers import Conv2D, MaxPooling2D,LSTM,Bidirectional,Attention,concatenate,MultiHeadAttention
from tensorflow.keras.layers import DepthwiseConv2D,Add, ReLU, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Activation,ActivityRegularization, AvgPool2D, LeakyReLU, Conv2DTranspose
from tensorflow.keras import regularizers, optimizers,losses
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.metrics import Recall,Precision,AUC,TruePositives,TrueNegatives,FalseNegatives,FalsePositives, SpecificityAtSensitivity,SensitivityAtSpecificity
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
import pandas as pd 
import matplotlib
import seaborn as sns
import sklearn
#import imblearn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from glob import glob
from tensorflow_addons.metrics import F1Score
from keras import backend as K
import skimage.io
import skimage.color
import skimage.filters

img_dir='data/road'
mask_dir='data/mask'
Name= 'road0_'

rel_dirname = os.path.dirname(__file__)

img_shape = (128,512, 3)
road_images=[]
mask_images=[]
mimg=[]
for filename in glob(os.path.join(rel_dirname, img_dir+'/*.png')):
     img = image.load_img(os.path.join(rel_dirname, filename),target_size=img_shape)
     img = image.img_to_array(img)
     img = img/255.0
     road_images.append(img)
     
for filename in glob(os.path.join(rel_dirname, mask_dir+'/*.png')):
     img = image.load_img(os.path.join(rel_dirname, filename),target_size=img_shape)
     img = image.img_to_array(img)
     img = img/255.0
     img[img[:,:,2]==1]=[0,0,0]
     img[img[:,:,0]==0]=[1,1,1]
     img[img[:,:,2]==0]=[0,0,0]
     mask_images.append(img)
plt.imshow(road_images[0])
plt.show()
plt.imshow(mask_images[0])
plt.show()
plt.imshow(road_images[0]*mask_images[0])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(road_images, mask_images, random_state=7, test_size=0.25)

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_vgg19_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg19.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg19.get_layer("block3_conv4").output         ## (128 x 128)
    s4 = vgg19.get_layer("block4_conv4").output         ## (64 x 64)

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output         ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(3, 1, activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',Recall(),Precision(),AUC()])
    plot_model(model, to_file=Name+'seg.png',show_shapes= True , show_layer_names=True)
    model.summary()
    return model

unet_model=build_vgg19_unet(img_shape)
earlystopper = EarlyStopping(patience=100, verbose=1)
checkpointer = ModelCheckpoint('model_unet_checkpoint.h5', verbose=1, save_best_only=True)

results = unet_model.fit(X_train,  y_train, batch_size=4, epochs=1, 
                        validation_data=(X_test, y_test),callbacks=[earlystopper, checkpointer])

unet_model.save(Name+'.h5')
pd.DataFrame.from_dict(results.history).to_csv(Name+'.csv',index=False)

img_id=3
new_img= np.array(road_images[img_id])
plt.imshow(road_images[img_id])
plt.show()
new_img = new_img.reshape(1,256,1024,3)
new_impr= unet_model.predict(new_img)
plt.imshow(new_impr[0])
plt.show()