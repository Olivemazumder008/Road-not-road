# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 20:49:30 2022

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


Name='Unet1'

img_dir='road'
#mask_dir='seg/tum'

rel_dirname = os.path.dirname(__file__)

img_shape = (272, 480, 3)
images=[]
mask_img=[]
mimg=[]
for filename in glob(os.path.join(rel_dirname, img_dir+'/*.png')):
     img = image.load_img(os.path.join(rel_dirname, filename),target_size=img_shape)
     img = image.img_to_array(img)
     img = img/255.0
     images.append(img)
     



raw_img= np.array(images)
mask_img = np.array(mask_img)

plt.imshow(raw_img[3])
plt.show()
#plt.imshow(mask_img[3])
#plt.show()



model= keras.models.load_model('pspunet_weight.h5')
"""
earlystopper = EarlyStopping(patience=100, verbose=1)
checkpointer = ModelCheckpoint('model_unet_checkpoint.h5', verbose=1, save_best_only=True)
results = model.fit(mask_img, raw_img, validation_split=0.2, batch_size=16, epochs=300, 
                    callbacks=[earlystopper, checkpointer])
model.save(Name+'.h5')
pd.DataFrame.from_dict(results.history).to_csv(Name+'.csv',index=False)
"""
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]   

img_id=4
new_img= np.array(raw_img[img_id])
plt.imshow(raw_img[img_id])
plt.show()

new_img = new_img.reshape(1,272, 480, 3)
pre= model.predict(new_img)
pre = create_mask(pre).numpy()
new_img[0][(pre==1).all(axis=2)] += [0, 0, 0] #""bike_lane_normal", "sidewalk_asphalt", "sidewalk_urethane""
new_img[0][(pre==2).all(axis=2)] += [0.5, 0.5,0] # "caution_zone_stairs", "caution_zone_manhole", "caution_zone_tree_zone", "caution_zone_grating", "caution_zone_repair_zone"]
new_img[0][(pre==3).all(axis=2)] += [0.2, 0.7, 0.5] #"alley_crosswalk","roadway_crosswalk"
new_img[0][(pre==4).all(axis=2)] += [0, 0.5, 0.5] #"braille_guide_blocks_normal", "braille_guide_blocks_damaged"
new_img[0][(pre==5).all(axis=2)] += [0, 0, 0.5] #"roadway_normal","alley_normal","alley_speed_bump", "alley_damaged""
new_img[0][(pre==6).all(axis=2)] += [0.5, 0, 0] #"sidewalk_blocks","sidewalk_cement" , "sidewalk_soil_stone", "sidewalk_damaged","sidewalk_other"
plt.imshow(new_img[0])
plt.show()