import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import cv2
import numpy as np
from ROIPoolingLayer import RoiPoolingConv


def make_RPN(anchors_num=9):
    feature_map_title = layers.Input(shape=(None, None, 2048))

    k = anchors_num
    x = layers.Conv2D(512, kernel_size=(3, 3))(feature_map_title)
    reg = layers.Conv2D(4*k, kernel_size=(1, 1), activation='sigmoid')(x)
    scores = layers.Conv2D(1*k, kernel_size=(1, 1))(x)

    model = Model(inputs=[feature_map_title], outputs=[scores, reg])

    # model.summary()
    return model


img = cv2.imread('test.jpg')
h, w, _ = img.shape

img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, 0)

model_rpn = make_RPN()
resnet = tf.keras.applications.ResNet50(include_top=False)
feature_map = resnet.predict(img)
print(feature_map.shape)
proposals = model_rpn(feature_map)
print(proposals.shape)


x = RoiPoolingConv(pool_size=7, num_rois=proposals.shape[1])(
    feature_map, proposals)
