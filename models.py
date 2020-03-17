import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import TimeDistributed, Dense, Flatten, Dropout, AveragePooling2D
from ROIPoolingLayer import RoiPoolingConv


def make_rpn(base_layers, num_anchors):
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                      kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = layers.Conv2D(num_anchors, (1, 1), padding="same", activation='sigmoid',
                            kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = layers.Conv2D(num_anchors * 4, (1, 1), activation='linear',
                           kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):

    pooling_regions = 14
    # densenet output channels are 1024..
    input_shape = (num_rois, 14, 14, 1024)

    # from vgg version..
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([
        base_layers, input_rois])

    out_roi_pool = TimeDistributed(AveragePooling2D(
        (7, 7)), name='avg_pool')(out_roi_pool)
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax',
                                      kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear',
                                     kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


input_shape_img = (None, None, 3)
img_input = layers.Input(shape=input_shape_img)
roi_input = layers.Input(shape=(C.num_rois, 4))
feature_map_input = layers.Input(shape=input_shape_features)

resnet = tf.keras.applications.ResNet50()
x = resnet(input)
