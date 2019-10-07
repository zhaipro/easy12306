# coding: utf-8
import cv2
import numpy as np
from keras import backend as K
from keras import layers
from keras import models
from keras.losses import categorical_hinge
from keras import Input
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.applications import VGG16


def load_data(fn='dataset.npz'):
    data = np.load(fn)
    texts, images = data['texts'], data['images']

    _, h, w = texts.shape
    texts = texts.astype('float32')
    texts /= 255.0
    texts.shape = (-1, h, w, 1)

    images = images.astype('float32')
    # 我是用cv2来读取的图片，其已经是BGR格式了
    mean = [103.939, 116.779, 123.68]
    images -= mean

    n = int(texts.shape[0] * 0.9)   # 90%用于训练，10%用于测试
    return (texts[:n], images[:n]), (texts[n:], images[n:])


def acc(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true + y_pred, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def loss(y_true, y_pred):
    return y_pred


def build_text_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(None, None, 1)),
        layers.MaxPooling2D(),  # 19 -> 9
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 9 -> 4
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 4 -> 2
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu'),
        layers.Dense(80, activation='softmax'),
    ], name='adsfa')
    return model


def build_image_model():
    base = VGG16(weights='imagenet', include_top=False, input_shape=(None, None, 3))
    for layer in base.layers[:-4]:
        layer.trainable = False
    model = models.Sequential([
        base,
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(80, activation='softmax')
    ])
    return model


class ReshapeLayer(layers.Layer):

    def call(self, inputs):
        images = inputs
        images = K.reshape(images, (-1, 67, 67, 3))
        return images

    def compute_output_shape(self, input_shape):
        return None, 67, 67, 3


class LossLayer(layers.Layer):

    def build(self, input_shape):
        # 为该层创建一个不可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(80,),
                                      initializer='zeros',
                                      trainable=False)
        # 一定要在最后调用它
        super().build(input_shape)

    def call(self, inputs):
        y_pred, y_true = inputs
        y_true = K.reshape(y_true, (-1, 8, 80))
        y_true = K.max(y_true, axis=1)
        self.add_update(
            K.update(self.kernel, 0.9 * self.kernel + 0.1 * K.mean(y_pred, axis=0)),
            inputs)
        return categorical_hinge(y_pred, y_true) + K.mean(self.kernel * y_pred, axis=-1)

    def compute_output_shape(self, input_shape):
        return None,


def main():
    (train_texts, train_images), (test_x, test_y) = load_data()
    imodel = models.load_model('immodel.h5')
    tmodel = build_text_model()
    # imodel = build_image_model()
    text_input = Input(shape=(None, None, 1), dtype='float32')
    images_input = Input(shape=(None, None, None, 3), dtype='float32')
    text_output = tmodel(text_input)
    images_output = imodel(ReshapeLayer()(images_input))
    loss_output = LossLayer()([text_output, images_output])
    model = models.Model([text_input, images_input], loss_output)
    model.compile(optimizer='rmsprop',
                  loss=loss)
    model.summary()
    model.fit([train_texts, train_images], np.zeros(train_texts.shape[0]),
              validation_data=([test_x, test_y], np.zeros(test_x.shape[0])),
              epochs=10)
    tmodel.save('tmodel.h5', include_optimizer=False)
    imodel.save('imodel.h5', include_optimizer=False)


if __name__ == '__main__':
    main()
