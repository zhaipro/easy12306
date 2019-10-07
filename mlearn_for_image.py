# coding: utf-8
import sys

import cv2
import numpy as np
from keras import backend as K
from keras import Input
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


def preprocess_input(x):
    x = x.astype('float32')
    # 我是用cv2来读取的图片，其已经是BGR格式了
    mean = [103.939, 116.779, 123.68]
    x -= mean
    return x


def load_data():
    # 这是统计学专家提供的训练集
    data = np.load('captcha.npz')
    train_x, train_y = data['images'], data['labels']
    train_x = preprocess_input(train_x)
    # 由于是统计得来的信息，所以在此给定可信度
    sample_weight = train_y.max(axis=1) / np.sqrt(train_y.sum(axis=1))
    sample_weight /= sample_weight.mean()
    train_y = train_y.argmax(axis=1)

    # 这是人工提供的验证集
    data = np.load('captcha.test.npz')
    test_x, test_y = data['images'], data['labels']
    test_x = preprocess_input(test_x)
    return (train_x, train_y, sample_weight), (test_x, test_y)


def build_model():
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


def learn():
    (train_x, train_y, sample_weight), (test_x, test_y) = load_data()
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 vertical_flip=True)
    train_generator = datagen.flow(train_x, train_y, sample_weight=sample_weight)
    model = build_model()
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    reduce_lr = ReduceLROnPlateau(verbose=1)
    model.fit_generator(train_generator, epochs=400,
                        steps_per_epoch=100,
                        validation_data=(test_x[:800], test_y[:800]),
                        callbacks=[reduce_lr])
    result = model.evaluate(test_x, test_y)
    print(result)
    model.save('12306.image.model.h5', include_optimizer=False)


def loss(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, 8, 80))
    y_pred = K.max(y_pred, axis=1)
    return K.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False, axis=-1)


class ReshapeLayer(layers.Layer):

    def call(self, inputs):
        images = inputs
        images = K.reshape(images, (-1, 67, 67, 3))
        return images

    def compute_output_shape(self, input_shape):
        return None, 67, 67, 3


def learn_v2():
    data = np.load('dataset.npz')
    images, texts = data['images'], data['texts']
    images = preprocess_input(images)
    texts = texts.astype('float32')
    texts /= 255.0
    n, h, w = texts.shape
    texts.shape = -1, h, w, 1
    texts = models.load_model('model.v1.h5').predict(texts)
    texts = texts.argmax(axis=-1)
    n = int(n * 0.9)    # 90%用于训练，10%用于测试
    (train_x, train_y), (test_x, test_y) = (images[:n], texts[:n]), (images[n:], texts[n:])

    input = Input(shape=(None, None, None, 3), dtype='float32')
    immodel = build_model()
    output = immodel(ReshapeLayer()(input))
    model = models.Model(input, output)
    model.compile(optimizer='rmsprop',
                  loss=loss)
    model.summary()
    model.fit(train_x, train_y,
              validation_data=(test_x, test_y),
              epochs=10)
    immodel.save('immodel.h5', include_optimizer=False)


def predict(imgs):
    imgs = preprocess_input(imgs)
    model = models.load_model('12306.image.model.h5')
    labels = model.predict(imgs)
    return labels


def _predict(fn):
    imgs = cv2.imread(fn)
    imgs = cv2.resize(imgs, (67, 67))
    imgs.shape = (-1, 67, 67, 3)
    labels = predict(imgs)
    print(labels.max(axis=1))
    print(labels.argmax(axis=1))


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        _predict(sys.argv[1])
    else:
        # learn()
        learn_v2()
