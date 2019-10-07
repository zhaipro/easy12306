# coding: utf-8
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import layers
from keras import models
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical


def load_data(fn='texts.npz', to=False):
    data = np.load(fn)
    texts, labels = data['texts'], data['labels']
    texts = texts / 255.0
    _, h, w = texts.shape
    texts.shape = (-1, h, w, 1)
    if to:
        labels = to_categorical(labels)
    n = int(texts.shape[0] * 0.9)   # 90%用于训练，10%用于测试
    return (texts[:n], labels[:n]), (texts[n:], labels[n:])


def savefig(history, start=1, last=30):
    loss = history.history['loss'][start - 1:last]
    val_loss = history.history['val_loss'][start - 1:last]
    epochs = list(range(start, len(loss) + start))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.jpg')

    acc = history.history['acc'][start - 1:last]
    val_acc = history.history['val_acc'][start - 1:last]
    plt.clf()   # 清空图像
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('acc.jpg')


def build_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(None, None, 1)),
        layers.MaxPooling2D(),  # 19 -> 9
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),  # 9 -> 4
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),  # 4 -> 2
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu'),
        layers.Dense(80, activation='softmax'),
    ])
    return model


def main():
    (train_x, train_y), (test_x, test_y) = load_data()
    model = build_model()
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # 当标准评估停止提升时，降低学习速率
    reduce_lr = ReduceLROnPlateau(verbose=1)
    history = model.fit(train_x, train_y, epochs=100,
                        validation_data=(test_x, test_y),
                        callbacks=[reduce_lr])
    savefig(history, start=10)
    model.save('model.v1.0.h5', include_optimizer=False)


def load_data_v2():
    (train_x, train_y), (test_x, test_y) = load_data(to=True)
    # 这里是统计学数据
    (train_v2_x, train_v2_y), (test_v2_x, test_v2_y) = load_data('texts.v2.npz')
    # 合并
    train_x = np.concatenate((train_x, train_v2_x))
    train_y = np.concatenate((train_y, train_v2_y))
    test_x = np.concatenate((test_x, test_v2_x))
    test_y = np.concatenate((test_y, test_v2_y))
    return (train_x, train_y), (test_x, test_y)


def acc(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true + y_pred, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def main_v19():     # 1.9
    (train_x, train_y), (test_x, test_y) = load_data_v2()
    model = models.load_model('model.v1.0.h5')
    model.compile(optimizer='RMSprop',
                  loss='categorical_hinge',
                  metrics=[acc])
    reduce_lr = ReduceLROnPlateau(verbose=1)
    history = model.fit(train_x, train_y, epochs=100,
                        validation_data=(test_x, test_y),
                        callbacks=[reduce_lr])
    savefig(history)
    model.save('model.v1.9.h5', include_optimizer=False)


def main_v20():
    (train_x, train_y), (test_x, test_y) = load_data()
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(None, None, 1)),
        layers.MaxPooling2D(),  # 19 -> 9
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 9 -> 4
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 4 -> 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 2 -> 1
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu'),
        layers.Dense(80, activation='softmax'),
    ])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=10,
              validation_data=(test_x, test_y))
    (train_x, train_y), (test_x, test_y) = load_data_v2()
    model.compile(optimizer='rmsprop',
                  loss='categorical_hinge',
                  metrics=[acc])
    reduce_lr = ReduceLROnPlateau(verbose=1)
    history = model.fit(train_x, train_y, epochs=100,
                        validation_data=(test_x, test_y),
                        callbacks=[reduce_lr])
    savefig(history)
    # 保存，并扔掉优化器
    model.save('model.v2.0.h5', include_optimizer=False)


def predict(texts):
    model = models.load_model('model.h5')
    texts = texts / 255.0
    _, h, w = texts.shape
    texts.shape = (-1, h, w, 1)
    labels = model.predict(texts)
    return labels


def _predict():
    texts = np.load('data.npy')
    labels = predict(texts)
    np.save('labels.npy', labels)


def show():
    texts = np.load('data.npy')
    labels = np.load('labels.npy')
    labels = labels.argmax(axis=1)
    pathlib.Path('classify').mkdir(exist_ok=True)
    for idx, (text, label) in enumerate(zip(texts, labels)):
        # 使用聚类结果命名
        fn = f'classify/{label}.{idx}.jpg'
        cv2.imwrite(fn, text)


if __name__ == '__main__':
    main()
    # main_v2()
    _predict()
    show()
