# coding: utf-8
import pathlib

import cv2
import numpy as np


def load_data(fn='texts.npz', to=False):
    from keras.utils import to_categorical
    data = np.load(fn)
    texts, labels = data['texts'], data['labels']
    texts = texts / 255.0
    _, h, w = texts.shape
    texts.shape = (-1, h, w, 1)
    if to:
        labels = to_categorical(labels)
    n = int(texts.shape[0] * 0.9)   # 90%用于训练，10%用于测试
    return (texts[:n], labels[:n]), (texts[n:], labels[n:])


def savefig(history, fn='loss.jpg'):
    import matplotlib.pyplot as plt
    # 忽略起点
    loss = history.history['loss'][1:]
    val_loss = history.history['val_loss'][1:]
    epochs = list(range(2, len(loss) + 2))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fn)


def main():
    from keras import models
    from keras import layers
    (train_x, train_y), (test_x, test_y) = load_data()
    _, h, w, _ = train_x.shape
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, 1)),
        layers.MaxPooling2D(),  # 19 -> 17 -> 8
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),  # 8 -> 6 -> 3
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dense(80, activation='softmax'),
    ])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_x, train_y, epochs=30,
                        validation_data=(test_x, test_y))
    savefig(history)
    model.save('model.h5')


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
    import keras.backend as K
    return K.cast(K.equal(K.argmax(y_true + y_pred, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def main_v2():
    from keras import models
    from keras import layers
    from keras import optimizers
    (train_x, train_y), (test_x, test_y) = load_data()
    _, h, w, _ = train_x.shape
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(h, w, 1)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 19 -> 9
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 9 -> 4
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 4 -> 2
        layers.Flatten(),
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu'),
        layers.Dense(80, activation='softmax'),
    ])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=10,
              validation_data=(test_x, test_y))
    (train_x, train_y), (test_x, test_y) = load_data_v2()
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='categorical_hinge',
                  metrics=[acc])
    model.fit(train_x, train_y, epochs=20,
              validation_data=(test_x, test_y))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='categorical_hinge',
                  metrics=[acc])
    history = model.fit(train_x, train_y, epochs=40, batch_size=10 * 80,
                        validation_data=(test_x, test_y))
    savefig(history)
    # 保存前，先编译回正式版本
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.save('model.h5')


def predict(texts):
    from keras import models
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
