# coding: utf-8
import pathlib

import cv2
import numpy as np


def load_data():
    data = np.load('texts.npz')
    texts, labels = data['texts'], data['labels']
    n = int(texts.shape[0] * 0.9)   # 90%用于训练，10%用于测试
    return (texts[:n], labels[:n]), (texts[n:], labels[n:])


def main():
    import matplotlib.pyplot as plt
    from keras import models
    from keras import layers
    (train_x, train_y), (test_x, test_y) = load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    _, h, w = train_x.shape
    train_x.shape = (-1, h, w, 1)
    test_x.shape = (-1, h, w, 1)
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
    loss = history.history['loss'][1:]
    val_loss = history.history['val_loss'][1:]
    epochs = list(range(2, len(loss) + 2))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.jpg')
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
    _predict()
    show()
