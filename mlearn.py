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
    from keras import models
    from keras import layers
    (train_x, train_y), (test_x, test_y) = load_data()
    _, train_x = cv2.threshold(train_x, 220, 1, cv2.THRESH_BINARY)
    _, test_x = cv2.threshold(test_x, 220, 1, cv2.THRESH_BINARY)
    model = models.Sequential([
        layers.Flatten(),
        layers.Dense(500, activation='relu'),
        layers.Dense(80, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=50)
    print(model.evaluate(test_x, test_y))
    model.save('model.h5')


def predict():
    from keras import models
    model = models.load_model('model.h5')
    texts = np.load('data.npy')
    _, texts = cv2.threshold(texts, 220, 1, cv2.THRESH_BINARY)
    labels = model.predict(texts)
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
    predict()
    show()
