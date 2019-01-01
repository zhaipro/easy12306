# coding: utf-8
# 用于观看pca的效果
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
import pathlib

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib


params = [0.99, 0.95, 0.90, 0.80]


def transform(vectors):
    n, c = vectors.shape
    new_vectors = np.zeros((n, 1 + len(params), c), dtype=np.uint8)
    new_vectors[:, 0, :] = vectors
    for i, n_components in enumerate(params, 1):
        pca = joblib.load(f'pca.{n_components}.pkl')
        t = pca.transform(vectors)
        # 恢复原形
        t = pca.inverse_transform(t)
        # 拼到一起
        t[t > 255] = 255
        t[t < 0] = 0
        new_vectors[:, i, :] = t
    return new_vectors


def main():
    texts = np.load('data.npy')
    print(texts.shape)
    # 向量化
    n, h, w = texts.shape
    texts.shape = (n, -1)
    print(texts.shape)
    for i, n_components in enumerate(params, 1):
        pca = PCA(n_components=n_components, whiten=True)
        pca.fit(texts)
        print(n_components, pca.n_components_)
        joblib.dump(pca, f'pca.{n_components}.pkl')
    n = 800     # 只保存这里提出的前n张图就够了
    imgs = np.zeros((n, 1 + len(params), h * w), dtype=np.uint8)
    imgs = transform(texts[:n])
    imgs.shape = (n, -1, w)
    # 写到目录中
    path = 'pca'
    pathlib.Path(path).mkdir(exist_ok=True)
    for idx, img in enumerate(imgs):
        cv2.imwrite(f'{path}/{idx}.jpg', img)


def test(fn):
    # 对自己构建的图像做pca
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    _, w = img.shape
    img.shape = (1, -1)
    img = transform(img)
    img.shape = (-1, w)
    cv2.imwrite(fn, img)


if __name__ == '__main__':
    main()
    test('text.jpg')
