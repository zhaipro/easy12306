#! env python
# coding: utf-8
# 功能：对文字部分使用k-means算法进行聚类
import os
import time
import sys

import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.externals import joblib


def get_img_as_vector(fn):
    im = cv2.imread(fn)
    im = im[:, :, 0]
    retval, dst = cv2.threshold(im, 128, 1, cv2.THRESH_BINARY_INV)
    return dst.reshape(dst.size)


def main():
    # 读取训练用数据
    print('Start: read data', time.process_time())
    fns = os.listdir('ocr')
    X = [get_img_as_vector(os.path.join('ocr', fn)) for fn in fns]
    print('Samples', len(X), 'Feature', len(X[0]))
    # PCA
    print('Start: PCA', time.process_time())
    pca = PCA(n_components=0.99)
    pca.fit(X)
    X = pca.transform(X)
    print('Samples', len(X), 'Feature', len(X[0]))
    sys.stdout.flush()
    # 训练
    print('Start: train', time.process_time())
    n_clusters = 2000    # 聚类中心个数
    estimator = KMeans(n_clusters, n_init=1, max_iter=20, verbose=True)
    estimator.fit(X)
    print('Clusters', estimator.n_clusters, 'Iter', estimator.n_iter_)
    print('Start: classify', time.process_time())
    fp = open('result11.txt', 'w')
    for fn, c in zip(fns, estimator.labels_):
        print(fn, c, file=fp)
    fp.close()
    print('Start: save model', time.process_time())
    joblib.dump(estimator, 'k-means11.pkl')

if __name__ == '__main__':
    main()
