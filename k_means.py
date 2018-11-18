#! env python
# coding: utf-8
# 功能：对文字部分使用k-means算法进行聚类
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.externals import joblib

import pretreatment


def main():
    # 读取训练用数据
    print('Start: read data', time.process_time())
    X = pretreatment.load_data()    # NOQA
    c, h, w = X.shape
    X = X.reshape((c, h * w))       # NOQA
    X = X / 255.0                   # NOQA
    print('Samples', len(X), 'Feature', len(X[0]))
    # PCA
    print('Start: PCA', time.process_time())
    pca = PCA(n_components=0.99)
    pca.fit(X)
    X = pca.transform(X)            # NOQA
    print('Samples', len(X), 'Feature', len(X[0]))
    sys.stdout.flush()
    # 训练
    print('Start: train', time.process_time())
    n_clusters = 2000    # 聚类中心个数
    estimator = KMeans(n_clusters, n_init=1, max_iter=20, verbose=True)
    estimator.fit(X)
    print('Clusters', estimator.n_clusters, 'Iter', estimator.n_iter_)
    print('Start: classify', time.process_time())
    np.save('labels.npy', estimator.labels_)
    print('Start: save model', time.process_time())
    joblib.dump(estimator, 'k-means.pkl')


if __name__ == '__main__':
    main()
