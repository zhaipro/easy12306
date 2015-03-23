# -*- coding: cp936 -*-
# 功能：对文字部分使用k-means算法进行聚类
import cv2
import numpy as np
import os


class Knn:
    def __init__(self, k):
        self.k = k
        self.cs = np.random.random((k, 1083))
        # 一下变量用于计算新的中心
        self._cs = np.zeros((k, 1083))
        self._count = np.zeros(k)

    def train(self, X):
        for t in xrange(50):
            print 'iter', t
            for x in X:
                i = np.argmin([np.linalg.norm(x-c) for c in self.cs])
                self._cs[i] += x
                self._count[i] += 1
            for i in xrange(self.k):
                if self._count[i] != 0:
                    self._cs[i] /= self._count[i]

            if sum((np.linalg.norm(x) for x in self._cs - self.cs)) < 0.05:
                break

            self.cs = self._cs
            self._cs = np.zeros((self.k, 1083))
            self._count = np.zeros(self.k)

    def classify(self, x):
        return np.argmin([np.linalg.norm(x-c) for c in self.cs])


def get_img_as_vector(fn):
    im = cv2.imread(fn)
    im = im[:, :, 0]
    (retval, dst) = cv2.threshold(im, 128, 1, cv2.THRESH_BINARY_INV)
    return dst.reshape(dst.size)

k = Knn(1000)    # 分1000个类别
fns = os.listdir('ocr')
train_fns = fns[0: 10000]
test_fns = fns[10000: ]
# 读取训练用数据
print 'Start: read data'
X = [get_img_as_vector(os.path.join('ocr', fn)) for fn in train_fns]
# 训练
print 'Start: train'
k.train(X)
print 'Start: classify'
for fn in test_fns:
    x = get_img_as_vector(os.path.join('ocr', fn))
    print fn, k.classify(x)
