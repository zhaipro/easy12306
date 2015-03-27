# -*- coding: cp936 -*-
# 功能：对文字部分使用k-means算法进行聚类
import cv2
import numpy as np
import os
import time


class KMeans:
    def __init__(self, k):
        self.k = k
        self.cs = np.zeros((k, 1083))
        self.count = np.zeros(k)     # 距离相应中心最近的点的个数

    def train(self, X):
        last_cs = X[0: self.k]  # or np.random.random((k, 1083))

        for t in xrange(50):
            print 'iter', t, 'time', time.clock()
            self.cs = np.zeros((self.k, 1083))
            self.count = np.zeros(self.k)
            for x in X:
                i = np.argmin([np.linalg.norm(x-c) for c in last_cs])
                self.cs[i] += x
                self.count[i] += 1
            for i in xrange(self.k):
                if self.count[i] != 0:
                    self.cs[i] /= self.count[i]

            move_len = sum((np.linalg.norm(x) for x in last_cs - self.cs))
            print 'move len', move_len
            if move_len < 0.05:
                break

            s = self.count.argsort()
            l = len([t for t in self.count if t == 0])
            print 'no point cs count', l
            print self.cs[s[self.k-1]]
            for i in xrange(l):
                self.cs[s[i]] = self.cs[s[self.k-i-1]] + np.random.random(1083)/10
            last_cs = self.cs

    def classify(self, x):
        return np.argmin([np.linalg.norm(x-c) for c in self.cs])


def get_img_as_vector(fn):
    im = cv2.imread(fn)
    im = im[:, :, 0]
    (retval, dst) = cv2.threshold(im, 128, 1, cv2.THRESH_BINARY_INV)
    return dst.reshape(dst.size)

k = KMeans(1600)    # 分1000个类别
fns = os.listdir('ocr')
train_fns = fns[0: 16000]
test_fns = fns[16000: ]
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
