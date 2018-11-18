# coding: utf-8
import sys

import cv2
import numpy as np

import pretreatment
import utils

result_fn = sys.argv[1]
classify_fn = sys.argv[2]

utils.mkdir(classify_fn)

# 用于统计有多少聚类中心是有样本的
result = np.load(result_fn)
print(np.unique(result).shape)

# 将聚类后的样本复制并使用聚类结果命名
imgs = pretreatment.load_data()
for idx, (img, classify) in enumerate(zip(imgs, result)):
    dst = f'{classify_fn}/{classify}({idx}).jpg'
    cv2.imwrite(dst, img)
