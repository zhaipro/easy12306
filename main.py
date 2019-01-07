# coding: utf-8
import sys

import cv2
import numpy as np
from keras import models

import pretreatment


def main(fn):
    # 读取并预处理验证码
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    text = pretreatment.get_text(img)
    imgs = pretreatment.get_imgs(img)
    _, text = cv2.threshold(text, 220, 1, cv2.THRESH_BINARY)
    h, w = text.shape
    text.shape = (1, h, w, 1)

    # 识别文字
    model = models.load_model('model.h5')
    label = model.predict(text)
    label = label.argmax()
    print(label)

    # 加载图片分类器
    data = np.load('images.npz')
    images, labels = data['images'], data['labels']
    labels = labels.argmax(axis=1)
    for pos, img in enumerate(imgs):
        try:
            img.dtype = np.uint64
            img = img[0]
            idx = list(images).index(img)
            label = labels[idx]
            print(pos // 4, pos % 4, label)
        except:
            print('unknown')


if __name__ == '__main__':
    main(sys.argv[1])
