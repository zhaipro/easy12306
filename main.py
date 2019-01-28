# coding: utf-8
import sys

import cv2
import numpy as np
from keras import models

import pretreatment


def main(fn):
    # 读取并预处理验证码
    img = cv2.imread(fn)
    text = pretreatment.get_text(img)
    text = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
    imgs = np.array(list(pretreatment._get_imgs(img)))
    imgs = imgs / 255.0
    text = text / 255.0
    h, w = text.shape
    text.shape = (1, h, w, 1)
    _, h, w, _ = imgs.shape
    imgs.shape = (-1, h, w, 3)

    # 识别文字
    model = models.load_model('model.h5')
    label = model.predict(text)
    label = label.argmax()
    print(label)

    # 加载图片分类器
    model = models.load_model('12306.image.model.h5')
    labels = model.predict(imgs)
    labels = labels.argmax(axis=1)
    for pos, label in enumerate(labels):
        print(pos // 4, pos % 4, label)


if __name__ == '__main__':
    main(sys.argv[1])
