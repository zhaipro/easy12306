#! env python
# coding: utf-8
# 功能：对图像进行预处理，将文字部分单独提取出来
# 并存放到ocr目录下
# 文件名为原验证码文件的文件名
import hashlib
import os
import pathlib

import cv2
import numpy as np
import requests


PATH = 'imgs'


def download_image():
    # 抓取验证码
    # 存放到指定path下
    # 文件名为图像的MD5
    url = 'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand'
    r = requests.get(url)
    fn = hashlib.md5(r.content).hexdigest()
    with open(f'{PATH}/{fn}.jpg', 'wb') as fp:
        fp.write(r.content)


def download_images():
    pathlib.Path(PATH).mkdir(exist_ok=True)
    for idx in range(40000):
        download_image()
        print(idx)


def pretreat():
    if not os.path.isdir(PATH):
        download_images()
    imgs = []
    for img in os.listdir(PATH):
        img = os.path.join(PATH, img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # 得到图像中的文本部分
        img = img[3:22, 120:177]
        imgs.append(img)
    return imgs


def load_data(path='data.npy'):
    if not os.path.isfile(path):
        imgs = pretreat()
        np.save(path, imgs)
    return np.load(path)


if __name__ == '__main__':
    imgs = load_data()
    print(imgs.shape)
    cv2.imwrite('temp.jpg', imgs[0])
