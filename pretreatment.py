#! env python
# coding: utf-8
# 功能：对图像进行预处理，将文字部分单独提取出来
# 并存放到ocr目录下
# 文件名为原验证码文件的文件名
import cv2
import os

import utils


def read_img(fn):
    '''
    得到验证码完整图像
    :param fn:图像文件路径
    :return:图像对象
    '''
    return cv2.imread(fn)


def write_img(im, fn):
    cv2.imwrite(fn, im)


def get_text_img(im):
    '''
    得到图像中的文本部分
    '''
    return im[3:22, 127:184]


def binarize(im):
    '''
    二值化图像
    '''
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    (retval, dst) = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    return dst


def show_img(im):
    print(im.ndim, im.dtype)
    cv2.imshow("image", im)
    cv2.waitKey(0)


if __name__ == '__main__':
    utils.mkdir('ocr')
    for img_name in os.listdir('img'):
        im = read_img(os.path.join('img', img_name))
        im = get_text_img(im)
        im = binarize(im)
        # show_img(im)
        write_img(im, os.path.join('ocr', img_name))
