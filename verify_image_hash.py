# coding: utf-8
import pathlib

import cv2
import numpy as np
import scipy.fftpack


def avhash(im):
    im = cv2.resize(im, (8, 8), interpolation=cv2.INTER_CUBIC)
    avg = im.mean()
    im = im > avg
    im = np.packbits(im)
    return im


def phash(im):
    im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_CUBIC)
    im = scipy.fftpack.dct(scipy.fftpack.dct(im, axis=0), axis=1)
    im = im[:8, :8]
    med = np.median(im)
    im = im > med
    im = np.packbits(im)
    return im


def phash_simple(im):
    im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_CUBIC)
    im = scipy.fftpack.dct(im)
    im = im[:8, 1:8 + 1]
    avg = im.mean()
    im = im > avg
    im = np.packbits(im)
    return im


def dhash(im):
    im = cv2.resize(im, (8 + 1, 8), interpolation=cv2.INTER_CUBIC)
    im = im[:, 1:] > im[:, :-1]
    im = np.packbits(im)
    return im


def dhash_vertical(im):
    im = cv2.resize(im, (8, 8 + 1), interpolation=cv2.INTER_CUBIC)
    im = im[1:, :] > im[:-1, :]
    im = np.packbits(im)
    return im


def whash(im):
    pass
    # 不是说我不做，我是真的看不懂其源码


def verify(_hash):
    # 用验证集测试各哈希函数的效果
    data = np.load('captcha.npz')
    images, labels = data['images'], data['labels']
    print(images.shape)
    himages = {}
    for idx, (img, label) in enumerate(zip(images, labels)):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = _hash(img)
        img.dtype = np.uint64
        img = img[0]
        if himages.get(img, (label,))[0] != label:
            cv2.imwrite(f'errors/{idx}.{label}.jpg', images[idx])
            pre_label, pre_idx = himages[img]
            cv2.imwrite(f'errors/{idx}.{pre_label}.jpg', images[pre_idx])
        else:
            himages[img] = label, idx
    print(len(himages))


if __name__ == '__main__':
    pathlib.Path('errors').mkdir(exist_ok=True)
    # verify(avhash)
    # 我觉得下面这个是最佳的
    verify(phash)
    # verify(phash_simple)
    # verify(dhash)
    # verify(dhash_vertical)
