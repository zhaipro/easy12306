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


def whash(image):
    """
    Wavelet Hash computation.

    based on https://www.kaggle.com/c/avito-duplicate-ads-detection/
    @image must be a PIL instance.
    """
    ll_max_level = int(np.log2(min(image.shape)))
    image_scale = 2**ll_max_level

    level = 3
    dwt_level = ll_max_level - level

    image = cv2.resize(image, (image_scale, image_scale))
    pixels = image / 255

    # Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
    coeffs = pywt.wavedec2(pixels, 'haar', level = ll_max_level)
    coeffs[0][:] = 0
    pixels = pywt.waverec2(coeffs, 'haar')

    # Use LL(K) as freq, where K is log2(@hash_size)
    coeffs = pywt.wavedec2(pixels, 'haar', level = dwt_level)
    dwt_low = coeffs[0]

    # Substract median and compute hash
    med = np.median(dwt_low)
    diff = dwt_low > med

    diff = np.packbits(diff)
    return diff


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
