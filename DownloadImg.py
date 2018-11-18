#! env python
# coding: utf-8
# 功能：抓取验证码
# 并存放到img目录下
# 文件名为图像的MD5
import hashlib
import sys
import time

import requests

import greenpool
import utils


def download_img(idx):
    print(idx)
    sys.stdout.flush()
    url = 'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand'
    response = requests.get(url)
    fn = hashlib.md5(response.content).hexdigest()
    with open(f'img/{fn}.jpg', 'wb') as fp:
        fp.write(response.content)


if __name__ == '__main__':
    utils.mkdir('img')
    # for idx in range(1000):
    #     download_img(idx)
    greenpool.spawn(download_img, range(1000))
    time.process_time()
