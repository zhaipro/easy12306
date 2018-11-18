#! env python3
# coding: utf-8
# 功能：抓取验证码
# 并存放到img目录下
# 文件名为图像的MD5
import hashlib

import requests

import utils


def download_img():
    url = 'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand'
    response = requests.get(url)
    fn = hashlib.md5(response.content).hexdigest()
    with open(f'img/{fn}.jpg', 'wb') as fp:
        fp.write(response.content)


if __name__ == '__main__':
    utils.mkdir('img')
    i = 0
    while True:
        try:
            download_img()
            i += 1
            print(i)
        except:
            print('error')
