# coding: utf-8
import base64

import cv2
import numpy as np
import requests


AK = 'unknown'
SK = 'unknown'


def get_token(ak, sk):
    # http://ai.baidu.com/docs#/Auth/top
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {
        'grant_type': 'client_credentials',
        'client_id': ak,
        'client_secret': sk,
    }
    r = requests.post(url, params=params)
    return r.json()['access_token']


TOKEN = get_token(AK, SK)


def ocr(img):
    # 文件名
    if isinstance(img, str):
        img = open(img, 'rb').read()
    # 或cv2图像
    elif isinstance(img, np.ndarray):
        _, img = cv2.imencode('.jpg', img)
    # https://ai.baidu.com/docs#/OCR-API/e1bd77f3
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic'
    params = {'access_token': TOKEN}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    img = base64.b64encode(img)
    data = {'image': img}
    r = requests.post(url, data=data, params=params, headers=headers)
    # 该项目只需要一个词
    return r.json()['words_result'][0]['words']


def main():
    import sys
    from pretreatment import load_data
    texts, _ = load_data()
    fp = open('texts.log', 'w', encoding='utf-8')
    for idx, text in enumerate(texts):
        try:
            text = ocr(text)
            print(idx, text, file=fp)
            print(idx, text)
        except Exception as e:
            print(e, file=sys.stderr)


if __name__ == '__main__':
    main()
