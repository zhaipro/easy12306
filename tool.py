# coding: utf-8
import pathlib

import cv2
from sklearn.externals import joblib

import pretreatment


pathlib.Path('classify').mkdir(exist_ok=True)
texts = pretreatment.load_data()
texts.shape = (texts.shape[0], -1)
pl = joblib.load('pipeline.pkl')
labels = pl.predict(texts)
for idx, (text, label) in enumerate(zip(texts, labels)):
    # 使用聚类结果命名
    fn = f'classify/{label}({idx}).jpg'
    cv2.imwrite(fn, text)
