import os

import cv2
import numpy as np

import pretreatment


path = 'imgs'
texts = []
images = []
i = 0
for fn in os.listdir(path):
    i += 1
    fn = os.path.join(path, fn)
    im = cv2.imread(fn)
    text = pretreatment.get_text(im)
    texts.append(text)
    images.append(list(pretreatment._get_imgs(im)))
    if i >= 2000:
        break
np.savez_compressed('dataset.npz', texts=texts, images=images)
