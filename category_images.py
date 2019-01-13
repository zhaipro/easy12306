import numpy as np

import mlearn
from pretreatment import load_data


def learn():
    texts, imgs = load_data()
    labels = mlearn.predict(texts)
    labels = labels.argmax(axis=1)
    imgs.dtype = np.uint64
    imgs.shape = (-1, 8)
    unique_imgs = np.unique(imgs)
    print(unique_imgs.shape)
    imgs_labels = []
    for img in unique_imgs:
        idxs = np.where(imgs == img)[0]
        counts = np.bincount(labels[idxs], minlength=80)
        imgs_labels.append(counts)
    np.savez('images.npz', images=unique_imgs, labels=imgs_labels)


if __name__ == '__main__':
    learn()
