# coding: utf-8
import os


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
