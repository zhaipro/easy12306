# coding: utf-8
import os
import shutil
import sys

import utils

result_fn = sys.argv[1]
classify_fn = sys.argv[2]

utils.mkdir(classify_fn)

# 用于统计有多少聚类中心是有样本的
s = set()
fp = open(result_fn)
for line in fp:
    fn, classify = line.split()
    s.add(classify)
print(len(s))

# 将聚类后的样本复制并使用聚类结果命名
fp.seek(0)
for idx, line in enumerate(fp):
    fn, classify = line.strip().split()
    src = os.path.join('ocr', fn)
    dst = f'{classify_fn}/{classify}({idx}).jpg'
    shutil.copy(src, dst)
