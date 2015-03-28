# -*- coding: cp936 -*-
import os
import shutil
import sys

result_fn = sys.argv[1]
classify_fn = sys.argv[2]

# 用于统计有多少聚类中心是有样本的
s = set()
fp = open(result_fn)
for line in fp:
    (fn, classify) = line.strip().split(' ')
    s.add(int(classify))
fp.close()
print len(s)

# 将聚类后的样本复制并使用聚类结果命名
fp = open(result_fn)
for idx, line in enumerate(fp):
    (fn, classify) = line.strip().split(' ')
    shutil.copy(os.path.join('ocr', fn), '%s/%s(%d).jpg' % (classify_fn, classify, idx))
