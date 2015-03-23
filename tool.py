# -*- coding: cp936 -*-
import os
import shutil

# 统计有多少聚类中心是有样本的
l = [0]*1000
fp = open('10000测试结果.txt')
for line in fp:
    (fn, classify) = line.strip().split(' ')
    l[int(classify)] += 1
print len([x for x in l if x>0])


'''
# 将聚类后的样本复制并使用聚类结果命名
fp = open('10000测试结果.txt')
i = 0
for line in fp:
    (fn, classify) = line.strip().split(' ')
    shutil.copy(os.path.join('ocr', fn), 'classify/%s(%d).jpg' % (classify, i))
    i += 1
'''
