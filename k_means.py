#! env python
# coding: utf-8
# 功能：对文字部分使用k-means算法进行聚类
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from pretreatment import load_data


def main():
    # 读取训练用数据
    texts = load_data()
    n = texts.shape[0]
    texts.shape = (n, -1)
    print('Shape:', texts.shape)
    # 训练
    n_clusters = 80     # 聚类中心个数
    pca = PCA(n_components=0.99, whiten=True)
    kmeans = KMeans(n_clusters, verbose=True)
    pl = Pipeline([('pca', pca), ('kmeans', kmeans)])
    pl.fit(texts)
    joblib.dump(pl, 'pipeline.pkl')


if __name__ == '__main__':
    main()
