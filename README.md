# easy12306

两个必要的数据集：

1. 文字识别，model.h5
2. 图片识别，images.npz

识别器数据的下载地址：

https://pan.baidu.com/s/1OsBIBM4rl8EnpZt7VYiD9g

`python main.py <img.jpg>`

我把设计思路写在维基中了：https://github.com/zhaipro/easy12306/wiki

### 如何？

![2](https://user-images.githubusercontent.com/8620842/51320752-d6f2cc00-1a9b-11e9-9d2d-7d1e25ddadc5.jpg)

```
~$ python3 main.py 2.jpg
41      # 要找的东西是41
0 0 41  # 第一行第一列就是41
0 1 39
0 2 73
0 3 73
1 0 33
1 1 41
1 2 31  # 最后的这两个是同一种东西
1 3 31
```

具体的编号：https://gist.github.com/zhaipro/97d46ff244f66d17961a5f1ef17b6d8f

识别前所未见的图片

![8](https://user-images.githubusercontent.com/8620842/51799645-a01c7300-225e-11e9-8214-296773112484.jpg)

```
~$ python3 mlearn_for_image.py 8.jpg
[0.8991613]  # 可信度
[0]          # 0 表示的就是打字机
```

源码很乱，整理中…
