# easy12306

两个必要的数据集：

1. 文字识别，model.h5
2. 图片识别，12306.image.model.h5

识别器数据的下载地址：

https://pan.baidu.com/s/1OsBIBM4rl8EnpZt7VYiD9g

`python3 main.py <img.jpg>`

我把设计思路写在维基中了：https://github.com/zhaipro/easy12306/wiki

### 如何？

![2](https://user-images.githubusercontent.com/8620842/51320752-d6f2cc00-1a9b-11e9-9d2d-7d1e25ddadc5.jpg)

```
~$ python3 main.py 2.jpg 2> /dev/null
电子秤
风铃        # 要找的是以上两样东西
0 0 电子秤  # 第一行第一列就是电子秤
0 1 绿豆
0 2 蒸笼
0 3 蒸笼
1 0 风铃
1 1 电子秤
1 2 网球拍
1 3 网球拍
```

识别前所未见的图片

![8](https://user-images.githubusercontent.com/8620842/51799645-a01c7300-225e-11e9-8214-296773112484.jpg)

具体的编号：[texts.txt](./texts.txt)

```
~$ python3 mlearn_for_image.py 8.jpg
[0.8991613]  # 可信度
[0]          # 0 表示的就是打字机
```

### 在线体验

识别验证码，暂不识别多标签。

http://shell.teachx.cn:12306/

![a](https://user-images.githubusercontent.com/8620842/51885312-e809da00-23c5-11e9-93a3-78d5e8b4ac18.png)

识别单个图片，可任意尺寸（总之由cv2简单的将其转为指定尺寸）。

http://shell.teachx.cn:5000/

![a](https://user-images.githubusercontent.com/8620842/51879603-21831b00-23af-11e9-8d16-9ae64866ca4c.jpg)
