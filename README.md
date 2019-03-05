# PCV-chapter01

开始记录学习Python Computer Vision的过程

第一次

## 一、说明

本次学习是使用[python(x,y)2.7.6版本](ftp://ftp.ntua.gr/pub/devel/pythonxy/)，该版本提供了此次学习所需要的大部分库以及所需要的编译器（spyder）。

## 二、安装准备工作

### 1.准备

你需要的主要四个文件包括[python(x,y)2.7.6版本](ftp://ftp.ntua.gr/pub/devel/pythonxy/)安装包、[PCV库](https://github.com/willard-yuan/pcv-book-code)、[VLfeat](http://www.vlfeat.org/download.html)和本书用到的数据库。

### 2.安装python(x,y)

在Windows下，推荐你安装Python(x,y) 2.7.6。Python(x,y) 2.7.6是一个库安装包，除了包含Python自身外，还包含了很多第三方库，下面是安装Python(x,y)时的界面：

![image](https://github.com/zengqq1997/PCVch01/blob/master/images/python(x,y).jpg)

安装完成后，为验证安装是否正确，可以在Python shell里确认一下OpenCV是否已安装来进行验证，在Python Shell里输入下面命令：

```python
from cv2 import __version__
__version__
```

输入上面命令，如果可以看到OpenCV的版本信息，则说明python(x,y)已安装正确。

### 3.安装PCV

PCV库文件的链接已经在上方给出，读者们可以自行下载，下载下来后在cmd上输入：

```python
cd PCV
python setup.py install
```

运行上面命令，即可完成PCV库的安装。为了验证PCV库是否安装成功，在运行上面命令后，可以打开Python自带的Shell，在Shell输入：

```python
import PCV
```

如果未报错，则表明你已成功安装了该PCV库。

### 4.VLfeat

VLFeat是一个跨平台的开源机器视觉库，它囊括了当前流行的机器视觉算法，如SIFT, MSER, HOG, 同时还包含了诸如K-MEANS, Hierarchical K-means的聚类算法。

![image](https://github.com/zengqq1997/PCVch01/blob/master/images/VLfeat.jpg)

## 三、Python Computer Vision

利用python语言的计算机视觉编程，学习对象识别、3D重建、立体图像、增强现实及其他计算机视觉应用技巧。

## 四、灰度图

利用PILLOW库实现对图像基本处理，将图像由RGB图转换为灰度图像

### 1.代码

```python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 20:07:33 2019

@author: ZQQ
"""

# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
figure()

pil_im = Image.open('C:/Users/ZQQ/Desktop/advanced/study/computer vision/images/zqq.jpg')
gray()
subplot(121)
title(u'原图',fontproperties=font)
axis('off')
imshow(pil_im)

pil_im = Image.open('C:/Users/ZQQ/Desktop/advanced/study/computer vision/images/zqq.jpg').convert('L')
subplot(122)
title(u'灰度图',fontproperties=font)
axis('off')
imshow(pil_im)

show()
```

### 2.实验结果图

![image](https://github.com/zengqq1997/PCVch01/blob/master/images/%E7%81%B0%E5%BA%A6%E5%9B%BE.jpg)

## 五、图像轮廓和直方图

图像轮廓线和图线等高线。在画图像轮廓前需要转换为灰度图像，因为轮廓需要获取每个坐标[x,y]位置的像素值。下面是画图像轮廓和直方图的代码：

### 1.代码

```python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 20:31:16 2019

@author: ZQQ
"""

 # -*- coding: utf-8 -*-
from PIL import Image
from pylab import *

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
im = array(Image.open('C:/Users/ZQQ/Desktop/advanced/study/computer vision/images/zqq.jpg').convert('L'))  # 打开图像，并转成灰度图像

figure()
subplot(121)
gray()
contour(im, origin='image')
axis('equal')
axis('off')
title(u'图像轮廓', fontproperties=font)

subplot(122)
hist(im.flatten(), 128)
title(u'图像直方图', fontproperties=font)
plt.xlim([0,260])
plt.ylim([0,11000])


show()
```

### 2.实验结果图

![image](https://github.com/zengqq1997/PCVch01/blob/master/%E7%81%B0%E5%BA%A6%E5%9B%BE.py)



## 六、高斯滤波（高斯模糊、高斯差分）

- 高斯模糊：一个经典的并且十分有用的图像卷积例子是对图像进行高斯模糊。高斯模糊可以用于定义图像尺度、计算兴趣点以及很多其他的应用场合。
- 高斯差分：图像强度的改变是一个重要的信息，被广泛用以很多应用中，高斯差分便是一个应用。

### 1.代码

```python
#高斯模糊
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 20:42:04 2019

@author: ZQQ
"""

# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
from scipy.ndimage import filters

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

#im = array(Image.open('board.jpeg'))
im = array(Image.open('C:/Users/ZQQ/Desktop/advanced/study/computer vision/images/zqq.jpg').convert('L'))

figure()
gray()
axis('off')
subplot(1, 4, 1)
axis('off')
title(u'原图', fontproperties=font)
imshow(im)

for bi, blur in enumerate([2, 5, 10]):
  im2 = zeros(im.shape)
  im2 = filters.gaussian_filter(im, blur)
  im2 = np.uint8(im2)
  imNum=str(blur)
  subplot(1, 4, 2 + bi)
  axis('off')
  title(u'标准差为'+imNum, fontproperties=font)
  imshow(im2)


show()
```

```python
#高斯差分
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 20:44:14 2019

@author: ZQQ
"""

 # -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
from scipy.ndimage import filters
import numpy

# 添加中文字体支持
#from matplotlib.font_manager import FontProperties
#font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

def imx(im, sigma):
    imgx = zeros(im.shape)
    filters.gaussian_filter(im, sigma, (0, 1), imgx)
    return imgx


def imy(im, sigma):
    imgy = zeros(im.shape)
    filters.gaussian_filter(im, sigma, (1, 0), imgy)
    return imgy


def mag(im, sigma):
    # there's also gaussian_gradient_magnitude()
    #mag = numpy.sqrt(imgx**2 + imgy**2)
    imgmag = 255 - numpy.sqrt(imgx ** 2 + imgy ** 2)
    return imgmag


im = array(Image.open('C:/Users/ZQQ/Desktop/advanced/study/computer vision/images/zqq.jpg').convert('L'))
figure()
gray()

sigma = [2, 5, 10]

for i in  sigma:
    subplot(3, 4, 4*(sigma.index(i))+1)
    axis('off')
    imshow(im)
    imgx=imx(im, i)
    subplot(3, 4, 4*(sigma.index(i))+2)
    axis('off')
    imshow(imgx)
    imgy=imy(im, i)
    subplot(3, 4, 4*(sigma.index(i))+3)
    axis('off')
    imshow(imgy)
    imgmag=mag(im, i)
    subplot(3, 4, 4*(sigma.index(i))+4)
    axis('off')
    imshow(imgmag)

show()
```

### 3.实验结果图

高斯模糊:下面第一幅图为待模糊图像，第二幅用高斯标准差为2进行模糊，第三幅用高斯标准差为5进行模糊，最后一幅用高斯标准差为10进行模糊。

![image](https://github.com/zengqq1997/PCVch01/blob/master/images/gauss_mohu.jpg)

高斯差分

![image](https://github.com/zengqq1997/PCVch01/blob/master/images/gauss_chafen.jpg)

## 七、直方图均衡化

一个极其有用的例子是灰度变换后进行直方图均衡化。图像均衡化作为预处理操作，在归一化图像强度时是一个很好的方式，并且通过直方图均衡化可以增加图像对比度。下面是对图像直方图进行均衡化处理的例子：

### 1.代码

```python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 20:47:37 2019

@author: ZQQ
"""

 # -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
from PCV.tools import imtools

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

im = array(Image.open('C:/Users/ZQQ/Desktop/advanced/study/computer vision/images/zqq.jpg').convert('L'))  # 打开图像，并转成灰度图像
#im = array(Image.open('../data/AquaTermi_lowcontrast.JPG').convert('L'))
im2, cdf = imtools.histeq(im)

figure()
subplot(2, 2, 1)
axis('off')
gray()
title(u'原始图像', fontproperties=font)
imshow(im)

subplot(2, 2, 2)
axis('off')
title(u'直方图均衡化后的图像', fontproperties=font)
imshow(im2)

subplot(2, 2, 3)
axis('off')
title(u'原始直方图', fontproperties=font)
#hist(im.flatten(), 128, cumulative=True, normed=True)
hist(im.flatten(), 128, normed=True)

subplot(2, 2, 4)
axis('off')
title(u'均衡化后的直方图', fontproperties=font)
#hist(im2.flatten(), 128, cumulative=True, normed=True)
hist(im2.flatten(), 128, normed=True)

show()
```

### 2.实验结果图

![image](https://github.com/zengqq1997/PCVch01/blob/master/images/zhifantujunheng.jpg)

## 八、小结

此次内容包含了下载安装python(x,y)并配置opencv-python，和PCV库。实现了灰度图的转换、高斯滤波器的使用、直方图以及直方图均衡化、图像轮廓。





