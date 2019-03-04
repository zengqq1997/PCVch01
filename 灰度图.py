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