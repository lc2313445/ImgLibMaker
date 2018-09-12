# <font color=#00ffa0 face=""><center>ImgLibMaker Manual</center></font>


## 1. Introduction

This is a simple program to zip image file(.jpg) to .plk file with Python pickle library. The python version is 3.6.4.

If you are working on Image classify/object detection area and you do not want to use those standard image library(mnist/cifar .etc), this program is suitable for you.

## 2. How to use

The main program is **ImageHandle.py** and there are four input arguments for the program:
* --Path_Dir
* --reshape
* --img_num_each_file
* --test_num_each_class

--Path_Dir is the direction of your images stored, your image recommanded to store in one folder like this:

<center>
  <img src="https://github.com/lc2313445/Image_Store/blob/master/ImageHandle1.png" alt="Img Broken" border=3 width="60%" height="60%"/>
</center>

So for mine, the --Path_Dir is "D:\xuexiziliao\Proj\VS PROJ\P1\RGB_IMG" shown on the up dialog. In this folder have several items, contains three folders and other documents.

It is **<font color=#990000>very important</font>** that you can only contains folders which contains your images, for example, mine have three folders: P-0000,P-0001,P-0002. If you want to make your own image library, the folder name must follow the style: P-XXXX.

So what's the meaning? I have three image classes, first class is 'human hands', those pictures are all contains in P-0000, second class is 'mouse' which contains in P-0001 and the last is 'cell phone' which contains in P-0002. If you run this program, it will generate a (or several) .plk file(s) which contains the raw images and their labels(classes), so P-0000 will have label '0', P-0001 will have label '1' etc.

So do not contains any folder which is not for zip your images, and the name must be P-XXXX








### aa
#### aa
##### aa

```python
import numpy as np

```
1.  <p> This is code </p>
<pre><code>
  write_log(Path_Dir)
  &amp;copy; asarray
</code><pre>
* c
***
---

I got a web [__baidu__] [id]

  [id]: www.baidu.com "aaa"


this is <code>printf()</code> func
