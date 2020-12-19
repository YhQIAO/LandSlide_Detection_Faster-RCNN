@[TOC](目录)

# Faster-RCNN实现遥感图像滑坡识别
本代码参考[github链接](https://github.com/bubbliiiing/faster-rcnn-pytorch)

# 操作指南
##  对数据进行测试
环境python+cuda+cuDNN+pytorch（还有其他一些杂七杂八的，报错提示有别的库没下就自己下一下）
在frcnn.py文件中，有如下代码：

```python
class FRCNN(object):
    _defaults = {
        "model_path"    : 'logs/Epoch49-Total_Loss0.2045-Val_Loss0.4614.pth',
        "classes_path"  : 'model_data/slide_class.txt',
        "confidence"    : 0.5,
        "iou"           : 0.3,
        "backbone"      : "resnet50",
        "cuda"          : True,
    }
.....
```
model_path表示你选择的权重文件的路径(默认已经放了一个)
运行predict.py，输入文件路径即可进行测试

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201219194438964.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201219194508663.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70#pic_center =300x)
## 数据集进行训练
resnet预训练权重百度网盘地址：[提取码vs9n](https://pan.baidu.com/s/1HGCku2t-zoroH30JsKWPJA)
预训练权重文件放到model_data文件夹下

landslide_train.txt文件每一行记录了图片的相对路径，以及标识框的坐标，类别（本例中只有一个类就是滑坡，数值位0）
运行train.py进行训练
训练过程中每一次迭代生成的权重文件会自动保存到logs文件夹下，这些权重文件就是训练出来的模型。理论上迭代次数最多的模型性能越好
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201219194936912.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70#pic_center =300x)

# 数据集
数据集百度网盘地址：[提取码mnd6](https://pan.baidu.com/s/1950sOcFfDFU6UWz-Dsm_7Q)
resnet预训练权重百度网盘地址：[提取码vs9n](https://pan.baidu.com/s/1HGCku2t-zoroH30JsKWPJA)
把下载好的数据集中的所有图片复制到项目LandSlide_Detection_Faster-RCNN\LandSlideDataSet\images文件夹下
预训练权重文件放到model_data文件夹下

## 自己制作数据集
详细教程可以参考[bilibili视频链接](https://www.bilibili.com/video/BV1BK41157Vs?p=14)
voc2frcnn用于在LandSlideDataSet/ImageSets文件下生成train.txt文件
voc_annotation用于在项目文件夹下生成landslide_train.txt
