# FireKitCV

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/Fire/blob/main/LICENSE) 
## 一、前言
FireKitCV is a deep learning Framework written in Python and used for Computer Vision tasks, include Image Classification, Object Detection, Seagmentation, etc. Running on Pytorch.



## 二、示例
首先git clone本项目

### 2.1 训练
1. 下载[fashion mnist](https://github.com/zalandoresearch/fashion-mnist)数据集的四个压缩包放到./data目录下，运行`python scripts/make_fashionmnist.py`自动提取图片并划分类别、验证集
2. 执行python train.py 训练
3. 执行python evaluate.py 测试（在config设置训练好的模型路径）

### 2.2 优化
* 迁移学习，下载对应模型的预训练模型，把路径填入config.py中
* 调整不同的模型、尺寸、优化器等等


## 三、支持

### 3.1 模型
* Resnet，Efficientnet, Swin Transformer, ConvNeXt等所有TIMM库所有模型

### 3.2 优化器
* Adam  
* SGD
* AdamW
* AdamMuon

### 3.3 学习率衰减
* ReduceLROnPlateau
* StepLR
* MultiStepLR
* SGDR

### 3.4 损失函数
* 交叉熵
* Focalloss

### 3.5 其他
* Metric(acc, F1)
* 训练日志保存
* 交叉验证
* 梯度裁剪
* earlystop
* weightdecay
* 冻结
* labelsmooth


## 五、Refer
1. [albumentations](https://github.com/albumentations-team/albumentations)
2. [timm](https://github.com/huggingface/pytorch-image-models)
