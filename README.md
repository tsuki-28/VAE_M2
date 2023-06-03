VAE_M2
===

VAE网络与生成式半监督相结合的小样本学习网络



## 概述

![绘图2](E:\毕设\代码交接\VAE_M2\绘图2.png)

## 环境配置

- windows 10
- pytorch 3.7

## 数据集

- `data`: 原始数据
- `process_data_n_new`：经过滤波和降采样后处理的数据
- `process_data_nolttb`：仅采用滤波处理的数据
- `process_data_nomean`：仅降采样处理的数据

## 代码说明

##### 1.小样本眼动姿势识别模型训练步骤及说明

- 1.1`vae_getdata.py`：对原始数据进行处理，并保存

  ```
   vae_getdata.py:
     ├─类myDataSet
     │  ├─函数dataProcess：对原始数据进行滤波和降采样处理
     │  │  
     │  ├─函数dataProcess_2：调用已处理数据
     │  │  
     │  ├─函数dataProcess_nolttb：仅采用滤波处理的数据
     │  │  
     │  ├─函数dataProcess_nomean：仅降采样处理的数据
     ├─类mySubset
     │
     └─函数data_set_split：将数据划分为训练集和测试集
     │ 
     └─函数savedata：保存数据到新建文件夹中
  ```

  

- 1.2.`VAE.py`：训练VAE网络，并保存生成模型参数

```
 VAE.py:
   ├─类Encode
   │  
   ├─类mDecode
   │
   └─类VAE
```

- 1.3.`vae_classify.py`：训练分类模型，并保存结果

##### 2.不同方法实验对比模型

`svm.py`：采用SVM模型训练

`knn.py`：采用KNN模型训练

`gan_classification`：VAE更换为GAN网络，进行数据生成

注：`vae_utils.py`：包含模型训练时用到的工具函数







