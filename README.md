Video source: ViSR Dataset (Liu, X., Liu, W., Zhang, M., Chen, J., Gao, L., Yan, C., & Mei, T. (2019). Social relation recognition from videos via multi-scale spatial-temporal reasoning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 3566-3574).)

movie: 《3 idiots》a 2009 Indian Hindi-language coming-of-age comedy-drama film

# Face Recognition

1.按要求创建环境并激活: https://blog.csdn.net/qq_42742522/article/details/118711256

```powershell
# 1.创建环境
conda create -n tf18 Python=3.6
# 2.激活环境
conda activate tf18
# 3.依赖包: ensorflow，numpy和scipy需要指定版本，不然会出错。
tensorflow == 1.7
numpy == 1.16.2
scipy == 1.2.1
scikit-learn
opencv-python
h5py
matplotlib
Pillow
requests
psutil
```


2.下载Facenet Github: https://github.com/davidsandberg/facenet

按照**blog的教程**，对facenet文件夹进行处理，以防出错。

(1) 删去没有用的文件夹(以防出错)，将剩余facenet文件夹放置在anaconda的packages路径下 

(2) align也要放到packages路径下



3.将要提取人脸的图片，以**单张图片放在一个文件夹**的形式存储。

```powershell
python  facenet/src/align/align_dataset_mtcnn.py   facenet/datasets/movie/raw  facenet/datasets/movie/mtcnn --image_size 160 --margin 32 --random_order
```



# Identity Recognition

1.创建Pytorch环境

2.安装Facenet(Pytorch)

3.train中的图片转换为512维的特征向量(embeddings.pt)

4.test中的图片测试特征向量的准确率(accuracy.csv)



# Emotion Recognition

## a fine-tune CNN model (ResNet-78) with the FER 2013 dataset

- 模型结果存储
  - 参数: checkpoints/ResNet78.bin
  - 结构: models/FineTuneModel.py
- 训练数据: data/fer2013
  - 训练集
  - 验证集: 调整超参数
  - 测试集
- **主要函数: main.py**
  - train()：训练模型
    - epoch数：机器学习次数 (机器只学习一遍所有数据学不会)
    - batch_size：深度学习为并行学习，将大量数据分成若干小块，一起跑
  - use_model()：输入图片，导出情绪识别结果; 修改all_path即可。
- runs: 训练结果的可视化
- Target: 0 anger 生气； 1 disgust 厌恶； 2 fear 恐惧； 3 happy 开心；4 sad 伤心； 5 surprised 惊讶；6 neutral 中性

## use model to detect emotion

- detect emotion: emotion_detect.py

- transfer video to frame: Video/video_to_frame.py
- Raju in different contexts as an example: Raju.ipynb
  - Identity and emotion detection are plotted in raw images. (Plot_results)
  - Save results in each frame as a csv file. (Data_results)



# Context Recognition

1.随机挑选图片，用yolov8n做场景中的物体识别。

2.可采用无监督的聚类方法，基于物体的出现频率探索共有几个主要场景。

3.也可基于经验，总结场景对应的物体识别条件，并基于总结的规则判断场景。



# Interactive Website

参考的模板：https://dash.gallery/dash-food-footprint/

模板源码：https://github.com/InesRoque3/GroupV_project2

dashR的安装

```R
install.packages(c("fiery", "routr", "reqres", "htmltools", "base64enc", "plotly", "mime", "crayon", "devtools"))

# installs dash, which includes dashHtmlComponents, dashCoreComponents, and dashTable
# and will update the component libraries when a new package is released
devtools::install_github("plotly/dashR", ref="dev", upgrade = TRUE)
```

