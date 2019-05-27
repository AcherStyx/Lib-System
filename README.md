# Lib-System

Library management system  
图书馆管理系统，关于位置占用状况的识别

## 前端

## 后端

## 识别

### `ImageIO.py`

&emsp;&emsp;包含关于从视频或相机获取输入并简单处理的类。仅支持视频或摄像头类的输入。

### `ImageAnalyze.py`

&emsp;&emsp;基于opencv库创建处理连续图像数据的函数、类。  
&emsp;&emsp;包含图像的特征点获取、差异比较、LK金字塔光流法运动追踪、稠密光流法计算（目前还很不稳定）以及蒙版的创建和变化百分比的检测。  
&emsp;&emsp;占用百分比的计算通过确定一系列点圈取多边形来完成。