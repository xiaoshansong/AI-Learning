## YOLOv3神经网络

### 安装及使用

1. 下载VOC数据集, 同时创建目录
	可运行下面的shell脚本，直接下载VOC数据集
	```Shell
	$ ./download_data.sh
	```

2. 下载YOLO权重[YOLO_small](https://drive.google.com/file/d/0B5aC8pI-akZUNVFZMmhmcVRpbTA/view?usp=sharing)
下载后将权重文件放置在 `data/weight` 目录下

3. 修改 `yolo/config.py` 文件可以修改YOLO配置

4. 在命令行下运行以下脚本，开始训练
	```Shell
	$ python train.py
	```

5. 在命令行下运行以下脚本，开始测试
	```Shell
	$ python test.py
	```

### 需求和依赖
1. Tensorflow
2. OpenCV
