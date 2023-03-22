# TF-FASTER-RCNN

## 需求和依赖
  - Tensorflow installation. 
  - Python


## 依赖和程序的安装
1. 获取代码。通过git命令克隆仓库到本地
  ```Shell
  git clone https://github.com/endernewton/tf-faster-rcnn.git
  ```

2. 通过以下命令，更新匹配设备GPU
  ```Shell
  cd tf-faster-rcnn/lib
  vim setup.py
  ```

3. 安装Cpython模块
  ```Shell
  make clean
  make
  cd ..
  ```

4. 获取安装COCO数据接口支持 [Python COCO API](https://github.com/pdollar/coco). 
  ```Shell
  cd data
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI
  make
  cd ../../..
  ```

## 使用预训练的模型进行测试
1. 运行fetch_faster_rcnn_models.sh下载预训练模型
  ```Shell
  # Resnet101 for voc pre-trained on 07+12 set
  ./data/scripts/fetch_faster_rcnn_models.sh
  ```
  如果下载预训练模型失败，以下为其他的下载地址，需要科学上网
  - Another server [here](http://xinlei.sp.cs.cmu.edu/xinleic/tf-faster-rcnn/).
  - Google drive [here](https://drive.google.com/open?id=0B1_fAEgxdnvJSmF3YUlZcHFqWTQ).

2. 创建文件夹结构，方便运行
  ```Shell
  NET=res101
  TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
  mkdir -p output/${NET}/${TRAIN_IMDB}
  cd output/${NET}/${TRAIN_IMDB}
  ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
  cd ../../..
  ```

3. 测试demo.py
  ```Shell
  GPU_ID=0 **GPUID号**
  CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
  ```
  **注意**: 
  测试需要消耗大量的GPU内存（显存），如果显存低于3G，推荐使用CPU模式版本

4. 使用预训练模型测试
  ```Shell
  GPU_ID=0
  ./experiments/scripts/test_faster_rcnn.sh $GPU_ID pascal_voc_0712 res101
  ```

## 训练自己的模型
1. 下载预先训练的模型和权重。 目前的代码支持VGG16和Resnet V1型号。Slim提供预先训练的模型，您可以获得预先训练的模型 [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) 并将它们放置在 ``data/imagenet_weights`` 文件夹。
如果是VGG16模型，可以通过如下命令设置:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ../..
   ```
   For Resnet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
   tar -xzvf resnet_v1_101_2016_08_28.tar.gz
   mv resnet_v1_101.ckpt res101.ckpt
   cd ../..
   ```

2. 训练
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # 参数含义：
  # GPU_ID 你想使用的GPU
  # NET {vgg16, res50, res101, res152} 选择对应的网络
  # DATASET {pascal_voc, pascal_voc_0712, coco} 在train_faster_rcnn.sh中定义的数据集
  # 使用例子:
  ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
  ./experiments/scripts/train_faster_rcnn.sh 1 coco res101
  ```

3. 可视化训练过程和结果
  ```Shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
  ```

4. 测试训练的模型
  ```Shell
  ./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID 你想使用的GPU
  # NET {vgg16, res50, res101, res152} 选择对应的网络
  # DATASET {pascal_voc, pascal_voc_0712, coco} 在train_faster_rcnn.sh中定义的数据集
  # 使用例子:
  ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
  ./experiments/scripts/test_faster_rcnn.sh 1 coco res101
  ```
