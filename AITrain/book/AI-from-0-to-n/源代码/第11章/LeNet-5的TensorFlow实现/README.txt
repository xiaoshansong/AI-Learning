LeNet说明
本例利用tensorflow实现手写体数字实现，使用MNIST数据集进行训练，在MNIST测试集上，准确率可到99.1%

代码文件说明
config.py:		神经网络的超参数配置
leNet.py:		构建LeNet网络
Train.py:		加载MINIST数据并训练
Inference.py:	加载模型完成对图片的识别
UI.py			生成可进行手写文字并预测的界面

数据文件说明
MNIST_data/:	从MNIST上下载的数据集
checkpoint:		保存的模型文件

使用说明
在config.py中进行超参数的配置，运行Train.py可以直接进行训练，也可以直接运行UI.py进行验证
UI中，用鼠标在黑色区域进行手写，点击检测即可进行识别