ResNet说明
本例是使用

代码文件说明
cifar10_input.py：		对数据集的读取和预处理
cifar10_train.py：		用于定义全局需要使用的参数
hyper_parameters.py：	ResNet的整体结构
resnet.py：				训练模型的主要文件,其中训练类中也包含了测试程序

数据文件说明
data_batch_1~5：			训练所用数据集
test_batch：				测试所用数据集

使用说明
在cifar10_train.py的最后，可以选择运行train.Train()，来训练模型；
也可以选择运行train.my_test()，来测试已经训练完成的模型。