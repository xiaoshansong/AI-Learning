VggNet说明
本例是使用VGG19模型构建的图像识别，采用预训练并保存为Numpy张量的模型文件。可在此基础上继续训练。

代码文件说明
vgg19.py:		构建VGG19网络，实现一个vgg19的类
utils.py:		工具类，实现图片的读取和打印最终的结果
test_vgg19.py：	测试VGG19类，读取测试图片并进行识别

数据文件说明
vgg19.npy:		预训练的模型文件
synset.txt:		储存所有的标签的文件
test_data/:		存放测试图片

使用说明
直接运行test_vgg19.py，会对test_data中的tiger.jpg进行识别并打印出TOP1和TOP5。
然后将此图片标注为老虎加入模型训练，再次进行识别。并保存新的模型为test-save.npy
需要更改图片进行识别，则需要更改加载图片的名称，如不需要加载测试图片进行训练可删除
test_vgg19.py的35-44行