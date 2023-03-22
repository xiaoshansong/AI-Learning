import tensorflow as tf
from PIL import Image,ImageOps
import numpy as np
from lenet import Lenet
import config as cfg

class inference:
    def __init__(self):
        """
        构建Lenet网络，设置模型参数文件路径，开启TensorFlow会话
        """
        self.lenet = Lenet()
        self.sess = tf.Session()
        self.parameter_path = cfg.PARAMETER_FILE
        self.saver = tf.train.Saver()

    def predict(self,image):
        """
        接收要测试的图片作为参数，返回预测值
        """
        # 将图片转换为合适的大小进行输入
        img = image.convert('L')
        img = img.resize([28, 28], Image.ANTIALIAS)
        image_input = np.array(img, dtype="float32") / 255
        image_input = np.reshape(image_input, [-1, 784])

        # 读取模型参数并对图片进行预测，返回预测值
        self.saver.restore(self.sess,self.parameter_path)
        predition = self.sess.run(self.lenet.prediction, feed_dict={self.lenet.raw_input_image: image_input})
        return predition




