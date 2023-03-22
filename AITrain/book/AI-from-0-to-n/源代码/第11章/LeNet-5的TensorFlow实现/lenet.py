import tensorflow as tf
import tensorflow.contrib.slim as slim
import config as cfg

class Lenet:
    def __init__(self):
        """
        初始化LeNet网络
        """
        # 设置网络输入的图片为二维张量，数据的类型为float32，行数不固定，列固定为784
        self.raw_input_image = tf.placeholder(tf.float32, [None, 784])

        # 改变网络输入张量的形状为四维，-1表示数值不固定
        self.input_images = tf.reshape(self.raw_input_image, [-1, 28, 28, 1])

        # 设置网络输入标签为二维张量，数据类型为float，行数不固定，列固定为10
        self.raw_input_label = tf.placeholder("float", [None, 10])

        # 改变标签的数据类型为int32
        self.input_labels = tf.cast(self.raw_input_label,tf.int32)

        # 设置网络的随机失活概率
        self.dropout = cfg.KEEP_PROB

        # 构建两个网络
        # train_digits为训练网络，开启dropout
        # pred_digits为预测网络，关闭dropout
        with tf.variable_scope("Lenet") as scope:
            self.train_digits = self.construct_net(True)
            scope.reuse_variables()
            self.pred_digits = self.construct_net(False)

        # 获取网络的预测数值
        self.prediction = tf.argmax(self.pred_digits, 1)

        # 获取网络的预测数值与标签的匹配程度
        self.correct_prediction = tf.equal(tf.argmax(self.pred_digits, 1), tf.argmax(self.input_labels, 1))
        
        # 将匹配程度转换为float类型，表示为精度
        self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        # 计算train_digits与labels之间的系数softmax交叉熵，定义为loss
        self.loss = slim.losses.softmax_cross_entropy(self.train_digits, self.input_labels)

        # 设置学习速率
        self.lr = cfg.LEARNING_RATE
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



    def construct_net(self,is_trained = True):
        """
        接收is_trained参数判断是否开启dropout
        用slim构建LeNet模型
        第一、三、五层为卷积层、第二、四层为池化层
        接下来对第五层扁平化，然后接入全连接
        然后进行随机失活防止过拟合，最后再次接入全连接层
        最后返回构建的网络
        """
        with slim.arg_scope([slim.conv2d], padding='VALID',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(self.input_images,6,[5,5],1,padding='SAME',scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net,16,[5,5],1,scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.conv2d(net,120,[5,5],1,scope='conv5')
            net = slim.flatten(net, scope='flat6')
            net = slim.fully_connected(net, 84, scope='fc7')
            net = slim.dropout(net, self.dropout,is_training=is_trained, scope='dropout8')
            digits = slim.fully_connected(net, 10, scope='fc9')
        return digits