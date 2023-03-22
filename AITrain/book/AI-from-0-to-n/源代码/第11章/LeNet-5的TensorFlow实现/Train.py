
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import config as cfg
import os
import lenet
from lenet import Lenet


def main():
    # 从指定路径加载训练数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 开启TensorFlow会话
    sess = tf.Session()

    # 设置超参数
    batch_size = cfg.BATCH_SIZE
    parameter_path = cfg.PARAMETER_FILE
    lenet = Lenet()
    max_iter = cfg.MAX_ITER

    # 加载已保存的模型参数文件，如果不存在则调用初始化函数生成初始网络
    saver = tf.train.Saver()
    if os.path.exists(parameter_path):
        saver.restore(parameter_path)
    else:
        sess.run(tf.initialize_all_variables())

    # 迭代训练max_iter次，每次抽取50个样本进行训练
    # 每100次打印出当前数据的精度
    # 训练完成后保存模型参数
    for i in range(max_iter):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(lenet.train_accuracy,feed_dict={
                lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]
            })
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(lenet.train_op,feed_dict={lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]})
    save_path = saver.save(sess, parameter_path)

if __name__ == '__main__':
    main()


