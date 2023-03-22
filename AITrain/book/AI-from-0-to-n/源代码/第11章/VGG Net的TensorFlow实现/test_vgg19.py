import tensorflow as tf

import vgg19 as vgg19
import utils

# 加载测试数据，这里用一张老虎的图片作为测数据
img1 = utils.load_image("./test_data/tiger.jpeg")
img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  

# 将测试数据转换为1维张量
batch1 = img1.reshape((1, 224, 224, 3))

# 指定在cpu:0上运行
with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [1, 1000])
    train_mode = tf.placeholder(tf.bool)

    # 加载参数文件
    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    # 打印出用到的参数总数
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # 首先在分类器上运行测试图片，将输出的张量在synset.txt中找到对应的标签并输出
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.print_prob(prob[0], './synset.txt')

    # 加载测试图片进行一次训练
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    # 再次在分类器上运行测试图片，对比两次分类的准确程度
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.print_prob(prob[0], './synset.txt')

    # 保存这次训练后的参数文件为test-save.npy
    vgg.save_npy(sess, './test-save.npy')
