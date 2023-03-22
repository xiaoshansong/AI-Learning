from resnet import *
from datetime import datetime
import time
from cifar10_input import *
import pandas as pd
from readimg import *



class Train(object):
    '''
    该对象负责所有训练和验证过程
    '''
    def __init__(self):
        # 设置所有占位符
        self.placeholders()


    def placeholders(self):
        '''
        共有五个占位符。
         image_placeholder和label_placeholder用于训练图像和标签
         vali_image_placeholder和vali_label_placeholder用于验证image和标签
         lr_placeholder用于学习率。 每次培训都要考虑学习率
         实现学习率容易衰减
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])


    def build_train_validation_graph(self):
        '''
        此函数同时构建训练图和验证图。
        
        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # 训练数据和验证数据的记录来自同一图表。 
        #验证数据的推断与训练数据共享所有权重。 
        #这是通过将reuse = True传递给训练图的变量范围来实现的
        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)

        # 以下代码计算训练损失，其由softmax交叉熵和重新插入损耗组成
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)


        # 验证损失
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)



    def train(self):
        '''
        这是主要的训练函数
        '''

        # 第一步，将所有的训练图像和验证图像加载到内存中
        all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
        vali_data, vali_labels = read_validation_data()

        # 构建训练和验证图
        self.build_train_validation_graph()

        # 初始化保护程序以保存检查点。 合并所有summary，以便我们可以通过运行summary_op来运行所有摘要操作。 初始化新会话
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()


        # 如果想读取一个检查点
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print ('Restored from checkpoint...')
        else:
            sess.run(init)

        # 此summary编写器对象有助于在tensorboard上编写摘要
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)


        # 这些列表最后用于保存csv文件
        step_list = []
        train_error_list = []
        val_error_list = []

        print ('Start training...')
        print ('----------------------------')

        for step in range(FLAGS.train_steps):

            train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels,
                                                                        FLAGS.train_batch_size)


            validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data,
                                                           vali_labels, FLAGS.validation_batch_size)

            # 想要在训练前验证一次。 您可以先检查理论验证损失
            if step % FLAGS.report_freq == 0:

                if FLAGS.is_full_validation is True:
                    validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                            top1_error=self.vali_top1_error, vali_data=vali_data,
                                            vali_labels=vali_labels, session=sess,
                                            batch_data=train_batch_data, batch_label=train_batch_labels)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                        simple_value=validation_error_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                else:
                    _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                     self.vali_top1_error,
                                                                 self.vali_loss],
                                                {self.image_placeholder: train_batch_data,
                                                 self.label_placeholder: train_batch_labels,
                                                 self.vali_image_placeholder: validation_batch_data,
                                                 self.vali_label_placeholder: validation_batch_labels,
                                                 self.lr_placeholder: FLAGS.init_lr})

                val_error_list.append(validation_error_value)


            start_time = time.time()

            _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                           self.full_loss, self.train_top1_error],
                                {self.image_placeholder: train_batch_data,
                                  self.label_placeholder: train_batch_labels,
                                  self.vali_image_placeholder: validation_batch_data,
                                  self.vali_label_placeholder: validation_batch_labels,
                                  self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time


            if step % FLAGS.report_freq == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_data,
                                                    self.label_placeholder: train_batch_labels,
                                                    self.vali_image_placeholder: validation_batch_data,
                                                    self.vali_label_placeholder: validation_batch_labels,
                                                    self.lr_placeholder: FLAGS.init_lr})
                summary_writer.add_summary(summary_str, step)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print (format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                                    sec_per_batch))
                print ('Train top1 error = ', train_error_value)
                print ('Validation top1 error = %.4f' % validation_error_value)
                print ('Validation loss = ', validation_loss_value)
                print ('----------------------------')

                step_list.append(step)
                train_error_list.append(train_error_value)



            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print ('Learning rate decayed to ', FLAGS.init_lr)

            # 每10000步保存检查点
            if step % 10000 == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                'validation_error': val_error_list})
                df.to_csv(train_dir + FLAGS.version + '_error.csv')


    def test(self, test_image_array):
        '''
        该函数用于评估测试数据。 请提前完成预先进行

       ：param test_image_array：具有形状的4D numpy数组[num_test_images，img_height，img_width，
        img_depth]
       ：return：具有形状的softmax概率[num_test_images，num_labels]

        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print ('%i test batches in total...' %num_batches)

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print ('Model restored from ', FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print ('%i batches finished!' %step)
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array



    ## Helper functions
    def loss(self, logits, labels):
        '''
        给出logits和真实标签，计算交叉熵损失
         ：param logits：具有形状的2D张量[batch_size，num_labels]
         ：param labels：1D张量与形状[batch_size]
         ：return：损失张量与形状[1]
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


    def top_k_error(self, predictions, labels, k):
        '''
        计算top-k错误
         ：param predictions：具有形状的2D张量[batch_size，num_labels]
         ：param labels：1D张量与形状[batch_size，1]
         ：param k：int
         ：return： tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)


    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        如果要使用随机批次的验证数据来验证而不是使用
         整个验证数据，此功能可帮助您生成该批次
         ：param vali_data：4D numpy数组
         ：param vali_label：1D numpy数组
         ：param vali_batch_size：int
         ：return：4D numpy数组和1D numpy数组
        '''
        offset = np.random.choice(10000 - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        vali_label_batch = vali_label[offset:offset+vali_batch_size]
        return vali_data_batch, vali_label_batch


    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        此功能有助于生成一批训练数据，以及随机裁剪，水平翻转
         并同时白化它们
         ：param train_data：4D numpy数组
         ：param train_labels：1D numpy数组
         ：param train_batch_size：int
         ：return：增强列车批次数据和标签。 4D numpy数组和1D numpy数组
        '''
        offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset+train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)

        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset+FLAGS.train_batch_size]

        return batch_data, batch_label


    def train_operation(self, global_step, total_loss, top1_error):
        '''
        定义训练操作
         ：param global_step：具有形状的张量变量[1]
         ：param total_loss：张量形状[1]
         ：param top1_error：形状张量[1]
         ：return：两个操作。 运行train_op将进行一次优化。 正在运行train_ema_op
         将产生列车误差和列车损失的移动平均值
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


    def validation_op(self, validation_step, top1_error, loss):
        '''
        定义验证操作
         ：param validation_step：张量形状[1]
         ：param top1_error：形状张量[1]
         ：param loss：张量与形状[1]
         ：return：验证操作
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
       对所有10000个valdiation图像运行验证
         ：param loss：张量与形状[1]
         ：param top1_error：形状张量[1]
         ：param session：当前的tensorflow会话
         ：param vali_data：4D numpy数组
         ：param vali_labels：1D numpy数组
         ：param batch_data：4D numpy数组。 培训批次以提供dict并获取权重
         ：param batch_label：1D numpy数组。 训练标签来喂养字典
         ：return：float，float
        '''
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)

    def my_test(self):
        # 从cifart10中读取10张图片
        imgX, imgY, label = load_test_ten()
        print("Read ten images:")

        # 将10张图片可视化并保存
        for i in range(10):
            imgs = imgX[i]
            img0 = imgs[0]
            img1 = imgs[1]
            img2 = imgs[2]

            # 读取每张图片的RGB并融合为一张图片
            i0 = Image.fromarray(img0)#从数据，生成image对象
            i1 = Image.fromarray(img1)
            i2 = Image.fromarray(img2)
            img = Image.merge("RGB",(i0,i1,i2))
            name = "img" + str(i)+".png"

            #文件夹下是RGB融合后的图像
            img.save("./cifar10_images/"+name,"png")

            # 打印出每张图片对应标签
            img_label = imgY[i]
            print(label[img_label])


        # 运行模型开始验证
        print("Begin modle test:")

        # 将测试的图片重塑形状以便输入网络
        data = reshape_data(imgX)

        # 输入网络，获取结果列表
        prediction_array = self.test(data)

        # 打印每张图片的softmax概率,并据此打印出预测的标签并与真实标签进行对比
        for i in range(10):
            img_label = imgY[i]
            pred = prediction_array[i]
            flag = 0
            maxes = 0
            for j in range(10):
                if (pred[j]>maxes):
                    flag = j
                else:
                    pass
            print(label[flag])

            print(pred)




maybe_download_and_extract()
# Initialize the Train object
train = Train()
#训练模型
#train.Train()

#测试模型
#train.my_test()





