# 导入依赖库   
import numpy as np  
import tensorflow as tf   
# 定义数据分类汇总  
def variable_summaries(var):  
    with tf.name_scope('summaries'):  
        tf.summary.scalar('mean',tf.reduce_mean(var))  
        tf.summary.scalar('max',tf.reduce_max(var))  
        tf.summary.scalar('min',tf.reduce_min(var))  
        tf.summary.histogram('histogram',var)  
# 构造数据  
with tf.name_scope('input'):  
    x_data = np.random.rand(100).astype(np.float32) #随机生成100个类型为float32的值  
    variable_summaries(x_data)  
    y_data = x_data*0.1+0.3  #定义方程式y=x_data*A+B  
    variable_summaries(y_data)  
# 建立TensorFlow神经计算结构  
with tf.name_scope('compute_struct'):  
    weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))   
    variable_summaries(weight)  
    biases = tf.Variable(tf.zeros([1]))     
    variable_summaries(biases)  
    y = weight * x_data+biases  
    tf.summary.histogram('y',y)  
    loss = tf.reduce_mean(tf.square(y-y_data))  #判断与正确值的差距  
    variable_summaries(loss)  
# 定义优化器和训练器     
optimizer = tf.train.GradientDescentOptimizer(0.5) #根据差距进行反向传播修正参数  
train=optimizer.minimize(loss) #建立训练器  
# 定义初始化变量操作，创建会话  
init = tf.initialize_all_variables() #初始化TensorFlow训练结构  
sess = tf.Session()  #建立TensorFlow训练会话  
# 汇总所有summary  
merge_op = tf.summary.merge_all()  
train_writer = tf.summary.FileWriter('./logs/train',sess.graph)   #保存log文件路径  
sess.run(init)     #将训练结构装载到会话中  
# 开始训练  
for  step in range(301): #循环训练301次  
     sess.run(train)  #使用训练器根据训练结构进行训练  
     summary = sess.run(merge_op)  
     if  step%20==0:  #每20次打印一次训练结果  
         train_writer.add_summary(summary,step)  
         print(step,sess.run(weight),sess.run(biases))  