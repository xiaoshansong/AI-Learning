# 导入依赖库  
import numpy as np  
import tensorflow as tf  
import matplotlib.pyplot as plt  
# 准备数据  
x_data = np.linspace(-2, 0, 300)  
noise = np.random.randn(*x_data.shape) * 0.1  #加入噪声  
y_data = np.square(x_data) + 2 * x_data + 1 + noise  # y=x^2+2x+1  
# 显示模拟数据  
plt.plot(x_data, y_data, 'ro', label='original data')  
plt.legend()  
plt.show()  
# 创建模型  
X = tf.placeholder('float')  # 占位符  
Y = tf.placeholder('float')  # 占位符  
w1 = tf.Variable(tf.random_normal([1]), name='weight1')  # 模型参数  
w2 = tf.Variable(tf.random_normal([1]), name='weight2')  # 模型参数  
b = tf.Variable(tf.zeros([1]), name='bias')  # 模型参数  
# 前向结构  
z = tf.multiply(np.square(X), w2) + tf.multiply(X, w1) + b  
# 反向优化  
cost = tf.reduce_mean(tf.square(Y-z))  
learning_rate = 0.02  
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  
# 训练模型  
train_r = 100  # 训练总次数  
display_r = 10  # 显示间隔  
saver = tf.train.Saver()   # 保存操作
with tf.Session() as sess:  # 创建会话  
    sess.run(tf.global_variables_initializer())  
    for r in range(train_r):  
        for(x, y) in zip(x_data, y_data): # 模型数据输入  
            sess.run(optimizer, feed_dict={X: x, Y: y})  
        if r % display_r == 0:  
            print('round:', r+1,'  cost:', sess.run(cost, feed_dict={X: x_data, Y: y_data}))  
            print('  w2:', sess.run(w2), '  w1:',sess.run(w1), '  b:',sess.run(b))  
    # 保存模型
    saver.save(sess,'D:/python/checkpoint/test.ckpt') 
    # 结果可视化  
    plt.plot(x_data, y_data, 'ro', label='original data')  
    plt.plot(x_data,  sess.run(w2) * np.square(x_data) + sess.run(w1) * x_data  + sess.run(b), label='fittedcurve')  
    plt.legend()  
    plt.show()  