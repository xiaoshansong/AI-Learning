
"""
设置模型的超参数
 

KEEP_PROB:			网络随机失活的概率
LEARNING_RATE:		学习的速率，即梯度下降的速率
BATCH_SIZE:			一次训练所选取的样本数
PARAMETER_FILE:		模型参数保存的路径
MAX_ITER:			最大迭代次数
"""

KEEP_PROB = 0.5
LEARNING_RATE = 1e-5
BATCH_SIZE =50
PARAMETER_FILE = "checkpoint/variable.ckpt"
MAX_ITER = 50000