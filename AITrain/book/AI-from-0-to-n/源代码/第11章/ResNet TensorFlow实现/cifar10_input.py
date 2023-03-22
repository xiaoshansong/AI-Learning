import tarfile
from six.moves import urllib
import sys
import numpy as np
import pickle
import os
import cv2

data_dir = 'cifar10_data'
full_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_'
vali_dir = 'cifar10_data/cifar-10-batches-py/test_batch'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10

TRAIN_RANDOM_LABEL = False # 训练数据使用随机标签？
VALI_RANDOM_LABEL = False # 使用随机标签进行验证？

NUM_TRAIN_BATCH = 5 # 您想要读取多少批文件，从0到5
EPOCH_SIZE = 10000 * NUM_TRAIN_BATCH


def maybe_download_and_extract():
    '''
    将自动下载并提取cifar10数据

    return: 无
    '''
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size)
                                                             / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _read_one_batch(path, is_random_label):
    '''
     训练数据总共包含五个数据批次。 验证数据只有一个批次。 此函数获取一批数据的目录并返回图像和
     相应的标签为numpy数组

    :param path: 一批数据的目录
    :param is_random_label: 是否使用随机标签
    :return: image numpy 数组和 label numpy 数组
    '''
    fo = open(path, 'rb')
    dicts = pickle.load(fo,encoding = 'bytes')
    fo.close()

    data = dicts[b'data']
    if is_random_label is False:
        label = np.array(dicts[b'labels'])
    else:
        labels = np.random.randint(low=0, high=10, size=10000)
        label = np.array(labels)
    return data, label


def read_in_all_images(address_list, shuffle=True, is_random_label = False):
    """
    此函数读取所有训练或验证数据，如果需要，将其打乱，然后返回图像和相应的标签为numpy数组

    :param address_list: pickle文件的路径列表
    :return: 连接numpy数据和标签数组。 数据在4D数组中： [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])

    for address in address_list:
        print ('Reading images from ' + address)
        batch_data, batch_label = _read_one_batch(address, is_random_label)
        # 默认情况下沿轴0连接
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    # 这种重组形状非常重要。 不要改变
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))


    if shuffle is True:
        print ('Shuffling')
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label

def reshape_data(data):
    data = data.reshape((10, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((10, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    return data

def horizontal_flip(image, axis):
    '''
    以50％的可能性翻转图像

    :param image: 表示图像的3维numpy数组
    :param axis: 0表示垂直翻转，1表示水平翻转
    :return: 翻转后的3D图像
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image


def whitening_image(image_np):
    '''
    执行per_image_whitening
    :param image_np:表示一批图像的4D numpy数组
    :return: 白化后的图像numpy数组
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):
    '''
    辅助随机裁剪并随机翻转一批图像
    :param padding_size: 每边添加了多少个0填充层
    :param batch_data:一个4D批处理数组
    :return: 随机裁剪和翻转图像
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def prepare_train_data(padding_size):
    '''
    将所有训练数据读入numpy数组，并在图像的每一侧添加padding_size的0填充层
    :param padding_size:  每侧添加多少0填充层
    :return: 所有训练数据和相应的标签
    '''
    path_list = []
    for i in range(1, NUM_TRAIN_BATCH+1):
        path_list.append(full_data_dir + str(i))
    data, label = read_in_all_images(path_list, is_random_label=TRAIN_RANDOM_LABEL)
    
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    
    return data, label



def read_validation_data():
    '''
    读入验证数据。 同时白化
    :return: 验证图像数据为4D numpy数组。 验证标签为1D numpy数组
    '''
    validation_array, validation_labels = read_in_all_images([vali_dir],
                                                       is_random_label=VALI_RANDOM_LABEL)
    validation_array = whitening_image(validation_array)

    return validation_array, validation_labels



