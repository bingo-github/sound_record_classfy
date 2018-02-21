"""
使用两个GRU模型分类器，对['闭', '开', '动', '收', '叶', '右', '展']七个音进行语音分类
用两个分类器的结果综合判断最终的分类结果
"""

# -*- coding: UTF-8 -*-
import record_sound
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import librosa  # pip install librosa
from tqdm import tqdm  # pip install tqdm
import random

''' 参数设置 '''
# 数据集相关参数
# 验证集所占比重
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
# 音频文件的父目录
tf.flags.DEFINE_string("parent_dir", "audio/", "Parent dir of the audio files.")
# 音频文件的子目录
tf.flags.DEFINE_string("tr_sub_dirs", ['fold1/', 'fold2/', 'fold3/', 'fold4/'], "Sub dir of the audio files.")
# 目标音频的所在子目录
tf.flags.DEFINE_string("des_sub_dirs", ['des/'], "Sub dir of the des audio files.")

# 模型相关参数
# MFCC信号数量
tf.flags.DEFINE_integer("n_inputs", 40, "Number of MFCCs")
# cell个数
tf.flags.DEFINE_string("n_hidden_1", 300, "Number of cells in graph1")
tf.flags.DEFINE_string("n_hidden_2", 250, "Number of cells in graph2")
# 分类数
tf.flags.DEFINE_integer("n_classes", 7, "Number of classes")
# 学习率
tf.flags.DEFINE_integer("lr", 0.005, "Learning rate")
# dropout参数
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")

# 训练相关参数
# 批次大小
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size")
# 迭代周期
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs")
# 多少step测试一次
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps")
# 多少step保存一次模型
tf.flags.DEFINE_integer("checkpoint_every", 250, "Save model after this many steps")

# flags解析
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def get_wav_files(parent_dir, sub_dirs):
    """
    :description: 获得训练用的wav文件路径列表
    :param parent_dir: wav文件的父目录
    :param sub_dirs: wav文件的子目录
    :return: 所有目标文件的文件路径名
    """
    wav_files = []
    for l, sub_dir in enumerate(sub_dirs):  # 获得sub_dirs的索引和值
        wav_path = os.path.join(parent_dir, sub_dir)  # 合并路径
        print('wav_path is\n', wav_path)
        for (dirpath, dirnames, filenames) in os.walk(wav_path):  # 遍历wav_path路径下的文件夹和文件
            for filename in filenames:  # filename为路径下的文件
                if filename.endswith('.wav') or filename.endswith('.WAV'):
                    filename_path = os.sep.join([dirpath, filename])  # 获取文件路径，包含文件名
                    wav_files.append(filename_path)
    return wav_files  # 返回所有目标文件的文件路径名


def extract_features(wav_files):
    """
    :description: 获取文件mfcc特征和对应标签
    :param wav_files: wav文件路径名
    :return: wav文件的mfcc特征及其标签
    """
    inputs = []
    labels = []
    for wav_file in tqdm(wav_files):  # 将wav_files中的值依次赋给wav_file，同事产生进度条；可以看做读取进度条
        # 读入音频文件
        audio, fs = librosa.load(wav_file)  # 加载音频，audio:audio time series; fs:sampling rate of `audio`
        # 获取音频mfcc特征
        # [n_steps, n_inputs]
        mfccs = np.transpose(librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=FLAGS.n_inputs), [1, 0])
        inputs.append(mfccs.tolist())
        # 获取label
    for wav_file in wav_files:
        label = wav_file.split('/')[-1].split('-')[0][1]
        print('label is', label)
        print('type(label) is', type(label))
        labels.append(label)
    return inputs, np.array(labels, dtype=np.int)


def Read_wav_file(parent_dirc, sub_dirc, feature_npy, label_npy):
    """
    :description: 根据wav文件的父目录和子目录，获取得到mfcc特征和标签，并保存为.npy文件
    :param parent_dirc: wav文件父目录
    :param sub_dirc: wav文件子目录
    :param feature_npy: 保存wav文件的mfcc特征
    :param label_npy: 保存wav文件的标签
    :return: wav文件的mfcc特征和标签
    """
    wav_files = get_wav_files(parent_dirc, sub_dirc)
    features, labels = extract_features(wav_files)
    np.save(feature_npy, features)
    np.save(label_npy, labels)
    return features, labels


def Load_wav_file(feature_npy, label_npy):
    """
    :description: 从.npy文件中直接获取wav文件的mfcc特征和标签（这样比重新从wav文件中获取快很多）
    :param feature_npy: wav文件的mfcc特征
    :param label_npy: av文件的标签
    :return: wav文件的mfcc特征和标签
    """
    features = np.load(feature_npy)
    labels = np.load(label_npy)
    return features, labels


# # 从wav源文件中获取特征和标签
# tr_features, tr_labels = Read_wav_file(FLAGS.parent_dir, FLAGS.tr_sub_dirs, 'tr_features.npy', 'tr_labels.npy')

# 从npy文件中获取特征和标签
tr_features, tr_labels = Load_wav_file('tr_features.npy', 'tr_labels.npy')

# 计算最长的step
wav_max_len = max([len(feature) for feature in tr_features])


# print("max_len:", wav_max_len)


def Fill_0_to_max_len(srcToBeFilled, stdFeature):
    """
    :description: 在音频信号特征后面补充0，使之达到标准特征最长特征矩阵的长度
                    对于太长的特征矩阵进行阶段，实现特征矩阵对齐
    :param srcToBeFilled: 将被填充的特征矩阵
    :param stdFeature: 作为参考的标准特征矩阵
    :return: 修改了的特征矩阵
    """
    # 计算最长的step
    max_len = max([len(feature) for feature in stdFeature])
    data = []
    for element in srcToBeFilled:
        while len(element) < max_len:  # 只要小于wav_max_len就补n_inputs个0
            element.append([0] * FLAGS.n_inputs)
        while len(element) > wav_max_len:
            element = element[0:max_len]
        data.append(element)
    return np.array(data)


tr_data = Fill_0_to_max_len(tr_features, tr_features)


# print('new tr_data is', type(tr_data))


def Get_train_and_test_set(data, lebal, dev_percentage):
    """
    :description: 获取训练集和测试集
    :param data: 源特征数据
    :param lebal: 源标签数据
    :param dev_percentage: 测试集百分比（dev_percentage = 测试集数量/数据总量）
    :return: 训练集和测试集的特征及标签 train_x, train_y, test_x, test_y
    """
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    x_shuffled = data[shuffle_indices]  # 将数据和标签都进行洗牌
    y_shuffled = lebal[shuffle_indices]
    # 数据集切分为两部分
    dev_sample_index = -1 * int(dev_percentage * float(len(y_shuffled)))
    train_x, test_x = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    train_y, test_y = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = Get_train_and_test_set(tr_data, tr_labels, FLAGS.dev_sample_percentage)

''' 模型训练 '''
g1 = tf.Graph()  # 建立图1
g2 = tf.Graph()  # 建立图2


def grucell(hidden):
    """
    :description: 建立GRU模型
    :param hidden: 隐层单元数
    :return: GRU模型单元
    """
    cell = tf.contrib.rnn.GRUCell(hidden)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
    return cell


with g1.as_default():
    with tf.variable_scope('net_1'):
        x_1 = tf.placeholder("float", [None, wav_max_len, FLAGS.n_inputs], name='x-input_1')
        y_1 = tf.placeholder("float", [None])
        dropout_1 = tf.placeholder(tf.float32)
        # learning rate
        lr_1 = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False)
        # 定义RNN网络
        # 初始化权制和偏置
        weights_1 = tf.Variable(tf.truncated_normal([FLAGS.n_hidden_1, FLAGS.n_classes], stddev=0.1))
        biases_1 = tf.Variable(tf.constant(0.1, shape=[FLAGS.n_classes]))
        # 多层网络
        num_layers_1 = 2
        cell_1 = tf.contrib.rnn.MultiRNNCell([grucell(FLAGS.n_hidden_1) for _ in range(num_layers_1)])
        outputs_1, final_state_1 = tf.nn.dynamic_rnn(cell_1, x_1, dtype=tf.float32)
        # 预测值
        prediction_1 = tf.nn.softmax(tf.matmul(final_state_1[0], weights_1) + biases_1, name='output_1')
        # labels转one_hot格式
        one_hot_labels_1 = tf.one_hot(indices=tf.cast(y_1, tf.int32), depth=FLAGS.n_classes)
        # loss
        cross_entropy_1 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=prediction_1, labels=one_hot_labels_1))
        # optimizer
        optimizer_1 = tf.train.AdamOptimizer(learning_rate=lr_1).minimize(cross_entropy_1)
        # Evaluate model
        correct_pred_1 = tf.equal(tf.argmax(prediction_1, 1), tf.argmax(one_hot_labels_1, 1))
        accuracy_1 = tf.reduce_mean(tf.cast(correct_pred_1, tf.float32))
        result_index_1 = tf.argmax(prediction_1, 1)

with g2.as_default():
    with tf.variable_scope('net_2'):
        x_2 = tf.placeholder("float", [None, wav_max_len, FLAGS.n_inputs], name='x-input_2')
        y_2 = tf.placeholder("float", [None])
        dropout_2 = tf.placeholder(tf.float32)
        # learning rate
        lr_2 = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False)
        # 定义RNN网络
        # 初始化权制和偏置
        weights_2 = tf.Variable(tf.truncated_normal([FLAGS.n_hidden_2, FLAGS.n_classes], stddev=0.1))
        biases_2 = tf.Variable(tf.constant(0.1, shape=[FLAGS.n_classes]))
        # 多层网络
        num_layers_2 = 3
        cell_2 = tf.contrib.rnn.MultiRNNCell([grucell(FLAGS.n_hidden_2) for _ in range(num_layers_2)])
        outputs_2, final_state_2 = tf.nn.dynamic_rnn(cell_2, x_2, dtype=tf.float32)
        # 预测值
        prediction_2 = tf.nn.softmax(tf.matmul(final_state_2[0], weights_2) + biases_2, name='output_2')
        # labels转one_hot格式
        one_hot_labels_2 = tf.one_hot(indices=tf.cast(y_2, tf.int32), depth=FLAGS.n_classes)
        # loss
        cross_entropy_2 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=prediction_2, labels=one_hot_labels_2))
        # optimizer
        optimizer_2 = tf.train.AdamOptimizer(learning_rate=lr_2).minimize(cross_entropy_2)
        # Evaluate model
        correct_pred_2 = tf.equal(tf.argmax(prediction_2, 1), tf.argmax(one_hot_labels_2, 1))
        accuracy_2 = tf.reduce_mean(tf.cast(correct_pred_2, tf.float32))
        result_index_2 = tf.argmax(prediction_2, 1)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    :description: 生成batch_size大小的数据用于训练
    :param data: 源数据集
    :param batch_size: 一个batch数据的大小
    :param num_epochs: epochs数
    :param shuffle: 是否打乱顺序，默认为是
    :return: 用生成器产生数据
    """
    data = np.array(data)
    data_size = len(data)  # ME: 返回2148，表示2148个音频样本
    # 每个epoch的num_batch         # ME: batch_size表示每一批次取的音频样本个数，本程序中为50
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1  # ME: 每一轮训练所用的批次数
    print("num_batches_per_epoch:", num_batches_per_epoch)
    for epoch in range(num_epochs):
        if shuffle:
            # 将训练数据打乱，洗牌
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]  # 一批次一批次的生产训练数据（输出）


def move_array(arr, dirction='left', step=1):
    """
    :description: 将array进行左右移，产生新的数据
    :param arr: 源数据
    :param dirction: 移动方向，默认向左
    :param step: 移动步长
    :return: 移动后新的数据
    """
    new_arr = np.zeros(len(arr))
    if 'left' == dirction:
        new_arr[0:(len(new_arr) - step - 1)] = arr[step:-1]
        new_arr[len(new_arr) - step - 1] = arr[-1]
        new_arr[len(new_arr) - step:-1] = arr[0:step - 1]
        new_arr[-1] = arr[step - 1]
        # new_list = list[step:-1]+[list[-1]]+list[0:step]
    elif 'right' == dirction:
        new_arr[0:step - 1] = arr[-step:-1]
        new_arr[step - 1] = arr[-1]
        new_arr[step:-1] = arr[0:-step - 1]
        new_arr[-1] = arr[-step - 1]
    return new_arr


def change_sound(wavfile, new_path, new_dirction='left', move_step=10, save=1):
    """
    :description: 移动数据，从而增加数据样本量
    :param wavfile: 源wav文件
    :param new_path: 新的数据地址
    :param new_dirction: 移动方向，默认为左移
    :param move_step: 移动步长，默认为10
    :param save: 是否保存新数据，默认保存
    :return: 移动后的数据
    """
    # 增加样本量
    audio, fs = librosa.load(wavfile)
    audio[10:-10] = move_array(audio[10:-10], dirction=new_dirction, step=move_step)
    if 1 == save:
        librosa.output.write_wav(new_path, audio, fs)
    return audio


# with g1.as_default():
#     # 初始化变量
#     init_1 = tf.global_variables_initializer()
#     # 定义saver
#     saver_1 = tf.train.Saver()
#     print('out of sess')
#     with tf.Session(graph=g1) as sess_1:
#         sess_1.run(init_1)
#         # saver.restore(sess, 'net_1/my_net.ckpt')
#         print('in sess')
#         batches = batch_iter(list(zip(train_x, train_y)), FLAGS.batch_size, FLAGS.num_epochs)
#         for i, batch in enumerate(batches):
#             i = i + 1
#             x_batch, y_batch = zip(*batch)
#             sess_1.run([optimizer_1], feed_dict={x_1: x_batch, y_1: y_batch, dropout_1: FLAGS.dropout_keep_prob})
#             # 测试
#             if i % FLAGS.evaluate_every == 0:
#                 sess_1.run(tf.assign(lr_1, FLAGS.lr * (0.99 ** (i // FLAGS.evaluate_every))))
#                 learning_rate_1 = sess_1.run(lr_1)
#                 tr_acc_1, _loss_1 = sess_1.run([accuracy_1, cross_entropy_1], feed_dict={x_1: train_x, y_1: train_y, dropout_1: 1.0})
#                 ts_acc_1 = sess_1.run(accuracy_1, feed_dict={x_1: test_x, y_1: test_y, dropout_1: 1.0})
#                 print("Iter {}, loss_1 {:.5f}, tr_acc_1 {:.5f}, ts_acc_1 {:.5f}, lr {:.5f}".format(i, _loss_1, tr_acc_1, ts_acc_1,
#                                                                                              learning_rate_1))
#                 # 保存模型
#                 if ts_acc_1 > 0.95:
#                     path_1 = saver_1.save(sess_1, "sounds_models_1/model", global_step=i)
#                     print("Saved model checkpoint to {}\n".format(path_1))
#                     saver_1.save(sess_1, 'net_1/my_net.ckpt')
#                     break
#
#
# with g2.as_default():
#     # 初始化变量
#     init_2 = tf.global_variables_initializer()
#     # 定义saver
#     saver_2 = tf.train.Saver()
#     print('out of sess')
#     with tf.Session(graph=g2) as sess_2:
#         sess_2.run(init_2)
#         # saver_2.restore(sess, 'net_2/my_net.ckpt')
#         print('in sess')
#         batches = batch_iter(list(zip(train_x, train_y)), FLAGS.batch_size, FLAGS.num_epochs)
#         for i, batch in enumerate(batches):
#             i = i + 1
#             x_batch, y_batch = zip(*batch)
#             sess_2.run([optimizer_2], feed_dict={x_2: x_batch, y_2: y_batch, dropout_2: FLAGS.dropout_keep_prob})
#             # 测试
#             if i % FLAGS.evaluate_every == 0:
#                 sess_2.run(tf.assign(lr_2, FLAGS.lr * (0.99 ** (i // FLAGS.evaluate_every))))
#                 learning_rate_2 = sess_2.run(lr_2)
#                 tr_acc_2, _loss_2 = sess_2.run([accuracy_2, cross_entropy_2], feed_dict={x_2: train_x, y_2: train_y, dropout_2: 1.0})
#                 ts_acc_2 = sess_2.run(accuracy_2, feed_dict={x_2: test_x, y_2: test_y, dropout_2: 1.0})
#                 print("Iter {}, loss_2 {:.5f}, tr_acc_2 {:.5f}, ts_acc_2 {:.5f}, lr {:.5f}".format(i, _loss_2, tr_acc_2, ts_acc_2,
#                                                                                              learning_rate_2))
#                 # 保存模型
#                 if ts_acc_2 > 0.95:
#                     path_2 = saver_2.save(sess_2, "sounds_models_2/model", global_step=i)
#                     print("Saved model checkpoint to {}\n".format(path_2))
#                     saver_2.save(sess_2, 'net_2/my_net.ckpt')
#                     break


''' 进行识别 '''
voice_list = ['闭', '开', '动', '收', '叶', '右', '展']
while True:
    input_char = input('start record sound?(Y/N)\n')
    if input_char == 'y' or input_char == 'Y':
        record_sound.Record_sound(2, r'.\\audio\\des\\1-000.wav')
        des_features, des_labels = Read_wav_file(FLAGS.parent_dir, FLAGS.des_sub_dirs, 'des_features.npy',
                                                 'des_labels.npy')
        des_data = Fill_0_to_max_len(des_features, tr_features)
        des_x = des_data
        des_y = des_labels

        predict_index = np.array([0] * FLAGS.n_classes)
        with g1.as_default():
            with tf.Session(graph=g1) as sess_1:
                saver_1 = tf.train.Saver()
                saver_1.restore(sess_1, 'new_net_1/my_net.ckpt')
                max_index_1, softmax_predit_1 = sess_1.run([result_index_1, prediction_1],
                                                           feed_dict={x_1: des_x, y_1: des_y, dropout_1: 1.0})
                # print('prediction_1为\n', softmax_predit_1)
                if max(softmax_predit_1[0]) > 0.8:
                    the_index = np.where(softmax_predit_1[0] > 0.8)[0]
                    predict_index[the_index[0]] += 1
                    # predict_index[softmax_predit_1.index(max(softmax_predit_1[0]))] += 1
                    print('识别结果1为', voice_list[np.where(softmax_predit_1[0] > 0.8)[0][0]])
                else:
                    print('识别结果1为 其他')

        with g2.as_default():
            with tf.Session(graph=g2) as sess_2:
                saver_2 = tf.train.Saver()
                saver_2.restore(sess_2, 'new_net_2/my_net.ckpt')
                max_index_2, softmax_predit_2 = sess_2.run([result_index_2, prediction_2],
                                                           feed_dict={x_2: des_x, y_2: des_y, dropout_2: 1.0})
                # print('prediction_2为\n', softmax_predit_2)
                if max(softmax_predit_2[0]) / sum(softmax_predit_2[0]) > 0.9:
                    the_index = np.where(softmax_predit_2[0] > 0.8)[0][0]
                    predict_index[the_index] += 1
                    # predict_index[softmax_predit_2.index(max(softmax_predit_2[0]))] += 1
                    print('识别结果2为', voice_list[np.where(softmax_predit_2[0] > 0.8)[0][0]])
                else:
                    print('识别结果2为 其他')

        if max(predict_index) == 2:
            print('最终判断为', voice_list[np.where(predict_index == 2)[0]])
        elif max(predict_index) == 1 and sum(predict_index) == 0:
            print('最终判断为', voice_list[np.where(predict_index == 1)[0]])
        else:
            print('最终判断为 其他')
