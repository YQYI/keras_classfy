#!/usr/bin/env python
# coding=utf-8
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50
import os
import numpy as np
import cv2
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #设置需要使用的GPU的编号
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 #设置使用GPU容量占GPU总容量的比例
sess = tf.Session(config=config)
KTF.set_session(sess)


def create_model(w, h, num):
    input_tensor = Input(shape = (w, h, 3))
    base_model = ResNet50(weights = 'imagenet',include_top = False, input_tensor = input_tensor)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    prediction = Dense(num, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def create_path_list(total_dir):
    # create path list
    path_list = []
    label = 0
    for i in os.listdir(total_dir):
        for j in os.listdir(os.path.join(total_dir, i)):
            path_list.append([os.path.join(total_dir, i, j), label])
        label += 1
    np.random.shuffle(path_list)
    print(path_list)
    return path_list

def data_generator(path_list, batch_size, w, h, num):
    # generator
    i = 0
    while True:
        image_data = []
        label_data = []
        for j in range(batch_size):
            image = cv2.imread(path_list[i][0])
            image = cv2.resize(image, (w, h))
            label = path_list[i][1]
            image_data.append(image)
            label_data.append(label)
            i = (i + 1)%len(path_list)
        image_data = np.array(image_data)
        label_data = np.array(label_data)
        label_data = np_utils.to_categorical(label_data, num)
        yield image_data, label_data

def train():
    # parameter
    w = 224
    h = 224
    batch_size = 2
    num = 7
    epoch = 200
    total_dir = "root"
    # create model
    model = create_model(w, h, num)
    # get data path list and spilt into train and validation
    path_list = create_path_list(total_dir)
    train_list = path_list[0:int(0.9*len(path_list))]
    val_list = path_list[len(train_list):]

    # 保存模型回调函数，保存最优模型
    checkpointer = ModelCheckpoint(filepath="checkpoint-{epoch:02d}e-val_acc_{val_acc:.2f}.hdf5",
            monitor = "val_acc",
            save_best_only=True, verbose=1,  period=5)
    # 训练模型，并返回训练过程中的中间数据
    history = model.fit_generator(
            data_generator(train_list, batch_size, w, h, num),
            validation_data = data_generator(val_list, batch_size, w, h, num),
            validation_steps = len(val_list)//batch_size,
            epochs = epoch,
            steps_per_epoch = len(train_list)//batch_size,
            verbose = 1,
            callbacks = [checkpointer])



if __name__ == "__main__":
    train()


