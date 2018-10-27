from model import MCFRVSR_model5
from model import FRVSR_model_2
from keras.optimizers import Adam
import imageio
import numpy as np
import scipy
import os
import random
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard

output_size = (256, 256)
input_size = (128, 128)

batch_size = 2


def read_file_names():
    file_names = []
    for (root, dirs, files) in os.walk('data'):
        for dir in dirs:
            file_names.append(os.path.join(root, dir))
    print("Loaded ", len(file_names))
    return file_names


def read_image_names(file):
    image_names = []
    for root, _, files in os.walk(file):
        for filename in files:
            image_names.append(os.path.join(root, filename))
    print("Loaded ", len(image_names))
    return image_names


def read_image(filename):
    high_res = imageio.imread(filename).astype(float)
    low_res = scipy.misc.imresize(high_res, size=50, interp='bilinear')
    return high_res, low_res


def generator1(filenames):
    batch_list = [None] * batch_size
    label_list = [None] * batch_size
    low_res_black = np.zeros((batch_size, input_size[1], input_size[0], 3))
    high_res_black = np.zeros((batch_size, output_size[1], output_size[0], 3))
    while True:
        for i in range(batch_size):
            index = random.choice(range(len(filenames)))
            high_res, low_res = read_image(filenames[index])
            batch_list[i] = low_res
            label_list[i] = high_res
        result, l = [np.asarray(batch_list), low_res_black, high_res_black], np.asarray(label_list)
        yield result, l


def generator(filenames):
    batch_list1 = [None] * batch_size
    label_list1 = [None] * batch_size
    batch_list2 = [None] * batch_size
    label_list2 = [None] * batch_size
    low_res_black = np.zeros((batch_size, input_size[1], input_size[0], 3))
    high_res_black = np.zeros((batch_size, output_size[1], output_size[0], 3))
    while True:
        for i in range(batch_size):

            index = random.choice(range(len(filenames)))
            image_names = read_image_names(filenames[index])
            low_res_list = []
            high_res_list = []
            for j in range(0, len(image_names)):
                # print(image_names,'im len')
                high_res, low_res = read_image(image_names[j])
                low_res_list.append(low_res)
                high_res_list.append(high_res)
            batch_list1[i] = low_res_list[0]
            label_list1[i] = high_res_list[0]
            batch_list2[i] = low_res_list[1]
            label_list2[i] = high_res_list[1]

        result, l = [np.asarray(batch_list1), np.asarray(batch_list2), low_res_black, high_res_black], [np.asarray(label_list1),np.asarray(label_list1)]
        yield result, l


def generator2(filenames):
    batch_list1 = [None] * batch_size
    label_list1 = [None] * batch_size
    batch_list2 = [None] * batch_size
    label_list2 = [None] * batch_size
    batch_list3 = [None] * batch_size
    label_list3 = [None] * batch_size
    batch_list4 = [None] * batch_size
    label_list4 = [None] * batch_size
    batch_list5 = [None] * batch_size
    label_list5 = [None] * batch_size
    flow_pre = np.zeros((batch_size, input_size[1], input_size[0], 2))
    low_res_black = np.zeros((batch_size, input_size[1], input_size[0], 3))
    high_res_black = np.zeros((batch_size, output_size[1], output_size[0], 3))

    while True:
        for i in range(batch_size):
            index = random.choice(range(len(filenames)))
            image_names = read_image_names(filenames[index])
            low_res_list = []
            high_res_list = []
            for j in range(0, len(image_names)):
                # print(image_names,'im len')
                high_res, low_res = read_image(image_names[j])
                low_res_list.append(low_res)
                high_res_list.append(high_res)
            batch_list1[i] = low_res_list[0]
            label_list1[i] = high_res_list[0]
            batch_list2[i] = low_res_list[1]
            label_list2[i] = high_res_list[1]
            batch_list3[i] = low_res_list[2]
            label_list3[i] = high_res_list[2]
            batch_list4[i] = low_res_list[3]
            label_list4[i] = high_res_list[3]
            batch_list5[i] = low_res_list[4]
            label_list5[i] = high_res_list[4]

        result, l = [np.asarray(batch_list1), np.asarray(batch_list2), np.asarray(batch_list3), np.asarray(batch_list4), np.asarray(batch_list5), low_res_black, high_res_black, flow_pre], [
            np.asarray(label_list1), np.asarray(label_list2), np.asarray(label_list3), np.asarray(label_list4), np.asarray(label_list5)]
        yield result, l


if __name__ == '__main__':
    KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
    adam_lr = 0.0004

    tensorboard = TensorBoard(log_dir="./logs", write_graph=True, write_images=True)

    # low_res, high_res = read_input()
    filenames = read_file_names()
    superes_net = MCFRVSR_model5((input_size[1], input_size[0], 3), (output_size[1], output_size[0], 3), (input_size[1], input_size[0], 2))
    # create the FRVSR object
    superes_net.model.compile(
        optimizer=Adam(lr=adam_lr),
        loss='mean_squared_error',
        loss_weights=[1., 1., 1., 1., 1.])

    # low_res_input = np.asarray(low_res)
    # high_res_input = np.asarray(high_res)
    # low_res_black = np.zeros(low_res_input.shape)
    # high_res_black = np.zeros(high_res_input.shape)

    # superes_net.model.fit([low_res_input, low_res_black, high_res_black], high_res_input, epochs=10, callbacks=[tensorboard], validation_split=0.2)
    superes_net.model.fit_generator(generator2(filenames), epochs=10, samples_per_epoch=32, callbacks=[tensorboard])

    superes_net.model.save("model.h5")

    print("train finish")
    # low_res = np.asarray(low_res)[-1:]
    # high_res = np.asarray(high_res)[-1:]
    # high_res, low_res = read_image(filenames[-1])
    #
    # inp = [np.asarray([low_res]), np.asarray([np.zeros(low_res.shape)]), np.asarray([np.zeros(high_res.shape)])]
    # pred = superes_net.model.predict_on_batch(inp)
    #
    # print(pred.shape)
    #
    # imageio.imwrite("output.png", pred[0])
