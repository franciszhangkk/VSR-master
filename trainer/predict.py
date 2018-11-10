from keras.models import load_model
import os
from model import MCFRVSR_model5
from model import LocalizationNetwork,SpatialTransformer
from model import FRVSR_model_2
from keras.optimizers import Adam
import imageio
import numpy as np
import scipy
import random
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard
from keras import backend


output_size = (256, 256)
input_size = (128, 128)


def read_file_names():
    file_names = []
    for (root, dirs, files) in os.walk(r'C:\Users\Francis\Desktop\FRVSR-master\test_data'):
        for dir in dirs:
            file_names.append(os.path.join(root, dir))
    print("Loaded ", len(file_names))
    return file_names, len(file_names)


def read_image_names(file):
    image_names = []
    for root, _, files in os.walk(file):
        for filename in files:
            image_names.append(os.path.join(root, filename))
    # print("Loaded ", len(image_names))
    return image_names

def read_image(filename):
    high_res = imageio.imread(filename).astype(float)
    low_res = scipy.misc.imresize(high_res, size=50, interp='bilinear')
    return high_res, low_res

def read_input(batch_size,filenames):
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

    for i in range(batch_size):
        image_names = read_image_names(filenames[i])
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
    inp = [np.asarray(batch_list1), np.asarray(batch_list2), np.asarray(batch_list3), np.asarray(batch_list4), np.asarray(batch_list5), low_res_black, high_res_black, flow_pre]
    return inp



if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    KTF.set_session(tf.Session(config=config))


    superes_net = load_model('model.h5', custom_objects={"tf": tf,
                                                         "LocalizationNetwork":LocalizationNetwork,
                                                         "SpatialTransformer": SpatialTransformer,
                                                         })
    filenames, batch_size = read_file_names()
    input_batch = read_input(batch_size, filenames)
    pred = superes_net.predict_on_batch(input_batch)
    for i in range(len(pred)):
        for j in range (len(pred[i])):
            output_path = r"C:\Users\Francis\Desktop\project_document\SuperRezil-master\trainer\output\{j_}_{i_}.png".format(i_=i,j_ = j)
            imageio.imwrite(output_path, pred[i][j])

