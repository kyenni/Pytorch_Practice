import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dir_data = './tif_data'

name_label='train-labels.tif'
name_input='train-volume.tif'

img_label=Image.open(os.path.join(dir_data,name_label))
img_input=Image.open(os.path.join(dir_data,name_input))

ny,nx=img_label.size
nframe=img_label.n_frames

nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train=os.path.join(dir_data, 'train')
dir_save_val=os.path.join(dir_data, 'val')
dir_save_test=os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)
##

for i in range(train_size):
    input_data=Image.open(os.path.join(dir_input,inputs[i]))
    label_data=Image.open(os.path.join(dir_label,labels[i]))

    input_=np.asarray(input_data)
    label_=np.asarray(label_data)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

offsets=train_size
for i in range(val_size):
    input_data=Image.open(os.path.join(dir_input,inputs[i+offsets]))
    label_data=Image.open(os.path.join(dir_label,labels[i+offsets]))

    input_=np.asarray(input_data)
    label_=np.asarray(label_data)
    print(label_.shape)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

offsets=train_size+val_size
for i in range(test_size):
    input_data=Image.open(os.path.join(dir_input,inputs[i+offsets]))
    label_data=Image.open(os.path.join(dir_label,labels[i+offsets]))

    input_=np.asarray(input_data)
    label_=np.asarray(label_data)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)
