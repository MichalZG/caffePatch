import numpy as np
import skimage.io as io
import lmdb 
import caffe
import os
from random import shuffle

image_name = '11537b_x40_01.tif'
image_labels_name = '11537b_x40_01_labels.tif'

dir_to_save = '/home/pi/Temp/pixel/images/'
train_db_name = 'train_lmdb'
val_db_name = 'val_lmdb'

train_part = 0.7
half_image_size = 25
# color values for labels BGR
labels = {'1': [0, 0, 0],
          '2': [0, 0, 255],
          '3': [255, 0, 0]}

im = io.imread(image_name)
im_l = io.imread(image_labels_name)

r = im_l[:,:,0]
g = im_l[:,:,1]
b = im_l[:,:,2]


x_shape, y_shape, z_shape = im_l.shape


def createDB(db_name):
    map_size=104857600000
    db = lmdb.open(db_name, map_size=map_size)
    
    return db


def savePatch(patch, x, y, label):
    patch_name = '_'.join((str(x), str(y), str(label)))
    patch_name += '.tif'
    io.imsave(os.path.join(dir_to_save, patch_name), patch)


def createPatch(x, y, label):
    print x, y
    patch = im[x-half_image_size:x+half_image_size,
               y-half_image_size:y+half_image_size]
    savePatch(patch, x, y, label)

    return patch.tobytes()


def addToDb(env_train, env_val, x_shape, y_shape):
    env_train = env_train
    env_val = env_val
    counter = 0
    all_tab = []
    train_tab = []
    val_tab = []

    for x in range(half_image_size, x_shape-half_image_size):
        for y in range(half_image_size, y_shape-half_image_size):
            pixel_value = [b[x, y], g[x, y], r[x, y]]

            for label, value in labels.iteritems():
                if pixel_value == value:
                    counter += 1
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.channels = 3
                    datum.height = half_image_size * 2
                    datum.width = half_image_size * 2
                    datum.data = createPatch(x, y, int(label))
                    datum.label = int(label)
                    str_id = '{:08}'.format(counter)
                    all_tab.append([str_id, datum])
    shuffle(all_tab)
    train_tab = all_tab[:int(len(all_tab)*train_part)]
    val_tab = all_tab[int(len(all_tab)*train_part):]
    print len(train_tab), len(val_tab)

    with env_train.begin(write=True) as txn_train:
        for d in train_tab:
            txn_train.put(d[0].encode('ascii'), d[1].SerializeToString())

    
    with env_val.begin(write=True) as txn_val:
        for d in val_tab:
            txn_val.put(d[0].encode('ascii'), d[1].SerializeToString())



if __name__ == '__main__':
    env_train = createDB(train_db_name)
    env_val = createDB(val_db_name)
    addToDb(env_train, env_val, x_shape, y_shape)
