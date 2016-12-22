import numpy as np
import skimage.io as io
import lmdb 
import caffe
import os
from random import shuffle
from tools import SimpleTransformer
import shutil

r_val = 0
g_val = 0
b_val = 0

image_name = './source_images/11537b_x40_01.tif'
image_labels_name = './source_images/11537b_x40_01_labels.tif'

dir_to_save = '/home/pi/Temp/pixel/images_val/'
train_db_name = 'train_lmdb'
test_db_name = 'val_lmdb'


try:
    # pass
    shutil.rmtree(dir_to_save)
except OSError:
    pass

os.mkdir(dir_to_save)

train_part = 0.8
half_image_size = 14
# color testues for labels BGR
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
    map_size=10e12
    db = lmdb.open(db_name, map_size=map_size)
    
    return db


def savePatch(patch, x, y, label):
    patch_name = '_'.join((
        os.path.basename(image_name.replace('.tif', '')), str(x), str(y), str(label)))
    patch_name += '.png'
    io.imsave(os.path.join(dir_to_save, patch_name), patch)


def createPatch(x, y, label):
    print x, y
    patch = im[x-half_image_size:x+half_image_size,
               y-half_image_size:y+half_image_size]
    
    print np.mean(patch[:,:,0]), np.mean(patch[:,:,1]), np.mean(patch[:,:,2])
    st = SimpleTransformer()
    patch = st.preprocess(patch) # no mean
    savePatch(patch, x, y, label)
    print np.mean(patch[:,:,0]), np.mean(patch[:,:,1]), np.mean(patch[:,:,2])
    bgrMean(np.mean(patch[:,:,0]), np.mean(patch[:,:,1]), np.mean(patch[:,:,2]))

    return patch.tobytes()


def bgrMean(b=0, g=0, r=0, calc=False, num=0):
    global g_val, r_val, b_val
    g_val += g
    r_val += r
    b_val += b

    if calc is True:
        return b_val/num, g_val/num, r_val/num


def addToDb(env_train, env_test, x_shape, y_shape):
    env_train = env_train
    env_test = env_test
    counter = 0
    all_tab = []
    train_tab = []
    test_tab = []

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
    test_tab = all_tab[int(len(all_tab)*train_part):]
    print len(train_tab), len(test_tab)

    with env_train.begin(write=True) as txn_train:
        for d in train_tab:
            txn_train.put(d[0].encode('ascii'), d[1].SerializeToString())

    
    with env_test.begin(write=True) as txn_test:
        for d in test_tab:
            txn_test.put(d[0].encode('ascii'), d[1].SerializeToString())
    
    print bgrMean(calc=True, num=counter)


if __name__ == '__main__':

    env_train = createDB(train_db_name)
    env_test = createDB(test_db_name)
    addToDb(env_train, env_test, x_shape, y_shape)
