# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from matplotlib import colors
import os
# display plots in this notebook
cmap = colors.ListedColormap(['black','red','blue'])
bounds=[0.5, 1.5, 2.5, 3.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

counter = [0, 0, 0]

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/home/pi/Programs/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')



import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

model_folder = '/home/pi/Programs/digits/digits/jobs/20161221-203937-682b'
dataset_folder = '/home/pi/Programs/digits/digits/jobs/20161221-203828-b4a9'

caffe.set_mode_gpu()
model_def = model_folder + '/deploy.prototxt' 
model_weights = model_folder + '/snapshot_iter_6412.caffemodel' 

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

#load mean image file and convert it to a .npy file--------------------------------
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(dataset_folder + '/mean.binaryproto',"rb").read()
blob.ParseFromString(data)
nparray = caffe.io.blobproto_to_array(blob)
os.remove('/home/pi/Temp/pixel/imgmean.npy')
f = file('/home/pi/Temp/pixel/imgmean.npy',"wb")
np.save(f,nparray)

f.close()


# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu1 = np.load('/home/pi/Temp/pixel/imgmean.npy')
mu1 = mu1.squeeze()
mu = mu1.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)
print 'mean shape: ',mu1.shape
print 'data shape: ',net.blobs['data'].data.shape

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# set the size of the input (we can skip this if we're happy

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
# transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
# net.blobs['data'].reshape(50,        # batch size
#                         3,         # 3-channel (BGR) images
#                        28, 28)  # image size is 227x227

#load image

def pred(image):
    transformed_image = transformer.preprocess('data', image)
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()
    output_prob = output['softmax']  # the output probability vector for the first image in the batch
    counter[output_prob.argmax()] += 1
    print output
    # print 'predicted class is:', output_prob.argmax() + 1
    print counter
    return output_prob.argmax() + 1

if __name__ == "__main__":
    image_name = '/home/pi/Programs/python-programs/caffePatch' +\
            '/source_images/11537b_x40_03.tif' 
    im = io.imread(image_name)
    # im = im[300:600, 200:600]
    x_shape, y_shape, z_shape = im.shape
    patch_test = '/home/pi/Temp/pixel/images_val/1/' + '11537b_x40_01_37_355_1.png'
    patch_test = io.imread(patch_test)
    half_patch_size = 14
    pred_map = np.zeros_like(im[:,:,0])
    for x in range(half_patch_size, x_shape-half_patch_size):
        for y in range(half_patch_size, y_shape-half_patch_size):
            patch = im[x-half_patch_size:x+half_patch_size,
                       y-half_patch_size:y+half_patch_size]
            pred_class = pred(patch)
            # pred_class = pred(patch_test)
            pred_map[x][y] = int(pred_class)
    plt.pcolor(pred_map, cmap=cmap, norm=norm)
    plt.show()
