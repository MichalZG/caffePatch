from skimage import io
import sys

labels = {'1': [0, 0, 0],
          '2': [0, 0, 255],
          '3': [255, 0, 0]}

counter = [0, 0, 0]

half_image_size = 14

if __name__ == '__main__':

    args = sys.argv[1:]
    im = io.imread(args[0])
    x_shape, y_shape, z_shape = im.shape

    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]


    for x in range(half_image_size, x_shape-half_image_size):
        for y in range(half_image_size, y_shape-half_image_size):
            pixel_value = [b[x, y], g[x, y], r[x, y]]

            for label, value in labels.iteritems():
                    if pixel_value == value:
                        counter[int(label)-1] += 1
    print counter

