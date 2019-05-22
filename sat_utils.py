from PIL import Image
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pickle
from skimage.morphology import disk, dilation,erosion, opening



def load_map_tiff(filename):
    with rasterio.open(filename) as ds:
        mask = ds.read()
        mask = mask.astype(np.float32)
        return mask


# Load Pickle file
def read_pickle_data(filename):
    # Open and read codemap
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def crop_img(img, x, y, width, height, channelsFirst=True):
    if channelsFirst:
        # channels, x, y
        return img[:, x:x + width, y:y + height]
    else:
        # x, y, channels
        return img[x:x + width, y:y + height, :]


def get_rgbi(in_img, channelsFirst=True):
    if channelsFirst:
        rgbi = np.stack((in_img[4, :, :], in_img[2, :, :], in_img[1, :, :], in_img[6, :, :]))
    else:
        rgbi = np.dstack((in_img[:, :, 4], in_img[:, :, 2], in_img[:, :, 1], in_img[:, :, 6]))
    return rgbi


def get_rgb(in_img, channelsFirst=True):
    if channelsFirst:
        rgbi = np.stack((in_img[4, :, :], in_img[2, :, :], in_img[1, :, :]))
    else:
        rgbi = np.dstack((in_img[:, :, 4], in_img[:, :, 2], in_img[:, :, 1]))
    return rgbi


# Feature scaling functions
# https://en.wikipedia.org/wiki/Feature_scaling
def img_standardization(in_img):
    return (in_img - np.mean(in_img)) / np.var(in_img)


def img_mean_norm(in_img):
    return (in_img - np.mean(in_img)) / (np.max(in_img) - np.min(in_img))


def img_minmax_norm(image):
    """Min-max normalisation."""
    out = np.zeros_like(image).astype(np.float32)
    if image.sum() == 0:
        return image  # bands  # Probably wrong - error in supplied file.

    for jj in range(image.shape[2]):
        cc = image[:, :, jj].min()
        dd = image[:, :, jj].max()

        tt = (image[:, :, jj] - cc) / (dd - cc)
        out[:, :, jj] = tt
    return out.astype(np.float32)


	
    #return (in_img - np.min(in_img)) / (np.max(in_img) - np.min(in_img))


# Crop some blocks from image
def crop_blocks(input, target, size_box=76, display=False, earlyStop=None):
    size_box = 76
    num_hz_box = int(input.shape[1] / size_box)
    num_vt_box = int(input.shape[2] / size_box)
    cnt_img = 0

    dict_input = {}
    dict_output = {}

    for box_h in range(num_hz_box):
        for box_v in range(num_vt_box):
            x = box_h * size_box
            y = box_v * size_box
            input_box = crop_img(input, x, y, size_box, size_box)
            target_box = crop_img(target, x, y, size_box, size_box)
            #erode the touching border
            target_box[1,:,:] = erosion(target_box[1, :, :], disk(1))


            dict_input[cnt_img] = input_box
            dict_output[cnt_img] = target_box
            cnt_img += 1

            if cnt_img > earlyStop:
                break

            if display:
                print('img_%s_input.png' % cnt_img)
                input_box_rgb = get_rgb(input_box)
                input_box_rgb = np.moveaxis(input_box_rgb, 0, 2)
                input_box_rgb_norm = img_minmax_norm(input_box_rgb[:, :, 0:3])

                f, axarr = plt.subplots(1, 5)
                axarr[0].imshow(input_box_rgb_norm)
                axarr[1].imshow(target_box[0, :, :])
                axarr[2].imshow(target_box[1, :, :])
                axarr[3].imshow(target_box[2, :, :])
                axarr[4].imshow(target_box[0, :, :] - target_box[2, :, :])
                plt.show()

        if cnt_img > earlyStop:
            break

    return dict_input, dict_output
