from PIL import Image
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pickle
from skimage.morphology import disk, dilation,erosion, opening


# Load Pickle file
def read_pickle_data(filename):
    # Open and read codemap
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load_map_tiff(filename):
    with rasterio.open(filename) as ds:
        mask = ds.read()
        mask = mask.astype(np.float32)
        return mask
    
    
def crop_img(img, x, y, width, height, channelsFirst=True):
    if channelsFirst:
        #channels, x, y
        return img[:,x:x+width, y:y+height]
    else:
        #x, y, channels
        return img[x:x+width, y:y+height,:]

    
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
    return (in_img-np.mean(in_img))/np.var(in_img)


def img_mean_norm(in_img):
    return (in_img-np.mean(in_img))/(np.max(in_img)-np.min(in_img))

def img_minmax_norm_torch(in_tensor):
    if len(in_tensor.shape) > 3:
        batch, channels, h, w = in_tensor.size()
    else:
        batch, h, w = in_tensor.size()
        channels = 1
    output = torch.zeros_like(in_tensor)
    
    # Populate output
    if channels > 1:
        for chanel in range(channels):
            # Get min/max values per channel
            min_val = in_tensor[:,chanel,:,:].min()
            max_val = in_tensor[:,chanel,:,:].max()
            output[:,chanel,:,:] = (in_tensor[:,chanel,:,:] - min_val) / (max_val - min_val)
    else:
        # Get min/max values per channel
        min_val = in_tensor.min()
        max_val = in_tensor.max()
        output = (in_tensor - min_val) / (max_val - min_val)
    
    return output


def img_minmax_norm(image, channelsFirst=True):
    """Min-max normalisation."""
    out = np.zeros_like(image).astype(np.float32)
   # if image.sum() == 0:
   #     raise ValueError("Probably wrong - error in supplied file.")
    
    if channelsFirst:
        num_channels = image.shape[0]
    else:
        # Channels are located on the last dimension
        num_channels = image.shape[-1]

    # For each channel
    for channel in range(num_channels):
        if channelsFirst:
            min_val = image[channel, :, :].min()
            max_val = image[channel, :, :].max()

            norm_channel = (image[channel, :, :] - min_val) / (max_val - min_val)
            out[channel, :, :] = norm_channel
        else:
            min_val = image[:, :, channel].min()
            max_val = image[:, :, channel].max()

            norm_channel = (image[:, :, channel] - min_val) / (max_val - min_val)
            out[:, :, channel] = norm_channel
    
    return out.astype(np.float32)


# Crop some blocks from image
def crop_blocks(input, target, size_box=76, display=False, earlyStop=None, disk_size=1, channelsFirst=True, do_preprocess=False, offset=0):
    size_box = size_box
    if channelsFirst:
        num_hz_box = int(input.shape[1] / size_box)
        num_vt_box = int(input.shape[2] / size_box)
    else:
        num_hz_box = int(input.shape[0] / size_box)
        num_vt_box = int(input.shape[1] / size_box)
        
    cnt_img = 0

    dict_input = {}
    dict_output = {}

    for box_h in range(num_hz_box):
        for box_v in range(num_vt_box):
            x = box_h * size_box
            y = box_v * size_box
            input_box = crop_img(input, x, y, size_box, size_box, channelsFirst)
            if do_preprocess:
                # Doing the min_max norm on the cropped data
                input_box = img_minmax_norm(input_box, channelsFirst)
            target_box = crop_img(target, x, y, size_box, size_box, channelsFirst)
            
            # erode the touching border
            if disk_size:
                target_box[1, :, :] = erosion(target_box[1, :, :], disk(disk_size))


            dict_input[cnt_img+offset] = input_box
            dict_output[cnt_img+offset] = target_box
            cnt_img += 1

            if cnt_img > earlyStop:
                break

            if display:
                print('img_%s_input.png' % cnt_img)
                input_box_rgb = get_rgb(input_box, channelsFirst)
                if not do_preprocess:
                    input_box_rgb = img_minmax_norm(input_box_rgb, channelsFirst)
                if channelsFirst:
                    input_box_rgb = np.moveaxis(input_box_rgb, 0, 2)
                
                f, axarr = plt.subplots(1, 5)
                axarr[0].imshow(input_box_rgb)
                axarr[1].imshow(target_box[0, :, :])
                axarr[2].imshow(target_box[1, :, :])
                axarr[3].imshow(target_box[2, :, :])
                axarr[4].imshow(target_box[0, :, :] - target_box[2, :, :])
                plt.show()

        if cnt_img > earlyStop:
            break

    return dict_input, dict_output