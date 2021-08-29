import random
import numpy as np
import cv2
import torch

import SimpleITK as sitk
import numpy as np
import os
import random
import cv2
import torch

# Read in all images for a patient as a dict of np arrays
# Always in shape CDHW
def read_data(patient_dir):
    dict_images = {}
    list_structures = ['CT', 'PTV70', 'PTV63', 'PTV56',
                       'possible_dose_mask', 'Brainstem', 'SpinalCord',
                       'RightParotid', 'LeftParotid', 'Esophagus',
                       'Larynx', 'Mandible', 'dose']

    for structure_name in list_structures:
        structure_file = patient_dir + '/' + structure_name + '.nii.gz'

        if structure_name == 'CT':
            dtype = sitk.sitkInt16
        elif structure_name == 'dose':
            dtype = sitk.sitkFloat32
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            # To numpy array (C * Z * H * W)
            dict_images[structure_name] = sitk.GetArrayFromImage(dict_images[structure_name])[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)

    return dict_images



def preprocess_image(dict_images):
    # PTVs
    PTVs = 70.0 / 70. * dict_images['PTV70'] \
           + 63.0 / 70. * dict_images['PTV63'] \
           + 56.0 / 70. * dict_images['PTV56']

    # OARs
    OAR_names = ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
                 'Esophagus', 'Larynx', 'Mandible']
    OAR_all = np.concatenate([dict_images[OAR] for OAR in OAR_names], axis=0)

    # CT image
    CT = dict_images['CT']
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = CT.astype(np.float32) / 1000.

    # Dose image
    dose = dict_images['dose'] / 70.

    # Possible_dose_mask, the region that can receive dose
    possible_dose_mask = dict_images['possible_dose_mask']

    list_images = [np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input
                   dose,  # Label
                   possible_dose_mask]
    return list_images

def random_flip_3d(list_images, list_axis=(0, 1, 2), p=0.5):
    if random.random() <= p:
        if 0 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, ::-1, :, :].copy()
        if 1 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, ::-1, :].copy()
        if 2 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, :, ::-1].copy()

    return list_images


# Random rotation using OpenCV
def random_rotate_around_z_axis(list_images,
                                list_angles,
                                list_interp,
                                list_boder_value,
                                p=0.5):
    if random.random() <= p:
        # Randomly pick an angle list_angles
        _angle = random.sample(list_angles, 1)[0]
        # Do not use random scaling, set scale factor to 1
        _scale = 1.

        for image_i in range(len(list_images)):
            for chan_i in range(list_images[image_i].shape[0]):
                for slice_i in range(list_images[image_i].shape[1]):
                    rows, cols = list_images[image_i][chan_i, slice_i, :, :].shape
                    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), _angle, scale=_scale)
                    list_images[image_i][chan_i, slice_i, :, :] = \
                        cv2.warpAffine(list_images[image_i][chan_i, slice_i, :, :],
                                       M,
                                       (cols, rows),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=list_boder_value[image_i],
                                       flags=list_interp[image_i])
    return list_images


# Random translation
def random_translate(list_images, roi_mask, p, max_shift, list_pad_value):
    if random.random() <= p:
        exist_mask = np.where(roi_mask > 0)
        ori_z, ori_h, ori_w = list_images[0].shape[1:]

        bz = min(max_shift - 1, np.min(exist_mask[0]))
        ez = max(ori_z - 1 - max_shift, np.max(exist_mask[0]))
        bh = min(max_shift - 1, np.min(exist_mask[1]))
        eh = max(ori_h - 1 - max_shift, np.max(exist_mask[1]))
        bw = min(max_shift - 1, np.min(exist_mask[2]))
        ew = max(ori_w - 1 - max_shift, np.max(exist_mask[2]))

        for image_i in range(len(list_images)):
            list_images[image_i] = list_images[image_i][:, bz:ez + 1, bh:eh + 1, bw:ew + 1]

        # Pad to original size
        list_images = random_pad_to_size_3d(list_images,
                                            target_size=[ori_z, ori_h, ori_w],
                                            list_pad_value=list_pad_value)
    return list_images


# To tensor, images should be C*Z*H*W
def to_tensor(list_images):
    for image_i in range(len(list_images)):
        list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
    return list_images


# Pad
def random_pad_to_size_3d(list_images, target_size, list_pad_value):
    _, ori_z, ori_h, ori_w = list_images[0].shape[:]
    new_z, new_h, new_w = target_size[:]

    pad_z = new_z - ori_z
    pad_h = new_h - ori_h
    pad_w = new_w - ori_w

    pad_z_1 = random.randint(0, pad_z)
    pad_h_1 = random.randint(0, pad_h)
    pad_w_1 = random.randint(0, pad_w)

    pad_z_2 = pad_z - pad_z_1
    pad_h_2 = pad_h - pad_h_1
    pad_w_2 = pad_w - pad_w_1

    output = []
    for image_i in range(len(list_images)):
        _image = list_images[image_i]
        output.append(np.pad(_image,
                             ((0, 0), (pad_z_1, pad_z_2), (pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
                             mode='constant',
                             constant_values=list_pad_value[image_i])
                      )
    return output