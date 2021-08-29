import SimpleITK as sitk
import numpy as np
import os
import random
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from augmentation import *

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

def train_transform(list_images):
    # list_images = [Input, Label(gt_dose), possible_dose_mask]
    # Random flip
    list_images = random_flip_3d(list_images, list_axis=(0, 2), p=0.8)

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angles=(0, 40, 80, 120, 160, 200, 240, 280, 320),
                                              list_boder_value=(0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    # Random translation, but make use the region can receive dose is remained
    list_images = random_translate(list_images,
                                   roi_mask=list_images[2][0, :, :, :],  # the possible dose mask
                                   p=0.8,
                                   max_shift=20,
                                   list_pad_value=[0, 0, 0])
    
    # To torch tensor
    list_images = to_tensor(list_images)
    return list_images


def validation_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images

def test_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images


from torch.utils.data import Dataset, DataLoader

class DoseData(Dataset):
    def __init__(self, phase):
        assert phase in ['train','validation','val','test']
        self.phase = phase
        if self.phase == 'val':
            self.phase = 'validation'
        self.root = 'Data/'
        
        if self.phase == 'train':
            self.patient_list = [self.root+'pt_'+str(i) for i in range(1,201)]
        elif self.phase == 'validation':
            self.patient_list = [self.root+'pt_'+str(i) for i in range(201,241)]
        else:
            self.patient_list = [self.root+'pt_'+str(i) for i in range(241,341)]
            
        self.transform = eval(self.phase+'_transform')
            
    def __len__(self):
        return len(self.patient_list)
    
    def __getitem__(self, idx):
        patient = self.patient_list[idx]
        
        # A dict of the CSVs for a patient
        data_dict = read_data(patient)
        # Turn the dict into an array with [input, label, possible_dose_mask]
        data_images = preprocess_image(data_dict) 
        data_images = self.transform(data_images)
        return data_images
    
def get_dataloaders(train_size = 2, val_size = 2, test_size = 2):
    batch_sizes = {'train': train_size, 'val': val_size, 'test': test_size}
    datasets = {}
    dataloaders = {}
    for p in ['train','val','test']:
        datasets[p] = DoseData(p)
        dataloaders[p] = DataLoader(datasets[p], batch_size = batch_sizes[p],
                                    shuffle = (p == 'train'))
    return datasets, dataloaders