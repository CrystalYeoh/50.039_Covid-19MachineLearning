# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
# Torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

class Lung_Train_Dataset(Dataset):
    
    def __init__(self, img_dir, covid=None, transform=None):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        self.dataset_type = 'train'
        self.transform = transform
        self.covid = covid
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # There will be two types of classifications
        # infected classification: normal and infected
        # covid classification (only done on classified infected): non-covid and covid
        # self.classes = {0: 'normal', 1: 'infected'}
        self.infected_classes = {0: 'normal', 1: 'infected'}
        self.covid_classes = {0: 'non-covid', 1: 'covid'}
        self.classes = {0: 'normal', 1: 'non-covid', 2: 'covid'}
        
        # The dataset has been split in training, testing and validation datasets
        self.groups = 'train'
        
        # Number of images in each part of the dataset
        self.dataset_numbers = {'train_normal': 1341,
                                'train_non-covid': 2530,
                                'train_covid': 1345,
                                }
        
        # Path to images for different parts of the dataset
        self.dataset_paths = {'train_normal': f'{img_dir}/train/normal/',
                              'train_non-covid': f'{img_dir}/train/infected/non-covid/',
                              'train_covid': f'{img_dir}/train/infected/covid/',
                              }
        
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the training dataset of th Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal', 'non-covid' or 'covid'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)] -1
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            img = Image.open(f)
            if self.transform:
              img = self.transform(img)
            im = np.asarray(img)/255
        f.close()
        return im
    
    
    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train'.
        - class_val variable should be set to 'normal', 'non-covid' or 'covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        # see if dataset is being used for covid model
        if self.covid:
            data_nums = list(self.dataset_numbers.values())
            return data_nums[1] + data_nums[2]

        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its infected_label, and covid_label as a one hot vector, both
        in torch tensor format in dataset.
        """
        # Get item special method
        max_num_vals = list(self.dataset_numbers.values())
        max_num_normal = int(max_num_vals[0])
        max_num_noncovid = int(max_num_vals[1])

        # see if dataset is being used for covid model
        if self.covid:
            if index < max_num_noncovid:
                class_val = 'non-covid'
                infected_label = 1
                covid_label = 0
            else:
                class_val = 'covid'
                index = index - max_num_noncovid
                infected_label = 1
                covid_label = 1
        else:
            if index < max_num_normal:
                class_val = 'normal'
                infected_label = 0
                covid_label = 0
            elif index < max_num_noncovid+max_num_normal:
                class_val = 'non-covid'
                index = index - max_num_normal
                infected_label = 1
                covid_label = 0
            else:
                class_val = 'covid'
                index = index - max_num_noncovid - max_num_normal
                infected_label = 1
                covid_label = 1

        im = self.open_img(self.groups, class_val, index)
        
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, infected_label, covid_label


class Lung_Test_Dataset(Dataset):
    
    def __init__(self, img_dir, covid=None, transform=None):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        self.dataset_type = 'test'
        self.transform = transform
        self.covid = covid
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # There will be two types of classifications
        # infected classification: normal and infected
        # covid classification (only done on classified infected): non-covid and covid
        # self.classes = {0: 'normal', 1: 'infected'}
        self.infected_classes = {0: 'normal', 1: 'infected'}
        self.covid_classes = {0: 'non-covid', 1: 'covid'}
        self.classes = {0: 'normal', 1: 'non-covid', 2: 'covid'}
        
        # The dataset has been split in training, testing and validation datasets
        self.groups = 'test'
        
        # Number of images in each part of the dataset
        self.dataset_numbers = {
            'test_normal': 234,
            'test_non-covid': 242,
            'test_covid': 139,
        }
        
        # Path to images for different parts of the dataset
        self.dataset_paths = {
            'test_normal': f'{img_dir}/test/normal/',
            'test_non-covid': f'{img_dir}/test/infected/non-covid/',
            'test_covid': f'{img_dir}/test/infected/covid/',
        }
        
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the test dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'test'.
        - class_val variable should be set to 'normal', 'non-covid' or 'covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'test'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal', 'non-covid' or 'covid'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)] -1
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            img = Image.open(f)
            if self.transform:
              img = self.transform(img)
            im = np.asarray(img)/255
        f.close()
        return im
    
    
    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'test'.
        - class_val variable should be set to 'normal', 'non-covid' or 'covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        # see if dataset is being used for covid model
        if self.covid:
            data_nums = list(self.dataset_numbers.values())
            return data_nums[1] + data_nums[2]

        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its infected_label, and covid_label as a one hot vector, both
        in torch tensor format in dataset.
        """
        # Get item special method
        max_num_vals = list(self.dataset_numbers.values())
        max_num_normal = int(max_num_vals[0])
        max_num_noncovid = int(max_num_vals[1])

        # see if dataset is being used for covid model
        if self.covid:
            if index < max_num_noncovid:
                class_val = 'non-covid'
                infected_label = 1
                covid_label = 0
            else:
                class_val = 'covid'
                index = index - max_num_noncovid
                infected_label = 1
                covid_label = 1
        else:
            if index < max_num_normal:
                class_val = 'normal'
                infected_label = 0
                covid_label = 0
            elif index < max_num_noncovid+max_num_normal:
                class_val = 'non-covid'
                index = index - max_num_normal
                infected_label = 1
                covid_label = 0
            else:
                class_val = 'covid'
                index = index - max_num_noncovid - max_num_normal
                infected_label = 1
                covid_label = 1

        im = self.open_img(self.groups, class_val, index)
        
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, infected_label, covid_label


class Lung_Val_Dataset(Dataset):
    
    def __init__(self, img_dir, covid=None, transform=None):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        self.dataset_type = 'val'
        self.transform = transform
        self.covid = covid
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
         # There will be two types of classifications
        # infected classification: normal and infected
        # covid classification (only done on classified infected): non-covid and covid
        # self.classes = {0: 'normal', 1: 'infected'}
        self.infected_classes = {0: 'normal', 1: 'infected'}
        self.covid_classes = {0: 'non-covid', 1: 'covid'}
        self.classes = {0: 'normal', 1: 'non-covid', 2: 'covid'}
        
        # The dataset has been split in training, testing and validation datasets
        self.groups = 'val'
        
        # Number of images in each part of the dataset
        self.dataset_numbers = {
            'val_normal': 8,
            'val_non-covid': 8,
            'val_covid': 9,
        }
        
        # Path to images for different parts of the dataset
        self.dataset_paths = {
            'val_normal': f'{img_dir}/val/normal/',
            'val_non-covid': f'{img_dir}/val/infected/non-covid/',
            'val_covid': f'{img_dir}/val/infected/covid/',
        }
        
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the val dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'val'.
        - class_val variable should be set to 'normal', 'non-covid' or 'covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'val'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal', 'non-covid' or 'covid'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)] -1
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            img = Image.open(f)
            if self.transform:
              img = self.transform(img)
            im = np.asarray(img)/255
        f.close()
        return im
    
    
    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'val'.
        - class_val variable should be set to 'normal', 'non-covid' or 'covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        # see if dataset is being used for covid model
        if self.covid:
            data_nums = list(self.dataset_numbers.values())
            return data_nums[1] + data_nums[2]

        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its infected_label, and covid_label as a one hot vector, both
        in torch tensor format in dataset.
        """
        # Get item special method
        max_num_vals = list(self.dataset_numbers.values())
        max_num_normal = int(max_num_vals[0])
        max_num_noncovid = int(max_num_vals[1])

        # see if dataset is being used for covid model
        if self.covid:
            if index < max_num_noncovid:
                class_val = 'non-covid'
                infected_label = 1
                covid_label = 0
            else:
                class_val = 'covid'
                index = index - max_num_noncovid
                infected_label = 1
                covid_label = 1
        else:
            if index < max_num_normal:
                class_val = 'normal'
                infected_label = 0
                covid_label = 0
            elif index < max_num_noncovid+max_num_normal:
                class_val = 'non-covid'
                index = index - max_num_normal
                infected_label = 1
                covid_label = 0
            else:
                class_val = 'covid'
                index = index - max_num_noncovid - max_num_normal
                infected_label = 1
                covid_label = 1

        im = self.open_img(self.groups, class_val, index)
        
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, infected_label, covid_label

# dataset_dir = './dataset'
# ld_train = Lung_Train_Dataset(dataset_dir, covid=True, transform=transforms.Compose([
#   transforms.Resize((100,100)),
# ]))

# print(len(ld_train))

# trainloader = DataLoader(ld_train, batch_size = 4, shuffle = False)
# images_data, target_infected_labels, target_covid_labels = ld_train[7]
# print(images_data.shape)
# print(target_infected_labels)
# print(target_covid_labels)


# images_data, target_infected_labels, target_covid_labels = ld_train[8]
# print(images_data.shape)
# print(target_infected_labels)
# print(target_covid_labels)

# for batch_idx, data in enumerate(trainloader):
#   images_data, target_infected_labels, target_covid_labels = data
  
#   print(f'Batch {batch_idx}')
#   print(images_data.shape)
#   print(target_infected_labels, target_covid_labels)
#   assert False, "Forced stop after one iteration of the mini-batch for loop"


