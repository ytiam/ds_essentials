
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import pandas as pd
from numpy import moveaxis
from tqdm import tqdm
from typing import Union


class ImageProcessing_Utils:
    
    def __init__(self, path: str = None):
        self.path = path
        
    def _list_images(self, file_path: str = None, img_format: str = '.jpg') -> dict:
        '''
        Get all the paths of the images (for the specified image format) listed in the given folder
        
        Args:
        file_path - Folder/file path of the images
        img_format - The image format under interest
        
        Out:
        image_file_paths - a dictionary containing all the image file names and their respective
        paths
        '''
        image_file_paths = {}
        if file_path is None:
            file_path = self.path
            
        if os.path.isdir(file_path):
            for path_, subdirs, files in os.walk(file_path):
                for name in files:
                    image_file_paths[name.split(img_format)[0]] = os.path.join(path_,name)
                    
        elif os.path.isfile(file_path):
            name = os.path.basename(file_path)
            image_file_paths[name.split(img_format)[0]] = file_path
            
        return image_file_paths
    
    def read_image(self, img_path: str, resize_dim: (int,int) = None):
        '''
        Read Image from a given path and resize the image height and width if required

        Arguments:
        img_path - Image file path
        resize_dim - Default is None. If not None, the Image will be resized to (w1,h1) size

        Output:
        Image file original/resized
        '''
        image_data = Image.open(img_path)
        if resize_dim:
            image_data = image_data.resize(resize_dim)
        return image_data
    
    def image_to_numpy_array(self, image):
        '''
        Image to numpy array
        
        Args:
        image - Image type object, not image path
        
        Out:
        Numpy array
        '''
        return np.array(image)