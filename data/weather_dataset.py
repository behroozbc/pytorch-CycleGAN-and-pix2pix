from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import os
import random
class WeatherDataset(BaseDataset):
    def __init__(self,opt):
        BaseDataset.__init__(self, opt)
        self.dir_camera = os.path.join(opt.dataroot, opt.phase + '_camera')  # create a path '/path/to/data/train_camera'
        self.dir_cp = os.path.join(opt.dataroot, opt.phase + '_cp')  # create a path '/path/to/data/train_cp'
        self.dir_weather = os.path.join(opt.dataroot, opt.phase + '_weather')  # create a path '/path/to/data/train_weather'
        self.camera_paths = sorted(make_dataset(self.dir_camera, opt.max_dataset_size))   # load images from '/path/to/data/train_camera'
        self.cp_paths = sorted(make_dataset(self.dir_cp, opt.max_dataset_size))    # load images from '/path/to/data/train_cp'
        self.weather_paths = sorted(make_dataset(self.dir_weather, opt.max_dataset_size))    # load images from '/path/to/data/train_weather'
        self.camera_size = len(self.camera_paths)  # get the size of dataset camera
        self.cp_size = len(self.cp_paths)  # get the size of dataset cp
        self.weather_size = len(self.cp_paths)  # get the size of dataset cp
        self.weather_size = len(self.weather_paths)  # get the size of dataset weather
        self.camera_transport=get_transform(opt)
        self.weather_trasport=get_transform(opt,grayscale=True)
        self.cp_transport=get_transform(opt,grayscale=True)
    def __getitem__(self, index):
        cameraPath = self.camera_paths[index % self.camera_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            indexCp = index % self.cp_size
            indexWeather= index % self.weather_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            indexCp = random.randint(0, self.cp_size - 1)
            indexWeather = random.randint(0, self.weather_size - 1)
        cpPath = self.cp_paths[indexCp]
        weatherPath = self.weather_paths[indexWeather]
        
        camera_img = Image.open(cameraPath).convert('RGB')
        cp_img = Image.open(cpPath).convert('L')
        weather_img=Image.open(weatherPath).convert('L')
        
        # apply image transformation
        camera = self.camera_transport(camera_img)
        cp = self.cp_transport(cp_img)
        weather=self.weather_trasport(weather_img)
        return {'camera': camera, 'cp': cp,'weather':weather, 'camera_paths': cameraPath, 'cp_paths': cpPath,'weather_paths':weatherPath}
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.weather_size,max(self.camera_size,self.cp_size))