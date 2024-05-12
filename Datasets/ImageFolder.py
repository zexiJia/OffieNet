import torch
import cv2, logging
from utils import get_all_files
import numpy as np
from torchvision import transforms 
import random
from perlin_noise import PerlinNoise

def centerCrop(img, size):
    h, w = img.shape[0],img.shape[1]
    return  img[h//2 - size//2:h//2 + size//2, w//2 - size//2:w//2 + size//2]


def random_image_augmentation(image, kernel_size=3, noise_mean=0, noise_var=10, salt_and_pepper_amount=0.02, salt_and_pepper_type='s&p'):  
    """  
    Perform random image augmentation by applying erosion, dilation, adding Gaussian noise, and salt and pepper noise.  
  
    :param image: The input image, which should be grayscale or color.  
    :param kernel_size: The size of the kernel used for erosion and dilation operations.  
    :param noise_mean: The mean of the Gaussian noise.  
    :param noise_var: The variance of the Gaussian noise.  
    :param salt_and_pepper_amount: The density (proportion) of salt and pepper noise.  
    :param salt_and_pepper_type: The type of salt and pepper noise, 's' for salt noise, 'p' for pepper noise, 's&p' for both.  
    :return: The augmented image.  
    """  
    # Create a kernel for erosion and dilation  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))  
  
    # Randomly choose whether to perform erosion or dilation  
    if random.choice([True, False]):  
        image = cv2.erode(image, kernel)  # Erosion operation  
    else:  
        image = cv2.dilate(image, kernel)  # Dilation operation

  
    # Add Gaussian noise
    if random.choice([True, False]):  
        row, col = image.shape 
        mean = noise_mean  
        var = noise_var  
        sigma = var**0.5  
        gauss = np.random.normal(mean, sigma, (row, col))  
        gauss = gauss.reshape(row, col)  
        noisy = image + gauss  
        # Ensure values are within a valid range  
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)  
        image = noisy

    # Add perlin noise
    if random.choice([True, False]):
        
        pnoise = PerlinNoise(octaves=10, seed=1)
        xpix, ypix = image.shape
        noise = [[pnoise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
        
        noise = ((noise - np.min(noise)) / (np.max(noise) - np.min(noise))) * 255  
        noise = np.uint8(noise)  
        
        noise_factor = 0.1  
        image = image + noise_factor * noise

    # Add sc mask
    if random.choice([True, False]):
        total_pixels = image.size 
        num_pixels_to_zero = int(total_pixels * 0.1)  
        height, width = image.shape  
        random_indices = np.random.choice(total_pixels, num_pixels_to_zero, replace=False)  
        
        indices = np.unravel_index(random_indices, (height, width))  
        image[indices] = 0 

  
    # Add salt and pepper noise  
    if random.choice([True, False]):  
        amount = salt_and_pepper_amount  
        num_salt = np.ceil(amount * image.size * 0.5)  
        num_pepper = np.ceil(amount * image.size * 0.5)  
  
        # Add salt noise  
        coords = [np.random.randint(0, i - 1, int(num_salt))  
                  for i in image.shape]  
        if salt_and_pepper_type in ['s', 's&p']:  
            image[coords[0], coords[1]] = 255  
  
        # Add pepper noise  
        coords = [np.random.randint(0, i - 1, int(num_pepper))  
                  for i in image.shape]  
        if salt_and_pepper_type in ['p', 's&p']:  
            image[coords[0], coords[1]] = 0  
  
    return image  


class Images(torch.utils.data.Dataset):

    def __init__(self, base_path, T, keywordforT):
        self.T = T
        self.keyT = keywordforT
        self.base_path = base_path if isinstance(base_path, list) else [base_path]
        self.images = []
        self.transform = transforms.Compose([  
            transforms.ToTensor()
        ])  
  
        for i in self.base_path:
            self.images += get_all_files(i)
            print(f'finish filter data, the total number is {len(self.images)}')
           

    def __getitem__(self, idx):
        ipath = self.images[idx]
        img = cv2.imread(ipath, flags=0)
        
        if img.shape!=(512,512):
            if img.shape[0] > 512:
                img = img[img.shape[0]//2 - 256:img.shape[0]//2 + 256,:]
            if img.shape[1] > 512:
                img = img[:,img.shape[1]//2 - 256:img.shape[1]//2 + 256]

            if img.shape[0] < 512 or img.shape[1] < 512 :
                tmp = np.ones([512,512])
                tmp = tmp * 255
                tmp[256-img.shape[0]//2:256+img.shape[0]//2,256-img.shape[1]//2:256+img.shape[1]//2] = img
                img = tmp

        img = self.transform(img)  
        return img, ipath

    def __len__(self):
        return len(self.images)
    

class NoiseImages(torch.utils.data.Dataset):

    def __init__(self, base_path, T, keywordforT):
        self.T = T
        self.keyT = keywordforT
        self.base_path = base_path if isinstance(base_path, list) else [base_path]
        self.images = []
        self.transform = transforms.Compose([  
            transforms.ToTensor()
        ])  
  
        for i in self.base_path:
            self.images += get_all_files(i)
            print(f'finish filter data, the total number is {len(self.images)}')
           

    def __getitem__(self, idx):
        ipath = self.images[idx]
        img = cv2.imread(ipath, flags=0)
        
        if img.shape!=(512,512):
            if img.shape[0] > 512:
                img = img[img.shape[0]//2 - 256:img.shape[0]//2 + 256,:]
            if img.shape[1] > 512:
                img = img[:,img.shape[1]//2 - 256:img.shape[1]//2 + 256]

            if img.shape[0] < 512 or img.shape[1] < 512 :
                tmp = np.ones([512,512])
                tmp = tmp * 255
                tmp[256-img.shape[0]//2:256+img.shape[0]//2,256-img.shape[1]//2:256+img.shape[1]//2] = img
                img = tmp
        
        random_number = random.randint(1, 10)
        if random_number > 5:
            img = random_image_augmentation(img)
        else:
            img = img

        img = self.transform(img)  
        return img, ipath

    def __len__(self):
        return len(self.images)
    


