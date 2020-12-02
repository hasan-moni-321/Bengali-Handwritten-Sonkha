
import os, cv2
import numpy as np 

import matplotlib.image as mpimg
from torch.utils.data import Dataset



class ImageData(Dataset):
    def __init__(self, df, data_dir, resize, transform):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.data_dir)
    
    def __getitem__(self, index): 
        label = self.df.digit[index]      
        img_name = self.df.filename[index] 
        #resize = self.resize[index] 
        
        img_path = os.path.join(self.data_dir, img_name)
        image = mpimg.imread(img_path)
        if self.resize is not None:
            image = cv2.resize(image, (self.resize, self.resize), interpolation=cv2.INTER_AREA)

        gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0) #unblur
        image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)  
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter
        image = cv2.filter2D(image, -1, kernel)
        image = image.transpose(2, 0,1)
        #print(image.shape) 
   
        return image, label


