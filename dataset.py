import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class UDSegDataset(Dataset):
    def __init__(self, file_names, path, transform):
        self.rgb_file_names = [path + 'CameraRGB/' + f for f in file_names]
        self.seg_file_names = [path + 'CameraSeg/' + f for f in file_names]
        self.transform = transform

    def __len__(self):
        return len(self.rgb_file_names)

    def __getitem__(self, idx):
        bgr_img = cv2.imread(self.rgb_file_names[idx])
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        seg_img = cv2.imread(self.seg_file_names[idx])[:544, :, 2]
        
        img = self.transform(rgb_img)
        road_mask = np.where((seg_img == 7) | (seg_img == 6), 1, 0)
        car_mask = np.where(seg_img == 10, 2, 0)
        car_mask[496:] = 0
#         mask = np.dstack((road_mask, car_mask)).transpose((2, 0, 1))
        mask = road_mask+car_mask
        return img.cuda(), torch.tensor(mask, dtype=torch.long).cuda()