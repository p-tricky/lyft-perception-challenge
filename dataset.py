import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class UDSegDataset(Dataset):
    def __init__(self, file_names, transform):
        self.file_names = file_names
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        bgr_img = cv2.imread(self.file_names[idx][0])
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        seg_img = cv2.imread(self.file_names[idx][1])[:544, :, 2]
        
        img = self.transform(rgb_img)
        road_mask = np.where((seg_img == 7) | (seg_img == 6), 1, 0)
        car_mask = np.where(seg_img == 10, 2, 0)
        car_mask[496:] = 0

        mask = road_mask+car_mask
        return img.cuda(), torch.tensor(mask, dtype=torch.long).cuda()