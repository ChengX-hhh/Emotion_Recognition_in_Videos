import os
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch


class FER2013(data.Dataset):
    """docstring for ClassName"""
    def __init__(self, root, opt, transforms = None,):
        super(FER2013,self).__init__()
        self.root = root

        emotion_root = [os.path.join(root,path) for path in os.listdir(root)]
        self.samples = [ ]

        for e in emotion_root:
            images = os.listdir(e)
            images = [os.path.join(e, img) for img in images]
            self.samples += images
        # print (self.samples)
        
        self.opt = opt


        print ('dataset init finish')
        self.train_transform = T.Compose([
            T.Resize((128, 128)),  # 缩放
            # transforms.RandomCrop(32, padding=4),  # 随机裁剪
            T.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
            T.Normalize(0, 1),  # 标准化均值为0标准差为1
        ])


    def __getitem__(self,index):
        

        im = Image.open(self.samples[index])

        # im = np.asarray(im)
        im = self.train_transform(im)
        # print (im.shape)

        # im = torch.transpose(im,0,2)

        im = im.repeat_interleave(3, dim = 0)

        label = self.samples[index].split('/')[-2]
        label = int(label)
        return im, label
        
    def __len__(self):
        return len(self.samples)

def main():
    pass

if __name__ == '__main__':
    main()


        


