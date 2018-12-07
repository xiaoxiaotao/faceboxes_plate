# coding:utf-8
'''
txt描述文件 image_name.jpg num x y w h 1 x y w h 1 这样就是说一张图片中有两个人脸
'''
import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2

from encoderl import DataEncoder

class ListDataset(data.Dataset):

    def __init__(self, root, list_file, train, transform):
        print('data init')
        self.image_size = 1024
        self.root=root
        self.train = train
        self.transform=transform
        self.fnames = [] # list: image name
        self.boxes = []
        self.labels = []
        self.small_threshold = 20./self.image_size  # face that small than threshold will be ignored
                                                    # it's 20 in the paper
        self.data_encoder = DataEncoder()

        with open(list_file) as f:
            lines  = f.readlines()

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_faces = int(splited[1])
            box=[]
            label=[]
            for i in range(num_faces):
                x = float(splited[2+5*i])
                y = float(splited[3+5*i])
                w = float(splited[4+5*i])
                h = float(splited[5+5*i])
                c = int(splited[6+5*i])
                box.append([x,y,x+w,y+h])
                label.append(c)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.tensor(label))
        self.num_samples = len(self.boxes) # num of images

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname))

        assert img is not None

        boxes = self.boxes[idx].clone()
        #print(boxes)
        labels = self.labels[idx].clone()

        if self.train:
            img, boxes= self.random_resize(img, boxes)
            #print(boxes)
            img, boxes = self.random_flip(img, boxes)

            img = img.astype(np.float32)
            img = self.random_bright(img)
            #img = self.random_swapchannel(img)
            img = self.random_distort(img)
            img = img.astype(np.uint8)

        h,w,_ = img.shape
        img = cv2.resize(img,(self.image_size,self.image_size))


        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        #print(labels)
        #print(boxes)
        for t in self.transform:
            img = t(img)

        loc_target,conf_target = self.data_encoder.encode(boxes,labels)

        return img,loc_target,conf_target

    def __len__(self):
        return self.num_samples

    def random_getim(self):
        idx = random.randrange(0, self.num_samples)
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname))
        box = self.boxes[idx].clone()
        label = self.labels[idx].clone()

        return img, box, label

    def random_flip(self, im, boxes):
        if random.random() < 0.65:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_resize(self, img, boxes):
# =============================================================================
#         if random.random() > 0 and random.random() <= 0.25:
#                 scale=1.5
#         elif random.random() > 0.25 and random.random() <= 0.5:
#             scale=2.5
#         elif random.random() > 0.5 and random.random() <= 0.75:
#             scale=3.25
#         else:
#             scale=1
# =============================================================================
        scale_list=[1.,1.5,2.,2.75,3.45,5.,6.2,5.5,8.5,9.8]
        index=np.random.randint(0,10)
        scale=scale_list[index]
        w=1024
        h=1024   
        img=cv2.resize(img,(int(w/scale),int(h/scale)))
        h1, w1, _ = img.shape
        diffh=int((h-h1)/2)
        diffw=int((w-w1)/2)
        pad1,pad2=(diffh,diffh),(diffw,diffw)
        pad = (pad1, pad2, (0, 0))
        img = np.pad(img, pad, 'constant', constant_values=255)
        x1 = boxes[0][0]/scale
        y1 = boxes[0][1]/scale
        x2 = boxes[0][2]/scale
        y2 = boxes[0][3]/scale
        #new_box=[[x1+diffw,y1+diffh,x2+diffw,y2+diffh]]
        boxes[0][0]=(x1+diffw).int()
        boxes[0][1]=(y1+diffh).int()
        boxes[0][2]=(x2+diffw).int()
        boxes[0][3]=(y2+diffh).int()
        
        #print (type(new_box))
        
        return img,boxes
    def random_bright(self, im, delta=48):
        if random.random() < 0.65:
            delta = random.uniform(-delta, delta)
            im += delta
            im = im.clip(min=0, max=255)
        return im


    def random_swapchannel(self, im):
        perms = ((0, 1, 2), (0, 2, 1),
                 (1, 0, 2), (1, 2, 0),
                 (2, 0, 1), (2, 1, 0))
        if random.random() < 0.5:
            swap = perms[random.randrange(0, len(perms))]
            im = im[:, :, swap]
        return im

    def RandomContrast(self, im, lower=0.4, upper=1.5):
        if random.random() < 0.65:
            alpha = random.uniform(lower, upper)
            im *= alpha
            im = im.clip(min=0, max=255)
        return im

    def RandomSaturation(self, im, lower=0.35, upper=1.55):
        if random.random() < 0.5:
            im[:, :, 1] *= random.uniform(lower, upper)

        return im

    def RandomHue(self, im, delta=18.0):
        if random.random() < 0.5:
            im[:, :, 0] += random.uniform(-delta, delta)
            im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
            im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
        return im

    def for_distort(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        self.RandomSaturation(im)
        self.RandomHue(im)
        im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        return im


    def random_distort(self, im):
        if random.random() < 0.6:
            self.RandomContrast(im)
            self.for_distort(im)
        else:
            self.for_distort(im)
            self.RandomContrast(im)

        return im





if __name__ == '__main__':
    file_root = '/home/bozhon/桌面/ccpd_plate_lpr/plate/big_picture/'
    list_file = '/home/bozhon/桌面/ccpd_plate_lpr/plate/traintest.txt'
    train_dataset = ListDataset(root=file_root,
                                list_file=list_file,
                                train=True,
                                transform = [transforms.ToTensor()] )

    print('the dataset has %d image' % (len(train_dataset)))
    image, boxes,labels = train_dataset[111]


