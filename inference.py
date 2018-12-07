from networks import FaceBox
from encoderl import DataEncoder

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import time


def detect(im):
    h, w, _ = im.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    input_img = np.pad(im, pad, 'constant', constant_values=255)
    input_img = cv2.resize(input_img,(1024,1024))/255

    im_tensor = torch.from_numpy(input_img.transpose((2,0,1))).float()
    #print (im_tensor)
    loc, conf = net(im_tensor.unsqueeze(0).cuda())
    loc, conf = loc.cpu(), conf.cpu()
    #print (loc)
    boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0),
                                                F.softmax(conf.squeeze(0), dim=1))
    if probs[0] != 0:
        boxes = boxes.numpy()
        probs = probs.detach().numpy()
        if h <= w:
            boxes[:,1] = boxes[:,1]*w-pad1
            boxes[:,3] = boxes[:,3]*w-pad1
            boxes[:,0] = boxes[:,0]*w
            boxes[:,2] = boxes[:,2]*w
        else:
            boxes[:,1] = boxes[:,1]*h
            boxes[:,3] = boxes[:,3]*h
            boxes[:,0] = boxes[:,0]*h-pad1
            boxes[:,2] = boxes[:,2]*h-pad1

    return boxes, probs

def testIm(file):
    #im = cv2.imread(file)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    capture = cv2.VideoCapture(file)
    out = cv2.VideoWriter('park2.avi', fourcc, 20.0, (1280, 720))
    while(True):
        ret, im = capture.read()
        if im is None:
            print("can not open image:", file)
            return
        h,w,_ = im.shape
        boxes, probs = detect(im)
        if (boxes is None):
            cv2.imshow('photo', im)
            cv2.waitKey(2)
            continue
        #print (boxes)
    
        if probs[0] == 0:
            print('There is no face in the image')
            cv2.imshow('photo', im)
            cv2.waitKey(2)
            continue
            #exit()
        for i, (box) in enumerate(boxes):
            print('i=', i, 'box=', box)
            x1 = int(box[0])
            x2 = int(box[2])
            y1 = int(box[1])
            y2 = int(box[3])
            srcImage=im[y1-5:y2+5,x1-5:x2+5]
            cv2.imwrite("crop2.jpg", srcImage);
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(im, str(probs[i]), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,0))
        #out.write(im)
        #im=cv2.resize(im,(960,540))
        
        cv2.imshow('photo', im)
        #cv2.imwrite('picture/1111.jpg', im)
        cv2.waitKey(10000)


if __name__ == '__main__':
    net = FaceBox()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net.cuda()
    net.load_state_dict(torch.load('weight/faceboxes.pt', map_location=lambda storage, loc:storage), strict=False) 
    net.eval()
    data_encoder = DataEncoder()

    # given image path, predict and show
    #root_path = "picture/"
    #picture = 'timg.jpg'
    #"rtsp://10.168.3.128/live_sub_6.sdp"
    #testIm('/media/bozhon/ESD-USB/save_video/test5.mp4')
    #testIm('/home/tao/opencv_save_video/save_video/test5.avi')
    #testIm('/home/bozhon/桌面/plate_faceboxes/11.jpg')
    #testIm('/home/tao/Desktop/plate_faceboxes/plate_refine/CCPD/        
    testIm('/home/tao/Desktop/plate_faceboxes/ER00714.jpg')


