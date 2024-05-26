import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input

# import pandas as pd


# -------------------------------这是多模态Unet数据输入（第三版-双标签数据集）---------------------------------  #
class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

        # 假设模态A和模态B的图像分别存储在以下两个文件夹中
        self.images_path_A = os.path.join(self.dataset_path, "Images")
        self.images_path_B = os.path.join(self.dataset_path, "ImagesB")
        # 假设模态A和模态B的标签分别存储在以下两个文件夹中
        self.labels_path_A = os.path.join(self.dataset_path, "Labels")
        self.labels_path_B = os.path.join(self.dataset_path, "LabelsB")


    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # 分别加载模态A和模态B的图像
        jpg_A = Image.open(os.path.join(self.images_path_A, name + ".jpg"))
        jpg_B = Image.open(os.path.join(self.images_path_B, name + ".jpg"))
        # 分别加载模态A和模态B的标签图像
        label_A = Image.open(os.path.join(self.labels_path_A, name + ".jpg"))
        label_B = Image.open(os.path.join(self.labels_path_B, name + ".jpg"))

        # 对两种模态的图像以及标签图像进行数据增强
        jpg_A, jpg_B, label_A, label_B = self.get_random_data(jpg_A, jpg_B, label_A, label_B, self.input_shape, random=self.train)

        jpg_A = np.transpose(preprocess_input(np.array(jpg_A, np.float64)), [2, 0, 1])
        jpg_B = np.transpose(preprocess_input(np.array(jpg_B, np.float64)), [2, 0, 1])
        label_A = np.array(label_A)
        label_B = np.array(label_B)

        # 对标签进行处理，例如二值化处理
        modify_label_A = np.zeros_like(label_A)
        modify_label_A[label_A <= 127.5] = 1
        modify_label_B = np.zeros_like(label_B)
        modify_label_B[label_B <= 127.5] = 1

        seg_labels_A = np.eye(self.num_classes + 1)[modify_label_A.reshape([-1])]
        seg_labels_A = seg_labels_A.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        seg_labels_B = np.eye(self.num_classes + 1)[modify_label_B.reshape([-1])]
        seg_labels_B = seg_labels_B.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        # 返回模态A和模态B的图像以及对应的标签
        return jpg_A, jpg_B, seg_labels_A, seg_labels_B
    
    
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
    # -------------------------------------传入两个Label------------------------------------- #
    def get_random_data(self, image_A, image_B, label_A, label_B, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        # 你需要实现这个方法来同时对模态A和模态B的图像以及标签进行数据增强
    # -------------------------------数据增强还需要改改---------------------------------  #
        image_A   = cvtColor(image_A)
        image_B   = cvtColor(image_B)
        label_A   = Image.fromarray(np.array(label_A))
        label_B   = Image.fromarray(np.array(label_B))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image_A.size
        h, w    = input_shape
        
        if not random:
            # 对两个图像进行相同的缩放操作
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            image_A = image_A.resize((nw, nh), Image.BICUBIC)
            image_B = image_B.resize((nw, nh), Image.BICUBIC)
            label_A = label_A.resize((nw, nh), Image.NEAREST)
            label_B = label_B.resize((nw, nh), Image.NEAREST)

            new_image_A = Image.new('RGB', [w, h], (128,128,128))
            new_image_B = Image.new('RGB', [w, h], (128,128,128))
            new_label_A = Image.new('L', [w, h], (0))
            new_label_B = Image.new('L', [w, h], (0))

            new_image_A.paste(image_A, ((w-nw)//2, (h-nh)//2))
            new_image_B.paste(image_B, ((w-nw)//2, (h-nh)//2))
            new_label_A.paste(label_A, ((w-nw)//2, (h-nh)//2))
            new_label_B.paste(label_B, ((w-nw)//2, (h-nh)//2))

            return new_image_A, new_image_B, new_label_A, new_label_B


        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image_A = image_A.resize((nw, nh), Image.BICUBIC)
        image_B = image_B.resize((nw, nh), Image.BICUBIC)
        label_A = label_A.resize((nw, nh), Image.NEAREST)
        label_B = label_B.resize((nw, nh), Image.NEAREST)

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip:
            image_A = image_A.transpose(Image.FLIP_LEFT_RIGHT)
            image_B = image_B.transpose(Image.FLIP_LEFT_RIGHT)
            label_A = label_A.transpose(Image.FLIP_LEFT_RIGHT)
            label_B = label_B.transpose(Image.FLIP_LEFT_RIGHT)

        #------------------------------------------#
        #   应用色域变换等操作
        #   先将图像多余的部分加上灰条
        #------------------------------------------#

        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))

        new_image_A = Image.new('RGB', (w, h), (128, 128, 128))
        new_image_B = Image.new('RGB', (w, h), (128, 128, 128))
        new_label_A = Image.new('L', (w, h), (0))
        new_label_B = Image.new('L', (w, h), (0))

        new_image_A.paste(image_A, (dx, dy))
        new_image_B.paste(image_B, (dx, dy))
        new_label_A.paste(label_A, (dx, dy))
        new_label_B.paste(label_B, (dx, dy))

        image_A = new_image_A
        image_B = new_image_B
        label_A = new_label_A
        label_B = new_label_B

       
        # 将PIL图像转换为numpy数组
        image_data_A = np.array(image_A, np.uint8)
        image_data_B = np.array(image_B, np.uint8)
        
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        
        for image_data in [image_data_A, image_data_B]:
            #---------------------------------#
            #   将图像转到HSV色域上
            #---------------------------------#      
            hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
            dtype           = image_data.dtype
            #---------------------------------#
            #   应用变换
            #---------------------------------#
            x       = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            
            image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        image_A = Image.fromarray(image_data_A)
        image_B = Image.fromarray(image_data_B)

        # final确保图像和标签大小一致
        image_A = image_A.resize((w, h), Image.BICUBIC)
        image_B = image_B.resize((w, h), Image.BICUBIC)
        label_A = label_A.resize((w, h), Image.NEAREST)
        label_B = label_B.resize((w, h), Image.NEAREST)
        return image_A, image_B, label_A, label_B
    

# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images_A = []
    images_B = []
    seg_labels_A = []
    seg_labels_B = []
    for img_A, img_B, label_A, label_B in batch:
        images_A.append(img_A)
        images_B.append(img_B)
        seg_labels_A.append(label_A)
        seg_labels_B.append(label_B)

    images_A = torch.from_numpy(np.array(images_A)).type(torch.FloatTensor)
    images_B = torch.from_numpy(np.array(images_B)).type(torch.FloatTensor)
    # jpgs = torch.from_numpy(np.array(jpgs)).long()
    seg_labels_A = torch.from_numpy(np.array(seg_labels_A)).type(torch.FloatTensor)
    seg_labels_B = torch.from_numpy(np.array(seg_labels_B)).type(torch.FloatTensor)
    return images_A, images_B, seg_labels_A, seg_labels_B
    # return images_A, images_B, jpgs, seg_labels




# -------------------------------这是原Unet数据输入---------------------------------  #
# class UnetDataset(Dataset):
#     def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
#         super(UnetDataset, self).__init__()
#         self.annotation_lines   = annotation_lines
#         self.length             = len(annotation_lines)
#         self.input_shape        = input_shape
#         self.num_classes        = num_classes
#         self.train              = train
#         self.dataset_path       = dataset_path

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         annotation_line = self.annotation_lines[index]
#         name            = annotation_line.split()[0]

#         #-------------------------------#
#         #   从文件中读取图像
#         #-------------------------------#
#         jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "Images"), name + ".png"))
#         png         = Image.open(os.path.join(os.path.join(self.dataset_path, "Labels"), name + ".png"))
#         #-------------------------------#
#         #   数据增强
#         #-------------------------------#
#         jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

#         jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
#         png         = np.array(png)
#         #-------------------------------------------------------#
#         #   这里的标签处理方式和普通voc的处理方式不同
#         #   将小于127.5的像素点设置为目标像素点。
#         #-------------------------------------------------------#
#         modify_png  = np.zeros_like(png)
#         modify_png[png <= 127.5] = 1
#         seg_labels  = modify_png
#         seg_labels  = np.eye(self.num_classes + 1)[seg_labels.reshape([-1])]
#         seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

#         return jpg, modify_png, seg_labels

#     def rand(self, a=0, b=1):
#         return np.random.rand() * (b - a) + a

#     def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
#         image   = cvtColor(image)
#         label   = Image.fromarray(np.array(label))
#         #------------------------------#
#         #   获得图像的高宽与目标高宽
#         #------------------------------#
#         iw, ih  = image.size
#         h, w    = input_shape

#         if not random:
#             iw, ih  = image.size
#             scale   = min(w/iw, h/ih)
#             nw      = int(iw*scale)
#             nh      = int(ih*scale)

#             image       = image.resize((nw,nh), Image.BICUBIC)
#             new_image   = Image.new('RGB', [w, h], (128,128,128))
#             new_image.paste(image, ((w-nw)//2, (h-nh)//2))

#             label       = label.resize((nw,nh), Image.NEAREST)
#             new_label   = Image.new('L', [w, h], (0))
#             new_label.paste(label, ((w-nw)//2, (h-nh)//2))
#             return new_image, new_label

#         #------------------------------------------#
#         #   对图像进行缩放并且进行长和宽的扭曲
#         #------------------------------------------#
#         new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
#         scale = self.rand(0.25, 2)
#         if new_ar < 1:
#             nh = int(scale*h)
#             nw = int(nh*new_ar)
#         else:
#             nw = int(scale*w)
#             nh = int(nw/new_ar)
#         image = image.resize((nw,nh), Image.BICUBIC)
#         label = label.resize((nw,nh), Image.NEAREST)
        
#         #------------------------------------------#
#         #   翻转图像
#         #------------------------------------------#
#         flip = self.rand()<.5
#         if flip: 
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#             label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
#         #------------------------------------------#
#         #   将图像多余的部分加上灰条
#         #------------------------------------------#
#         dx = int(self.rand(0, w-nw))
#         dy = int(self.rand(0, h-nh))
#         new_image = Image.new('RGB', (w,h), (128,128,128))
#         new_label = Image.new('L', (w,h), (0))
#         new_image.paste(image, (dx, dy))
#         new_label.paste(label, (dx, dy))
#         image = new_image
#         label = new_label

#         image_data      = np.array(image, np.uint8)
#         #---------------------------------#
#         #   对图像进行色域变换
#         #   计算色域变换的参数
#         #---------------------------------#
#         r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
#         #---------------------------------#
#         #   将图像转到HSV上
#         #---------------------------------#
#         hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
#         dtype           = image_data.dtype
#         #---------------------------------#
#         #   应用变换
#         #---------------------------------#
#         x       = np.arange(0, 256, dtype=r.dtype)
#         lut_hue = ((x * r[0]) % 180).astype(dtype)
#         lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
#         lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

#         image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
#         image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
#         return image_data, label

# # DataLoader中collate_fn使用
# def unet_dataset_collate(batch):
#     images      = []
#     pngs        = []
#     seg_labels  = []
#     for img, png, labels in batch:
#         images.append(img)
#         pngs.append(png)
#         seg_labels.append(labels)
#     images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     pngs        = torch.from_numpy(np.array(pngs)).long()
#     seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
#     return images, pngs, seg_labels
