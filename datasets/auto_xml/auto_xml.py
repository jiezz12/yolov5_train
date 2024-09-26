import xml.etree.ElementTree as ET
import pickle
import os
from collections import OrderedDict
from os import listdir, getcwd
from os.path import join
import copy

import torch
'''
fire-detect:
    VOC2020 to yolo format code
'''
# data_classes =[
#     "白鹭",
#     "苍鹭",
#     "红隼",
#     "家燕",
#     "普通鸬鹚",
#     "原鸽",
#     "其他"]
data_classes = ["circle"]
#with open('classes.txt','r') as f:
 #   data_classes=(f.readlines())
class SetAnnotation():
    def __init__(self,in_file_xml,in_dir_img,out_dir_xml,classes=data_classes):
        # assert(in_dir_xml!=out_dir_xml)
        self.in_file_xml = in_file_xml
        self.in_dir_img = in_dir_img
        self.out_dir_xml = out_dir_xml
        self.classes = classes
    #pred[0]  xmin ymin w h
    def __call__(self,image_name,imagesize,pred):
        global data_root
        #确定文件的输入输出
        filepath = f'{self.in_dir_img}/{image_name}.jpg'
        in_file = open(self.in_file_xml, encoding='utf-8')
        out_file = f'{self.out_dir_xml}/{image_name}.xml'
        #获取文件的层次结构
        tree=ET.parse(in_file)
        root = tree.getroot()  #获取root节点，即1.3中的annotation节点
        folder = root.find('folder')
        folder.text = self.in_dir_img
        filename = root.find('filename')
        filename.text = image_name
        path = root.find('path')
        path.text = filepath
        size = root.find('size') #获取size节点
        size.find('width').text = f'{imagesize[0]}'#获取宽度节点信息
        size.find('height').text=  f'{imagesize[1]}'#获取高度节点信息
        if len(imagesize) == 3:
            size.find('depth').text =  f'{imagesize[2]}'
        objects = root.find('object')
        for i in range(pred.shape[0] -1):
            # Create a copy
            obj = copy.deepcopy(objects)
            # Append the copy
            root.append(obj)
        for obj,p in zip(root.iter('object'),pred):  #迭代获取所有的object节点
            cls = self.classes[int(p[5])]
            obj.find('name').text = cls                #获取name节点信息，即bbox的类别信息
            xmlbox = obj.find('bndbox')          #获取bbox的左上角点与右下角点坐标信息
            xmlbox.find('xmin').text = f'{p[0]:.2f}'
            xmlbox.find('ymin').text = f'{p[1]:.2f}'
            xmlbox.find('xmax').text =f'{p[2]:.2f}'
            xmlbox.find('ymax').text = f'{p[3]:.2f}'
        tree.write(out_file,encoding='utf-8')

if  __name__ == '__main__':
    pred = torch.randint(0,10,(4,6))
    print(pred)
    in_dir_xml = 'circle.xml'
    in_dir_img ='images'
    out_dir_xml= 'Annotations'

    #in_dir_xml 模板xml,这里的object只能有一个
    #in_dir_img 图像地址
    # out_dir_xml 生成的xml
    #classes总共有多少类
    setAnn = SetAnnotation(in_dir_xml,in_dir_img,out_dir_xml)
    setAnn('02630059',[100,100,2],pred)