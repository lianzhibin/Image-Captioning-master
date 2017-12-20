#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:26:17 2017

@author: xiaolian
"""

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# extract features from each photo in the directory
def extract_features(directory):
    # 加载模型
    model = VGG16()
    # 重组模型结构
    model.layers.pop()
    model = Model(inputs = model.inputs, outputs = model.layers[-1].output)
    
    # 打印模型
    print(model.summary())
    
    # 从每张图片当中提取特征
    features = dict() 
    for name in listdir(directory):
        # 加载图片
        filename = directory + '/' + name
        image = load_img(filename, target_size = (224, 224))
        # 将图片像素模式转换到numpy array模式
        image = img_to_array(image)
        # reshape 数据
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # 为 VGG16 模型 准备数据
        image = preprocess_input(image)
        # 获取特征
        feature = model.predict(image)
        # 获取图片id 
        image_id = name.split('.')[0]
        # 存储特征
        features[image_id] = feature
        print('>%s' % name)
    return features
    
# extract features from all images
directory = 'Flickr8k_Data/Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))
    
    