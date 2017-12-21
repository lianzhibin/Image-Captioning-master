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
'''
用 VGG16 模型提取图片特征
'''
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
        image = preprocess_input(image) # shape (1, 224, 224, 3)
        # 获取特征
        feature = model.predict(image) # shape (1, 4096)
        # 获取图片id 
        image_id = name.split('.')[0] # 格式 47287348248268_98787878777887
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
    
'''
提取图像描述
'''

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

filename = 'Flickr8k_Text/Flickr8k_text/Flickr8k.token.txt'
# load_descriptions
doc = load_doc(filename)

# 提取图像字幕
def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        
        if image_id not in mapping:
            mapping[image_id] = list()
        
        mapping[image_id].append(image_desc)
        
    return mapping

descriptions = load_descriptions(doc)
print('Loaded: %d' % len(descriptions))

'''
清洗文本
'''
import string 
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# 所有的单词转换为小写字母
			desc = [word.lower() for word in desc]
			# 移除标点符号
			desc = [w.translate(table) for w in desc]
			# 移除只有一个字符的单词
			desc = [word for word in desc if len(word)>1]
			# 移除带数字的单词
			desc = [word for word in desc if word.isalpha()]
			# 放回去
			desc_list[i] =  ' '.join(desc)

# clean descriptions
clean_descriptions(descriptions)

# 把文本装换成词向量
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary)) 


# 保存图像标识符 和 描述
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# save descriptions
save_descriptions(descriptions, 'descriptions.txt')


'''
准备训练数据
1、找到训练集图片 identifier、描述、图片特征
'''
from pickle import load
 
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1 :
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
print(train)
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
print(train_descriptions)
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
print(train_features)


from keras.preprocessing.text import Tokenizer

# 把字典类型转换成列表类型
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# 根据给出的描述生成 tokenizer
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# 计算最大单词数
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    # 遍历每张图片
    for key, desc_list in descriptions.items():
        # 遍历每张图片对应的每个描述
        for desc in desc_list:
            # 编码序列
            print('desc: ', desc)
            seq = tokenizer.texts_to_sequences([desc])[0]
            print('seq: ', seq)
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                # 填充
                in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
                print('in_seq', in_seq)
                # 编码输出序列
                out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
                print('out_seq', out_seq)
                # 存储
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.appen(out_seq)
    return array(X1), array(X2), array(y)



from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import add
from keras.utils import plot_model

# 定义模型
def define_model(vocab_size, max_length):
    # 特征提取模型
    inputs1 = Input(shape = (4096, ))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation = 'relu')(fe1)

    # 序列模型 
    inputs2 = Input(shape = (max_length, ))
    se1 = Embedding(vocab_size, 256, mask_zero = True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # 编译模型
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # 模型 融合        
    model = Model(inputs = [inputs1, inputs2], outputs = outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    
    # 打印模型
    print(model.summary())
    
    plot_model(model, to_file = 'model.png', show_shapes = True)
    
    return model




max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)



# load training dataset (6K)
filename = 'Flickr8k_text/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)


 
from keras.callbacks import ModelCheckpoint

# load test set
filename = 'Flickr8k_text/Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

# define the model
model = define_model(vocab_size, max_length)

# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))


# 将 数字 转换成单词
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    
    return None

import numpy as np

# 生成 图像描述
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequence([in_text])[0]
        sequence = pad_sequences([sequence], maxlen = max_length)
        yhat = model.predict([photo, sequence], verbose = 0)
        # 将概率转换成数字下标
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

from nltk.translate.bleu_score import corpus_bleu

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    
    for key, desc_list in descriptions.items():
        # 生成描述
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        
        referencecs = [d.split() for d in desc_list]
        actual.append(referencecs)
        predicted.append(yhat.split())
        
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare test set

# load test set
filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))


from keras.models import load_model

# load the model
filename = 'model-ep002-loss3.245-val_loss3.612.h5'
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)





# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))



from numpy import argmax


# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature


# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('model-ep002-loss3.245-val_loss3.612.h5')
# load and prepare the photograph
photo = extract_features('example.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)






















 