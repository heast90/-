
#!/usr/bin/env python
# coding: utf-8
#加载包
#预测模型及测试
#测试数据准备
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import PIL
import csv
import numpy as np
input_dir = './data/test_img'
output_dir = './data/output'
def process_test_data():
    test_images = os.listdir(input_dir)
    for image_name in test_images:
        image_path = input_dir + '/' + image_name
        image = cv2.imread(image_path)
        resize_image = cv2.resize(image,(32,32),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(output_dir + '/' + image_name,resize_image)
def load_image(file):
    image = cv2.imread(file)
    image = np.array(image).astype(np.float32).reshape(1, 3, 32, 32)#Batch大小，图片通道数，图片的宽和图片的高
    image = (image - np.mean(image))/(np.max(image) - np.min(image))
    return image
def load_test_data():
    test_data = []
    resize_images = os.listdir(output_dir)
    print(resize_images)
    for image_name in resize_images:
        image_path = output_dir + '/' + image_name
        image=load_image(image_path)
        test_data.append(image)
    return test_data
def load_test_data1(ind):
    resize_images = os.listdir(output_dir)
    image_path = output_dir + '/' + resize_images[ind]
    image = cv2.imread(image_path)
    return image
def load_label():
    file_path = './data/signnames.csv'
    signnames = []
    with open(file_path) as f:
        lines = csv.reader(f,delimiter=',')
        next(lines)
        for line in lines:
            signnames.append(line[1])
    return signnames
if __name__=="__main__":
    process_test_data()
    test_img=np.array(load_test_data())
    signnames=load_label()
    #Inferencer 配置和预测Inferencer 配置和预测
    use_cuda=False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    [inference_program, feed_target_names,fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)
    results = []
    for i in range(test_img.shape[0]):
        with fluid.program_guard(inference_program):
            res = exe.run(inference_program,feed={feed_target_names[0]:test_img[i]},fetch_list=fetch_targets)
            results.append(np.argmax(res[0]))
            print("infer results[%d] = %d -> %s " % (i,results[i],signnames[results[i]]))
    #绘图
    plt.figure(num='test_data',figsize=(21,8))
    n = len(results)
    for i in range(n):
        image = test_img[i][0]
        plt.subplot(2,n,i+1)
        plt.title("No." + str(i+1) +" image")
        plt.imshow(load_test_data1(i))
    for i in range(n):
        title = signnames[results[i]]
        plt.subplot(2,n,n+i+1)
        plt.title(title, fontsize=8)
        plt.imshow(load_test_data1(i))
    plt.show()
