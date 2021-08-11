# 关于自动驾驶的交通标志分类
本项目采用GTSRB数据源,基于VGG与RESNet图像分类的深度学习算法,使用百度paddle平台的fluid实现图像分类算法,并完成训练与验证。

## 一、项目背景

智能驾驶作为汽车技术的发展形式之一，具有重要意义。汽车行驶的路况大部分属于城市道路，在其中根据交通标志进行驾驶行为的调整是一项最基础智能决策,所以考虑使用使用城市交通标志进行图形分类助理智能决策,显得十分具有意义。

## 二、数据集简介

本项目使用的数据集来源于Benchmark官方GTSRB数据集

### 1.数据加载
    

```python
%%bash 
mkdir data
cd ./data 
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip 
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip
unzip -n GTSRB_Final_Test_Images.zip 
unzip -n GTSRB_Final_Training_Images.zip
unzip -n GTSRB_Final_Test_GT.zip
cp ./GT-final_test.csv ./GTSRB/Final_Test/Images/GT-final_test.csv
unzip -n test_img.zip
mkdir output
```

### 2.数据读取及打包
```python
import numpy as np
import pickle
import os
import cv2
import csv
from PIL import Image
def process_train_data(path):
    file = os.listdir(path)
    classes = len(file)
    train_data = []
    train_labels = []
    cnt = 0
    for i in range(0,classes):
        dir_name = file[i]
        if dir_name=='.DS_Store':
            continue
        full_dir_path = path + dir_name
        csv_file_path = full_dir_path + '/' + 'GT-{0}.csv'.format(dir_name)
        with open(csv_file_path) as f:
            csv_reader = csv.reader(f,delimiter=';')
            # pass header
            next(csv_reader)
            for (filename,width,height,x1,y1,x2,y2,classid) in csv_reader:
                train_labels.append(classid)
                image_file_path = full_dir_path+'/'+filename
                resized_image = resize_image(image_file_path,(x1,y1,x2,y2))
                train_data.append(resized_image)
                cnt += 1
            f.close()
    print('训练集样本量: %d,'%cnt)
    return train_data,train_labels
def resize_image(path,index):
    image = cv2.imread(path)
    image = image[int(index[0]):int(index[2]),int(index[1]):int(index[3])]
    image = cv2.resize(image,(32,32),interpolation = cv2.INTER_CUBIC)
    image = np.array(image).astype(np.float32).reshape(1, 3, 32, 32)#Batch大小，图片通道数，图片的宽和图片的高
    image = (image - np.mean(image))/(np.max(image) - np.min(image))
    return image
def process_test_data(path):
    test_data = []
    test_labels = []
    csv_file_path = path + '/' + 'GT-final_test.csv'
    cnt = 0
    with open(csv_file_path) as f:
        csv_reader = csv.reader(f,delimiter=';')
        next(csv_reader)
        for (filename,width,height,x1,y1,x2,y2,classid) in csv_reader:
            test_labels.append(classid)
            image_file_path = path+'/'+filename
            resized_image = resize_image(image_file_path,(x1,y1,x2,y2))
            test_data.append(resized_image)
            cnt += 1
    print('测试集样本量： %d\n' % cnt)
    return test_data,test_labels

def main():
	train_data_path = './data/GTSRB/Final_Training/Images/'
	test_data_path = './data/GTSRB/Final_Test/Images'
	train_data,train_labels = process_train_data(train_data_path)
	test_data,test_labels = process_test_data(test_data_path)
	with open('./data/train.p','wb') as f:
		pickle.dump({"data":np.array(train_data),"labels":np.array(train_labels)},f)
	with open('./data/test.p','wb') as f:
		pickle.dump({"data":np.array(test_data),"labels":np.array(test_labels)},f)
		
if __name__=="__main__":
	main()
```
训练集样本量: 39209，验证集样本量: 12630


### 2.Data Feed格式转换


```python
import pickle 
def trainDataReader():
    train_file = './data/train.p'
    with open(train_file,'rb') as f:
        train=pickle.load(f)
    X_train,Y_train = train['data'],train['labels']
    trainset = []
    for img,label in zip(X_train,Y_train):
        yield img,int(label)
def testDataReader():
    test_file = './data/test.p'
    with open(test_file,'rb') as f:
        test = pickle.load(f)
    X_test,Y_test = test['data'],test['labels']
    testset = []
    for img,label in zip(X_test,Y_test):
        yield img,int(label)
```

## 三、模型选择

本项目采样VGG与ResNet算法进行实现:

### 1.VGG 16

<div align="center">
	<img src="https://githubraw.cdn.bcebos.com/PaddlePaddle/book/develop/03.image_classification/image/vgg16.png" width="70%">
</div>

```python
# 模型网络结构搭建
def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=43, act='softmax')
    return predict
```

### 2.ResNet
<div align="center">
	<img src="https://githubraw.cdn.bcebos.com/PaddlePaddle/book/develop/03.image_classification/image/resnet.png" width="70%">
</div>

```python
def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  bias_attr=False):
    tmp = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)
    return fluid.layers.batch_norm(input=tmp, act=act)

def shortcut(input, ch_in, ch_out, stride):
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input

def basicblock(input, ch_in, ch_out, stride):
    tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
    short = shortcut(input, ch_in, ch_out, stride)
    return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

def layer_warp(block_func, input, ch_in, ch_out, count, stride):
    tmp = block_func(input, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1)
    return tmp

def resnet(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) // 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(ipt, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    predict = fluid.layers.fc(input=pool, size=43, act='softmax')
    return predict
```


## 四、训练搭建
### 1.库加载及训练参数

```python
import paddle.fluid as fluid
import paddle 
import numpy as np 
class_size= 43
BATCH_SIZE = 150
learning_rate=0.01
EPOCH_NUM = 50
```

### 2.训练前配置
```python
def inference_program():
    # The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    paddle.enable_static()
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    predict = resnet(images, 32)
    # predict = vgg_bn_drop(images) # un-comment to use vgg net
    return predict
def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=learning_rate)
def train_program():
    global predict
    predict = inference_program()
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    print(label)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]
```
### 3.训练程序
#### 1）Data Feeders 配置
```python
BATCH_SIZE = 150
train_reader = paddle.batch(
    paddle.reader.shuffle(trainDataReader,buf_size=39209),
    batch_size=BATCH_SIZE)
# Reader for testing. A separated data set for testing.
test_reader = paddle.batch(testDataReader, 
                           batch_size=BATCH_SIZE)
```
#### 2）Train程序实现
```python
use_cuda = False 
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
feed_order = ['pixel', 'label']
main_program = fluid.default_main_program()
star_program = fluid.default_startup_program()
avg_cost, acc= train_program()
# Test program
test_program = main_program.clone(for_test=True)
optimizer = optimizer_program()
optimizer.minimize(avg_cost)
exe = fluid.Executor(place)
# For training test cost
def train_test(program, reader):
    count = 0
    feed_var_list = [
        program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder_test = fluid.DataFeeder(
        feed_list=feed_var_list, place=place)
    test_exe = fluid.Executor(place)
    accumulated = len([avg_cost, acc]) * [0]
    for tid, test_data in enumerate(reader()):
        avg_cost_np = test_exe.run(program=program,
                                   feed=feeder_test.feed(test_data),
                                   fetch_list=[avg_cost, acc])
        accumulated = [x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)]
        count += 1
    return [x / count for x in accumulated]
```
#### 3）训练主循环
```python
params_dirname = "image_classification_resnet.inference.model"
# params_dirname = "image_classification_vgg.inference.model"

train_prompt = "Train cost"
test_prompt = "Test cost"
plot_cost =[[[],[]],[[],[]]]
# main train loop.
def train_loop():
    feed_var_list_loop = [
        main_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(
        feed_list=feed_var_list_loop, place=place)
    exe.run(star_program)
    step = 0.0
    for pass_id in range(EPOCH_NUM):
        for step_id, data_train in enumerate(train_reader()):
            avg_loss_value = exe.run(main_program,
                                     feed=feeder.feed(data_train),
                                     fetch_list=[avg_cost, acc])
            if step % 50 == 0:
                plot_cost[0][0].append(step)
                plot_cost[0][1].append(avg_loss_value[0][0])
                print('Pass:%d, Batch:%d, Cost:%0.5f,Acc:%0.5f' % (pass_id, step_id, avg_loss_value[0],avg_loss_value[1]))   
            step += 1
        avg_cost_test, accuracy_test = train_test(test_program,
                                                  reader=test_reader)
        plot_cost[1][0].append(step)
        plot_cost[1][1].append(avg_cost_test)
        print('Test with Pass:%d, Cost:%0.5f,Acc:%0.5f' % (pass_id, avg_cost_test,accuracy_test))   
        # save parameters
        if params_dirname is not None:
            fluid.io.save_inference_model(params_dirname, ["pixel"],
                                          [predict], exe)
```
### 4.训练
```python
train_loop()
```
```
Pass:0, Batch:0, Cost:4.57999,Acc:0.04667
Pass:0, Batch:50, Cost:2.20741,Acc:0.29333
Pass:0, Batch:100, Cost:1.41025,Acc:0.60000
Pass:0, Batch:150, Cost:1.02150,Acc:0.68000
Pass:0, Batch:200, Cost:0.88321,Acc:0.72000
Pass:0, Batch:250, Cost:0.46722,Acc:0.84667
Test with Pass:0, Cost:1.00623,Acc:0.73482
Pass:1, Batch:38, Cost:0.53123,Acc:0.86667
Pass:1, Batch:88, Cost:0.43890,Acc:0.86667
Pass:1, Batch:138, Cost:0.37135,Acc:0.90667
```
### 5.训练结果可视化
```python
import matplotlib.pyplot as plt 
title=' vgg error rate'
# title=' resnet error rate'
label_train='Train error'
label_test='Test error'
plt.rcParams['font.sans-serif'] =['SimHei']
plt.title(title, fontsize=24)
plt.xlabel("epoch", fontsize=20)
plt.ylabel("error", fontsize=20)
x=39209/BATCH_SIZE
plt.plot(np.array(plot_cost[0][0])/x, plot_cost[0][1],color='red',label=label_train) 
plt.plot(np.array(plot_cost[1][0])/x, plot_cost[1][1],color='green',label=label_test) 
plt.legend()
plt.grid()
plt.show()
```
训练模型保存完成！\
画出训练测试误差图
![image](https://user-images.githubusercontent.com/69491295/128952970-6629f52f-3792-443a-a3af-d2adc8fba13f.png)
![image](https://user-images.githubusercontent.com/69491295/128952994-0387fa20-3391-42d1-b574-810fbf11307b.png)

## 五、模型预测
### 1.加载测试数据
```python
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
```
```python
 process_test_data()
 test_img=np.array(load_test_data())
 signnames=load_label()
```
### 2.预测
```python
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
[inference_program, feed_target_names,fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)
results = []
for i in range(test_img.shape[0]):
    with fluid.program_guard(inference_program):
        res = exe.run(inference_program,feed={feed_target_names[0]:test_img[i]},fetch_list=fetch_targets)
        results.append(np.argmax(res[0]))
        print("infer results[%d] = %d -> %s " % (i,results[i],signnames[results[i]]))
```
```
infer results[0] = 13 -> Yield 
infer results[1] = 34 -> Turn left ahead 
infer results[2] = 14 -> Stop 
infer results[3] = 4 -> Speed limit (70km/h) 
infer results[4] = 12 -> Priority road 
infer results[5] = 18 -> General caution 
infer results[6] = 35 -> Ahead only 
infer results[7] = 17 -> No entry 
```
## 四、效果展示
```python

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
```
![image](https://user-images.githubusercontent.com/69491295/128953009-4aeeca31-2bfa-4231-bdb2-a79936b0c346.png)
如上图训练的模型很好的实现了对交通标志的分类。

## 五、总结与升华

写写你在做这个项目的过程中遇到的坑，以及你是如何去解决的。

最后一句话总结你的项目

## 个人简介

此处可附上你的AI Studio个人链接，增加曝光率。
