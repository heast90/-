#!/usr/bin/env python
# coding: utf-8
#加载包
import pickle 
import paddle.fluid as fluid
import paddle 
import numpy as np 
import matplotlib.pyplot as plt 
#数据包加载,Feed格式转换
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

#模型构建
#训练参数
flag =True
class_size= 43 if flag else 10
BATCH_SIZE = 150
learning_rate=0.01
EPOCH_NUM = 50

#VGG

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


#ResNet
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


#训练搭建

#训练前配置
def inference_program():
    # The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    paddle.enable_static()
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    predict = resnet_cifar10(images, 32)
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

#训练程序
#Data Feeders 配置
# Reader for training
train_reader = paddle.batch(
    paddle.reader.shuffle(trainDataReader,buf_size=39209),
    batch_size=BATCH_SIZE)
# Reader for testing. A separated data set for testing.
test_reader = paddle.batch(testDataReader, 
                           batch_size=BATCH_SIZE)


#Trainer 程序的实现
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


# 训练主循环

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

if __name__=="__main__":
    train_loop()
    print('训练模型保存完成！')
    print("画出训练测试误差图")
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
