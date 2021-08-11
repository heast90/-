#!/usr/bin/env python
# coding: utf-8
import paddle.fluid as fluid
import paddle 
import numpy as np 
import cv2 
import csv 
from paddlehub.module.module import runnable, moduleinfo, serving
import base64
from io import BytesIO
import json
import argparse
def load_image(file):
    image = cv2.imread(file)
    image = np.array(image).astype(np.float32).reshape(1, 3, 32, 32)#Batch大小，图片通道数，图片的宽和图片的高
    image = (image - np.mean(image))/(np.max(image) - np.min(image))
    return image
def load_label():
    file_path = './data/data103558/signnames.csv'
    signnames = []
    with open(file_path) as f:
        lines = csv.reader(f,delimiter=',')
        next(lines)
        for line in lines:
            signnames.append(line[1])
    return signnames
class Model:
    def __init__(self,use_cuda,params_dirname,describle):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        paddle.enable_static() 
        self.params_dirname=params_dirname
        self.describle = describle
         
    def predict(self,img):
        place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        [inference_program, feed_target_names,fetch_targets] = fluid.io.load_inference_model(self.params_dirname, exe)
        res = []
        with fluid.program_guard(inference_program):
            r = exe.run(inference_program,feed={feed_target_names[0]:img},fetch_list=fetch_targets)
            res.append(np.argmax(r[0]))
        self.result = res[0]
        return np.array(res[0]).astype('int32')
    def output(self):
        print("infer results = %d -> %s\n" % (self.result,self.describle[self.result]))
    def signame(self):
        return self.describle[int(self.result)]
@moduleinfo(
    name="my_model",
    version="1.0.0",
    summary="This is a PaddleHub Module. Just for test.",
    author="7hinc",
    author_email="",
    type="cv/my_model",
)
class ModelPredict:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Run the mnist_predict module.",
            prog='hub run mnist_predict',
            usage='%(prog)s',
            add_help=True)
        self.parser.add_argument(
            '--input_img', type=str, default=None, help="img to predict")
        self.signnames = load_label()
        self.params_resnet = './image_classification_resnet.inference.model'
        self.params_vgg = './image_classification_vgg.inference.model'
        self.params = self.params_resnet
        self.model = Model(False,self.params,self.signnames)

    def model_predict(self, img_path):
        print('forward')
        img = load_image(img_path)
        res=self.model.predict(img)
        self.model.output()
        return res 
    
    @runnable
    def runnable(self, argvs):
        print('runnable')
        args = self.parser.parse_args(argvs)
        return self.model_predict(args.input_img)

    @serving
    def serving(self, img_b64):
        print('serving')
        model.eval()
        img_b = base64.b64decode(img_b64)
        img = load_image(BytesIO(img_b))
        result = ["%d -> %s\n"%(self.model(img),self.signname())]
        self.model.output()
        # 应该返回JSON格式数据
        # 从numpy读出的数据格式是 numpy.int32
        res = { 'res': np.array(result)}
        return json.dumps(res)

# 实例化应该全局
if __name__=="__main__":
my_model = ModelPredict()
