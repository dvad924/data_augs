import numpy as np
import sys
sys.path.insert(0,'/home/dl/DVDPL/Caffe/caffe/python')
import cv2 
import caffe
import matplotlib.pyplot as plt
import os
import detector_utils as DU
import argparse
import detconfig as cfg
class MyCaffeNet:
    def __init__(self,net_arch,weights,mode,mean=None,scale=256):
        self.arch = net_arch
        self.weights = weights
        self.mode = mode
        self.meanfile = mean
        self.scale = scale
        if self.meanfile:
            self.set_mean()
        self.prepare_net()
        self.set_transform()
        
    def prepare_net(self):
        caffe.set_mode_cpu()
        self.net = caffe.Net(self.arch,self.weights,self.mode)
        self.net.blobs['data'].reshape(1,3,128,128)
        
    def set_mean(self):
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.ParseFromString(open(self.meanfile,'rb').read())
        mean = caffe.io.blobproto_to_array(blob)
        shp = mean.shape
        mean = mean.reshape(shp[1],shp[2],shp[3])
        self.mu = mean
        
    def set_transform(self):
        transformer = caffe.io.Transformer({'data':self.net.blobs['data'].data.shape})
        transformer.set_transpose('data',(2,0,1))
        transformer.set_raw_scale('data',255)
        transformer.set_mean('data',self.mu)
        transformer.set_input_scale('data',float(1.0/256))
        transformer.set_channel_swap('data',(2,1,0))
        self.transformer = transformer
        
    def load_image(self,imgname):
        img = caffe.io.load_image(imgname)
        return self.transformer.preprocess('data',img)
        
    def set_img(self,img):
        self.net.blobs['data'].data[...] = img

    def run_image(self,imgname):
        img = self.load_image(imgname)
        self.set_img(img)
        self.net.forward()
        fname = imgname.split('/')[-1]
        clas = self.net.blobs[cfg.out].data.argmax(1)
        prob = self.net.blobs['cls_prob'].data
        print '{} class:{}, prob:{}\n'.format(fname,clas,prob)
        return clas[0]
    
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('net')
    parser.add_argument('weights')
    parser.add_argument('--mean')
    parser.add_argument('--scale')
    parser.add_argument('--image')
    parser.add_argument('--dir')
    parser.add_argument('--list')
    return parser.parse_args()

def test(dirr,infile,outfile,neg='__background',pos='person'):
    files2 = []    
    with open(infile) as fil:
        for line in fil.readlines():
            fields = line.split(' ')
            files2.append(fields[0].strip())

    accs = np.zeros((len(files2),))
    daccs = np.zeros((len(files2),))
    
    for i ,f in enumerate(files2):
        accs[i] = mynet.run_image(os.path.join(dirr,f))
        if f.find(neg)>=0 :
            daccs[i] = 0
        elif f.find(pos) >=0 :
            daccs[i] = 1
        print i, daccs[i]
        
    pdb.set_trace()
    val = 'final acc:{}'.format(np.sum(accs==daccs)/(len(files2)*1.0))
    print val
    
    with open(outfile,'w') as fil:
        fil.write('{}\n'.format(val))

if __name__ == '__main__':
    import pdb
    args = parseArgs()
    mynet = MyCaffeNet(args.net,args.weights,caffe.TEST,args.mean)
    dirr = '/Users/dvad/image_patches/'
    assigndir = '/Users/dvad/image_patches/assign'
    test(dirr,os.path.join(assigndir,'person_only_train.txt'),'train.txt')
    #train val
    # with open(dirr+'assign/person_only_test.txt') as fil:
    #     for lin in fil.readlines():
    #         fields = lin.split(' ')
    #         files.append(fields[0].strip())
            
    # accs = np.zeros((len(files),))
    # daccs = np.zeros((len(files),))
    # pdb.set_trace()
    # for i,f in enumerate(files):
    #     print i
    #     accs[i] = mynet.run_image(os.path.join(dirr,f))
    #     if f.find('__background') >=0 :
    #         daccs[i] = 0
    #     elif f.find('person') >=0 :
    #         daccs[i] = 1
    # pdb.set_trace()
    # val = 'final acc:{}'.format(np.sum(accs==daccs)/(len(files)*1.0))
    # print val

    # with open('test.txt','w') as fil:
    #     fil.write('{}\n'.format(val))

    
            
