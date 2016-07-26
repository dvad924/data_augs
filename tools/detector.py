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
import skimage.data

class MyCaffeNet:
    def __init__(self,net_arch,weights,mode,mean=None,scale=256,shape=None):
        self.arch = net_arch
        self.weights = weights
        self.mode = mode
        self.meanfile = mean
        self.scale = scale
        if self.meanfile:
            self.set_mean()
        if shape:
            self.prepare_net(shape=shape)
        else:
            self.prepare_net()
        self.set_transform()
        
    def prepare_net(self,shape=(1,3,128,128)):
        caffe.set_mode_cpu()
        self.net = caffe.Net(self.arch,self.weights,self.mode)
        depth,channels,height,width = shape
        self.net.blobs['data'].reshape(depth,channels,height,width)
        
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
    
    def load_and_propose(self,imgname):
        img = caffe.io.load_image(imgname);
        #this will be a time bottleneck
        props = DU.selective_window(img)
        return img,props
    def set_img(self,img):
        self.net.blobs['data'].data[...] = img

    def set_imgs(self,imgs):
        for i,img in enumerate(imgs):
            self.net.blobs['data'].data[i,...] = img
            
    def run_image(self,imgname,logfile=None):
        img = self.load_image(imgname)
        self.set_img(img)
        self.net.forward()
        fname = imgname.split('/')[-1]
        clas = self.net.blobs[cfg.out].data.argmax(1)
        prob = self.net.blobs['cls_prob'].data
        val = '{} class:{}, prob:{}\n'.format(fname,clas,prob)
        print val
        if logfile:
            logfile.write('{}\n'.format(val))
        return clas[0]

    def run_batch(self,imgs,logfile=None):
        self.set_imgs(imgs) #set the images into the network batch
        self.net.forward()  #run the batch through the network
        clas = self.net.blobs[cfg.out].data.argmax(1) #grab the assigned class labels
        prob = self.net.blobs['cls_prob'].data
        return clas,prob
    
    def propose_and_detect(self,imgname,logfile=None):
        img,props = self.load_and_propose(imgname)
        rects = map(lambda x:x['rect'],props)
        images = [img[y:y+h,x:x+w,:] for x,y,w,h in rects]
        batchsize,c,h,w = self.net.blobs['data'].data.shape
        for i in xrange(0,np.ceil(float(len(images)/(batchsize*1.0)))):
            l = i * batchsize
            u = min((i+1) * batchsize,len(images))
            batch = images[l:u]
            if len(batch) < batchsize:
                self.net.blobs['data'].reshape(len(batch),c,h,w)
            clas,prob =  self.run_batch(batch)
        
        
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

def test(dirr,infile,outfile,clasmap={'__background':0,'person':1}):
    files2 = []
    classes = []
    logfile = outfile+'.log'
    lfile = open(logfile,'a')
    with open(infile) as fil:
        for line in fil.readlines():
            fields = line.split(' ')
            files2.append(fields[0].strip())
            if fields[1].strip():
                classes.append(int(fields[1].strip()))

    accs = np.zeros((len(files2),))
    daccs = np.zeros((len(files2),))
    if len(classes) > 0:
        daccs = np.array(classes)
        usename = False
    for i ,f in enumerate(files2):
        accs[i] = mynet.run_image(os.path.join(dirr,f),logfile=lfile)
   
        if usename:
            for key in clasmap.keys():
                if f.startswith(key) :
                    daccs[i] = clasmap[key]
                    break
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
    ###########################random objects test###############
    dirr = '/Users/dvad/image_patches/'
    assigndir = '/Users/dvad/image_patches/assign/random.txt'
    test(dirr,assigndir,'randomresults.txt',)
    ###########################peta data test######################
    # dirr = '/Volumes/DVBackBox/PetaData/PETA/orgdata/images'
    # assigndir = '/Volumes/DVBackBox/PetaData/PETA/orgdata/assign'
    # test(dirr,os.path.join(assigndir,'labels.txt'),'petaresults.txt')
    ###########################Training Set Test###################
    # dirr = '/Users/dvad/image_patches/'
    # assigndir = '/Users/dvad/image_patches/assign'
    # test(dirr,os.path.join(assigndir,'person_only_train.txt'),'train.txt')
    #train val
    ###########################Testing Set Test####################
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

    
            
