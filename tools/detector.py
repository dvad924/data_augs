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
import time

class MyCaffeNet:
    def __init__(self,net_arch,weights,mode,mean=None,scale=256,shape=None,procmode='cpu'):
        self.arch = net_arch
        self.weights = weights
        self.mode = mode
        self.meanfile = mean
        self.scale = scale
        if self.meanfile:
            self.set_mean()
        if shape:
            self.prepare_net(shape=shape,procmode=procmode)
        else:
            self.prepare_net(procmode=procmode)
        self.set_transform()
        
    def prepare_net(self,shape=(1,3,128,128),procmode='cpu'):
        if(procmode == 'cpu'):
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
        self.net = caffe.Net(self.arch,self.weights,self.mode)
        depth,channels,height,width = shape
        self.net.blobs['data'].reshape(depth,channels,height,width)
    
    def get_depth(self):
        depth,channels,height,width = self.net.blobs['data'].shape
        return depth

    def get_shape(self):
        return  self.net.blobs['data'].shape

    def set_shape(self,depth,channels,height,width):
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
        batchsize,c,h,w = self.get_shape()
        if len(imgs) < batchsize:
            self.set_shape(len(imgs),c,h,w)
        s = time.clock()
        self.set_imgs(imgs) #set the images into the network batch
        print 'batch set time = {}'.format(time.clock() - s)
        s = time.clock()
        self.net.forward()  #run the batch through the network
        print 'process time = {}'.format(time.clock() - s)
        clas = self.net.blobs[cfg.out].data.argmax(1) #grab the assigned class labels
        prob = self.net.blobs['cls_prob'].data.max(1) # grab the associated probablities
        self.set_shape(batchsize,c,h,w)
        return clas,prob
    
    def propose_and_detect(self,imgname,logfile=None):
        img,props = self.load_and_propose(imgname)
        rects = map(lambda x:x['rect'],props)
        images = [img[y:y+h,x:x+w,:] for x,y,w,h in rects]
        batchsize,c,h,w = self.net.blobs['data'].data.shape
        goodrects = []
        for i in xrange(0,np.ceil(float(len(images)/(batchsize*1.0)))):
            l = i * batchsize
            u = min((i+1) * batchsize,len(images))
            batch = images[l:u]
            clas,prob =  self.run_batch(batch)
            batchrects = rects[l:u]
            idx = clas == 1
            goodrects += batchrects[idx]

        return img,goodrects;            
        
            
        
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('net')
    parser.add_argument('weights')
    parser.add_argument('--mean')
    parser.add_argument('--scale')
    parser.add_argument('--image')
    parser.add_argument('--dir')
    parser.add_argument('--list')
    parser.add_argument('--procmode')
    parser.add_argument('--outfile')
    return parser.parse_args()

def calc_class_accuracy(gt_class,calc_class,indicator_groups):
    # implement
    # gt_classs
    gt_mapped = np.zeros(len(gt_class))
    calc_class_mapped = np.zeros(len(calc_class))
    for ix,group in enumerate(indicator_groups):
        gt_mapped[np.in1d(gt_class,group)] = ix
        calc_class_mapped[np.in1d(calc_class,group)] = ix

    length = float(len(gt_class)*1.0)
    total = float(np.sum(gt_mapped == calc_class_mapped)*1.0)
    
    acc = total/length
    return acc

        

def test_batch(net,dir,infile,outfile,clasmap=[(0,),(1,),(2,)]):
    files = []
    classes = []
    logfile = outfile+'.log'
    lfile = open(logfile,'a')
    with open(infile) as fil:
        for line in fil.readlines():
            fields = line.split(' ')
            files.append(fields[0].strip())
            if fields[1].strip():
                classes.append(int(fields[1].strip()))
    accs = np.zeros((len(files),))
    daccs = np.zeros((len(files),))
    probs = np.zeros((len(files),))
    bsize = net.get_depth()
    if len(classes) > 0:
        daccs = np.array(classes)
        usename = False
    batches = range(0,len(files),bsize)
    for ix, pos in enumerate(batches):
        s = time.clock()
        ulimit = batches[ix+1] if ix+1 < len(batches) else len(files)
        if ulimit == len(files):
            pdb.set_trace()
        batch = [net.load_image(os.path.join(dir,imgname)) for imgname in files[pos:ulimit]]
        print 'batch load time = {}'.format(time.clock() - s)
       
        clas,prob = net.run_batch(batch)
       
        accs[pos:ulimit] = clas
        probs[pos:ulimit] = prob
        acc_est = calc_class_accuracy(daccs[pos:ulimit],clas,clasmap)
        print 'Batch Accuracy: {}'.format(acc_est)
        
    pdb.set_trace()
    final_acc = calc_class_accuracy(daccs,accs,clasmap)
    val = 'final acc:{}'.format(final_acc)
    print val

    with open(outfile,'w') as fil:
        fil.write('{}\n'.format(val))
    
    

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

def run_test(args,shape=(64,3,128,128),clas_clusters=[(0,),(1,)]):
    procmode = 'gpu' if args.procmode is None or args.procemode =='gpu' else 'cpu'
    mynet = MyCaffeNet(args.net,args.weights,caffe.TEST,mean=args.mean,
                       shape=shape,procmode=procmode)
    imdir = args.dir
    flist = args.list
    outfile = args.outfile
    test_batch(mynet,imdir,flist,outfile,clas_clusters)

if __name__ == '__main__':
    import pdb
    args = parseArgs()

    run_test(args)
    # mynet = MyCaffeNet(args.net,args.weights,caffe.TEST,mean=args.mean,shape=(64,3,128,128),procmode='gpu')
    # ###########################random objects test###############
    # dirr = 'data/person/image_patches/'
    # assigndir = 'data/person/assign/person_vs_background_vs_random_test.txt'
    # #test(dirr,assigndir,'p_v_b_v_r_results.txt')
    # test_batch(dirr,assigndir,'p_v_b_v_r_batch_br_merge_results.txt',[(0,2),(1,)])
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

    
            
