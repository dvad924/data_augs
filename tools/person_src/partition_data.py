import os
import argparse
import cv2
import numpy as np
import re


tdir = 'assign/'

def fitspattern(pattern):
    pat = re.compile(pattern)
    def fits(string):
        return pat.search(string) is not None
    return fits
person_prefix = [fitspattern('^person_[0-9]+_[0-9]{1,4}\.'),fitspattern('person_[0-9]+_person'),fitspattern('^person_[0-9]+_[0-9]{6}\.') ]
back_prefix = map(fitspattern,['__background__',
                               'aeroplane',
                               'bicycle',
                               'bird',
                               'boat',
                               'bottle',
                               'bus',
                               'cat',
                               'chair',
                               'cow',
                               'diningtable',
                               'dog',
                               'horse',
                               'motorbike',
                               'pottedplant',
                               'sheep',
                               'sofa',
                               'train',
                               'tvmonitor',])


#this function will create a list of lists based on file name prefix
def sift_data(fnames,prefixes):
    datagroups = []
    for prefix_check in prefixes:
        sat = filter(prefix_check,fnames)
        datagroups.append(sat)
    return datagroups    

def part_group(group,numtrain):
    tot = len(group) - 1
    trp = numtrain
    
    train = group[:trp]
    test = group[trp:]
    return train,test



def partition_groups(groups,maxlen,train_part=0.66,factors=0):
    train = []
    test = []
    if factors == 0:
        factors = np.array(map(lambda x:len(x),groups),dtype=np.float32)
        factors = train_part*factors
        tot = np.sum(factors)
        if tot > maxlen:
            factors *= maxlen/tot
        
    for (factor,group) in zip(factors,groups):
        tr,te = part_group(group,int(np.ceil(factor)))
        train += tr
        test += te
    return train,test


def readfiles(fname):
    data = []
    with open(fname) as f:
        for line in f.readlines():
            fields = line.split(' ');
            data.append(fields[0].strip())
    return data

def writelabels(labelfile,data):
    with open(labelfile,'w') as f:
        for line in data:
            f.write('{}\n'.format(line))

def assign_class(clas):
    def assigner(prefix):
        return '{} {}'.format(prefix,clas)
    return assigner

def move_data(datalen,data):
    trainmax = int(np.floor(trainfactor*mlen))
    testmax = int(np.ceil(testfactor * mlen)) + trainmax
    traindata = data[:trainmax]
    testdata  = data[trainmax:testmax]

    return (traindata,testdata)

def person_vs_background_vs_random(trainfactor=0.66,testfactor=0.34):
    #open the file containing person filenames
    persons = readfiles(os.path.join(tdir,'person.txt'))
    #open the file containing bacground filenames
    background = readfiles(os.path.join(tdir,'backgroundshuffle.txt'))
    #open the file containing random object filenames
    random = readfiles(os.path.join(tdir,'random.txt'))
    
    persons = map(assign_class('1'),persons)
    background = map(assign_class('0'),background)
    random = map(assign_class('2'),random)

    persond = sift_data(persons,person_prefix)
    ptr,pte = partition_groups(persond,len(persons))
    
    randomd = sift_data(random,back_prefix)
    randcontr_size = len(ptr)
    rtr,rte = partition_groups(randomd,randcontr_size)
    
    backgroundd = sift_data(background,back_prefix)
    backcontr_size = len(ptr)
    btr,bte = partition_groups(backgroundd,backcontr_size)

    train = ptr + rtr + btr
    test  = pte + rte + bte

    writelabels(os.path.join(tdir,'person_vs_background_vs_random_train.txt'),train);
    writelabels(os.path.join(tdir,'person_vs_background_vs_random_test.txt'),test);



def person_vs_backgroundandrandom_together(trainfactor=0.66,testfactor=0.34):
    #open the file containing person filenames
    persons = readfiles(os.path.join(tdir,'person.txt'))
    #open the file containing background filenames
    background = readfiles(os.path.join(tdir,'backgroundshuffle.txt'))
    #open the file containing random object filenames
    random = readfiles(os.path.join(tdir,'random.txt'))
    import pdb
    pdb.set_trace()
    persons = map(assign_class('1'),persons)
    background = map(assign_class('0'),background)
    random = map(assign_class('0'),random)

    persond = sift_data(persons,person_prefix)
    ptr,pte = partition_groups(persond,len(persons))
    
    randomd = sift_data(random,back_prefix)
    randcontr_size = len(ptr)/2.0
    rtr,rte = partition_groups(randomd,randcontr_size)
    
    backgroundd = sift_data(background,back_prefix)
    backcontr_size = len(ptr) - randcontr_size
    btr,bte = partition_groups(backgroundd,backcontr_size)

    train = ptr + rtr +btr
    test = pte + rte + bte

    writelabels(os.path.join(tdir,'person_vs_backgroundandrandom_train.txt'),train);
    writelabels(os.path.join(tdir,'person_vs_backgroundandrandom_test.txt'),test);
    

def person_vs_backgroundonly(trainfactor=0.66,testfactor=0.34):
    #open the file containing person filenames
    persons = readfiles(os.path.join(tdir,'person.txt'));
    background =  readfiles(os.path.join(tdir,'backgroundshuffle.txt'));
    import pdb
    pdb.set_trace()

    background = map(assign_class('0'),background)
    persons = map(assign_class('1'),persons)

    persond = sift_data(persons,person_prefix)
    ptr,pte = partition_groups(persond,len(persons))

    backgroundd = sift_data(background,back_prefix)
    backcontr_size = len(ptr)
    btr,bte = partition_groups(backgroundd,backcontr_size)


    train = ptr + btr
    test = pte + bte

    writelabels(os.path.join(tdir,'person_vs_backgroundonly_train.txt'),train);
    writelabels(os.path.join(tdir,'person_vs_backgroundonly_test.txt'),test);

    
    
if __name__ == '__main__':
    #person_vs_backgroundonly();
    #person_vs_backgroundandrandom_together();
    person_vs_background_vs_random()
