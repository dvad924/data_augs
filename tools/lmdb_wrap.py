import numpy as np
import lmdb
import os
import sys
import argparse
import re
sys.path.insert(0,'/home/dl/DVDPL/Caffe/caffe/python')

import caffe


class DbWrapper:
    def __init__(self,db_path):
        self.path = db_path
        self.env = None
        if not os.path.exists(self.path):
            print 'DB path does not exist, terminating ...'
            sys.exit()
        self.id_prefix_patt = re.compile('[0-9]{8}')
        
    def open_write(self):
        self.env = lmdb.open(self.path,map_size=int(1e12))
        
    def copy(self,dst):
        if not self.env:
            self.open_read()
        if not os.path.exists(dst):
            os.makedirs(dst)
        else:
            print 'directory exists.. exiting'
            sys.exit()


    def open_read(self):
        self.env = lmdb.open(self.path,readonly=True)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            
    def loop(self,action):
        if not self.env:
            self.open_write()
        dbkeys = []
        last_key,_ = self.get_last()
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for k,v in cursor:
                dbkeys.append(k)

            for k in dbkeys:
                last_key = action(k,txn.get(k),last_key)
                
    def print_keys(self):
        if not self.env:
            self.open_read()

        with self.env.begin() as txn:
            for k,v in txn.cursor():
                print k
    
    def keysplit(self,key):
        prefix = re.findall(self.id_prefix_patt,key)[0]
        return key.replace(prefix,'')

    def pkform(self,key):
        return '{:08}'.format(key)

    def get_last(self):
        if not self.env:
            self.open_read()
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            cursor.last()
            id = cursor.key()
            keyval = re.findall(self.id_prefix_patt,id)[0]
            print keyval
            return int(keyval),cursor.value()
    
    def insert(self,id,value):
        if not self.env:
            self.open_write()
        with self.env.begin(write=True) as txn:

            txn.put(id,value,dupdata=False,
                    overwrite=False,append=True)
            
        
    def decode_caffe_img_string(self,value):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        flat_img = np.fromstring(datum.data, dtype=np.uint8)
        img = flat_img.reshape(datum.channels, datum.height, datum.width)
        label = datum.label
        return (label,img)

    def encode_caffe_img_string(self,label,img):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = img.shape[0]
        datum.height = img.shape[1]
        datum.width = img.shape[2]
        datum.data = img.tobytes()

        datum.label = int(label)
        return datum.SerializeToString()


class DataAugmentor:
    def __init__(self,lmdb):
        self.lmdb = DbWrapper(lmdb)
        


    def color_aug(self,img):
        imgc = img.copy()
        flag = np.random.randint(1,8)# 001,010,011,100,101,110,111
        channel_vec = np.ones((3,1,1))
        channel_vec[0,:,:] = flag & 1
        channel_vec[1,:,:] = (flag >> 1) & 1
        channel_vec[2,:,:] = (flag >> 2) & 1
        rand_vec = np.random.randint(-20,21,(3,1,1))
        signal_aug = channel_vec * rand_vec
        imgc = imgc + signal_aug
        imgc = imgc.clip(min=0,max=255)
        
        return imgc.astype(np.uint8)

    def aug_records(self,action):
        self.lmdb.loop(action)

    def on_record(self,key,value,lastkey):
        newkey = lastkey+1
        label,img = self.lmdb.decode_caffe_img_string(value)
        imgNew = self.color_aug(img)
        valueNew = self.lmdb.encode_caffe_img_string(label,imgNew)
        keystring =  self.lmdb.pkform(newkey) + self.lmdb.keysplit(key)
        print keystring
        self.lmdb.insert(keystring,valueNew)
        return newkey
        
        
if __name__ == '__main__':
    dest= 'data/person_only_lmdb/people_color_patch_train_lmdb'
    src = 'data/person_only_lmdb/people_patch_train_lmdb'
    lmdb.open(src).copy(dest)
    da = DataAugmentor(dest)
    da.aug_records(da.on_record)
    
