import json
import argparse
import numpy as np
import cv2
import os


class ImageAugmentor:

    def __init__(self,settingsfile,img):
        self.settingsname = settingsfile
        if not os.path.isfile(settingsfile):
            print 'settings file does not exist terminating...\n'
        with open(settingsfile) as f:
            self.settings = json.loads(f.read())
            
    
    def flipAug(self,img,theta):
        if theta == 90:
            return cv2.transpose(1,0,2)
        if theta == 180:
            return cv2.tran
    
