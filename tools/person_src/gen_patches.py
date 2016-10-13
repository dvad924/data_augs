import os
import sys
import cv2
import numpy as np
import annotate_data
import pdb
import re


class FrameParse:
    def __init__(self,anndir,imdir,inputfile,assignfile,outdir):
        self.annotationsdir = anndir
        self.imagedir = imdir
        self.assignmentfile = assignfile
        self.outdir = outdir
        self.inputfile= inputfile
        self.exts = ['.jpg']

    def unpack_box(self,box):
        return (box[0],box[1],box[2],box[3])

    def grab_patch(self,box,img):
        x1,y1,w,h = [int (v) for v in self.unpack_box(box)]
        return img[y1:y1+h,x1:x1+w]

    def grab_overlaps(self,box,imgdims):
        maxWidth, maxHeight = imgdims
        #compute the 8 surrounding boxes around the gt with IoU > 0.5
        bx,by,bw,bh = box
        height_factor = int(bh*0.25)
        width_factor = int(bw*0.25)
        
        patches = []
        #four cardinal directions
        #up
        N_patch = [bx,max(0,by - height_factor),bw, bh + min(0,by - height_factor)]
        #down
        S_patch = [bx,by + height_factor, bw, by + min(0,by+height_factor - maxHeight)]
        #left 
        W_patch = [max(0,bx - width_factor),by,bw + min(0,bx - width_factor),bh]
        #right 
        E_patch= [bx + width_factor, by, bw + min(0,bw + width_factor - maxWidth)]

        #northeast
#        NE_patch = [bx + 

 #       patches.append([bx,int(by*0.25)
        
        

    def grab_patches(self,boxes,labels,img):
        patches = []
        for label,box in zip(labels,boxes):
            patch = self.grab_patch(box,img)
            overlaps = self.grab_overlaps
            patches.append((label,patch))
        return patches

    def does_overlap(self,box,boxes):
        xmin,ymin,xmax,ymax = self.unpack_box(box)
        
        for bbox in boxes:
            x1,y1,x2,y2 = self.unpack_box(bbox)
            if not (xmin > x2 or xmax < x1 or ymin > y2 or ymax < y1):
                return True
        return False

    def tile(self,nparr1,nparr2):
        return np.array(np.meshgrid(nparr1,nparr2)).T.reshape(-1,2)

    def random_neg_boxes(self,boxes,img,nmin):
        ymax,xmax,nchannels = img.shape
        neg_patches = []
        for step in range(2,8):
            xpoints = np.linspace(0,xmax,step,endpoint=False)
            ypoints = np.linspace(0,ymax,step,endpoint=False)
            grid = self.tile(xpoints,ypoints)
            height =ymax/step
            width = xmax/step
            for start in grid:
                x1=start[0]
                y1=start[1]
                box = [x1,y1,x1+width,y1+height]
                if not self.does_overlap(box,boxes):
                    neg_patches.append(box)
            if len(neg_patches) >= nmin:
                break
            else:
                neg_patches = []
        if len(neg_patches)  <= nmin:
            return neg_patches
        else:
            np.random.shuffle(neg_patches)
            return neg_patches[:nmin]
        
    def classmap(self,lab):
        if lab =='person':
            return '1'
        if lab =='car':
            return '2'
        if lab =='__background__':
            return '0'
        else:
            return '3'

    def write_patches(self,fname,patches):
        fs = fname.split('/')
        f = os.path.splitext(fs[0])[0]
        for ix, (lab,patch) in enumerate(patches):
            filename = '{}_{}_{}.jpg'.format(lab,ix,f)
            cv2.imwrite(os.path.join(self.outdir,filename),patch)
            with open(self.assignmentfile,'a') as ff:
                ff.write('{} {}\n'.format(filename,self.classmap(lab)))

    def parse_image(self,fname):
        labels,bboxs = annotate_data._grab_annotation(self.annotationsdir,fname.replace('.seq',''))
        img = cv2.imread(os.path.join(self.imagedir,fname))
        pospatches = self.grab_patches(bboxs,labels,img)
        negboxes = self.random_neg_boxes(bboxs,img,2)
        nlabels = ['__background__' for x in negboxes]
        negpatches = self.grab_patches(negboxes,nlabels,img)
        patches = pospatches + negpatches
        return patches

    def getimage_names(self):
        imageNames = []
        with open(self.inputfile) as f:
            for line in f:
                name = line.strip()
                i = 0
                for ext in self.exts:
                    if os.path.exists(os.path.join(self.imagedir,name + ext)):
                        imageNames.append(name+ext)
                    else:
                        i+=1
                if i == 2:
                    print name
        return imageNames

    def parse_images(self):
        imageNames = self.getimage_names()
        for name in imageNames:
            patches = self.parse_image(name)
            self.write_patches(name,patches)

if __name__ == '__main__':
    pdb.set_trace()
    patt = re.compile(r'V[0-9]{3}')
    dirs
    for p,dirs,files in os.walk('../annotations/'):
        if len(files) == 0:
            for d in dirs:
                if patt.match(d):

                    fp = FrameParse('../annotations/',
                                    '../images/',
                                    '../assign/test.txt',
                                    'assign/patch_test.txt',
                                    '../pedestrian_patches/images')
                    fp.parse_images()

                    fp1 = FrameParse('../annotations/',
                                     '../images/',
                                     '../assign/train.txt',
                                     'assign/patch_train.txt',
                                     '../pedestrian_patches/images')

                    fp1.parse_images()

            
