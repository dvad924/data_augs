import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import detector as DET
import os
import cv2

savedir = 'tests/detections/'
personclass = 1;
def write_det_rez(rects,fname):
    
    with open(fname,'w') as f:
        for (x,y,w,h) in rects:
            f.write('({},{},{},{}) {}\n'.format(x,y,w,h,personclass))

def detect_in_file_with_net(mfile,net):
    path,fil = os.path.split(mfile)
    basefile = os.path.splitext(fil)[0]
    rezfile = os.path.join(savedir,basefile + '_detection_rez.jpg')
    detsfile = os.path.join(savedir,basefile + '_detections.txt')

    img,rects,probs = net.propose_and_detect(mfile)
    
    im = img[:,:,(2,1,0)]
    im *= 255
    im = im.astype(np.uint8)
    pimg = im.copy()
    for i,(x,y,w,h) in enumerate(rects):
        cv2.rectangle(pimg,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.putText(pimg,'{:s} {:.3f}'.format('person',probs[i]),
                    (x,max(y-2,0)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
        cv2.imwrite(rezfile,pimg)

    write_det_rez(rects,detsfile)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import skimage.data

    net = DET.MyCaffeNet('nets/person_vs_background_vs_random/prod.prototxt',
                   'models/person_vs_background_vs_random/person_vs_background_vs_random_lr_0.00001_iter_100000.caffemodel',
                   mean='data/person_only_lmdb/person_vs_background_vs_random_color_mean.binaryproto',
                         procmode='gpu', shape=(64,3,128,128))
    filedir = '/home/dl/DVDPL/Caffe/caffe/data/person_clsfy_data/actions3/orgdata/images/'
    for fname in os.listdir(filedir):
        if os.path.splitext(fname)[1] == '.jpg':
            print fname
            detect_in_file_with_net(os.path.join(filedir,fname),net)
            
    #detect_in_file_with_net(os.path.join(filedir,'3227.jpg'),net)
            
    # img,rects = net.propose_and_detect('/home/dl/DVDPL/Caffe/caffe/data/person_clsfy_data/actions3/orgdata/images/1001.jpg')

    
    # fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6,6))
    # ax.imshow(img)
    # for x, y, w, h in rects:
    #     r = mpatches.Rectangle((x,y),w,h,
    #                            fill=False,
    #                            edgecolor='red',
    #                            linewidth=1)
    #     ax.add_patch(r)

    # plt.savefig('imgdisplay.png')
