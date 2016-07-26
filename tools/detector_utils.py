import numpy as np
import os
import cv2
from selectivesearch import selective_search
testdir = '/Users/dvad/code/deepLearning/peta/src/actions3/orgdata/images/'


def sliding_window(img,window=(138,60),hstep=25,wstep=25):
    #get image dimmensions
    rows,cols,channels =  img.shape
    #get dimensions of the window to use
    height,width = window
    #get starting coors of the window save final boarders
    ys = np.arange(0,rows-height-1,hstep)
    xs = np.arange(0,cols-width-1,wstep)

    #tack the final possible positions on the end
    ys = np.append(ys,rows-height-1)
    xs = np.append(xs,cols-width-1)
    # this will scan from left 2 right,
    # not the fastest implementation but should be plenty for this application
    points = [(y,x) for y in ys for x in xs]
    

    frames = [img[y:y+height,x:x+width] for (y,x) in points]

    return frames
    
def selective_window(img,scale=300,sigma=0.8,mins=20):
    #this is a time bottleneck
    im_lables,regions = selectivesearch(img,scale=scale,sigma=sigma,min_size=mins);

    true_regions = set()
    for reg in region:
        if reg in true_regions:
            continue
        if reg['size'] < 100:
            continue
        true_regions.add(reg['rect'])
    return true_regions



if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    img = cv2.imread(os.path.join(testdir,'1001.jpg'))
    stack = sliding_window(img,(138,60),25,25)
    print len(stack)
    for im in stack:
        cv2.imshow('win1',im)
        cv2.waitKey(33)
