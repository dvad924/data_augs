import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import detector as DET

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import skimage.data

    net = DET.MyCaffeNet('nets/person_vs_background_vs_random/prod.prototxt',
                   'models/person_vs_background_vs_random/person_vs_background_vs_random_lr_0.00001_iter_100000.caffemodel',
                   mean='data/person_only_lmdb/person_vs_background_vs_random_color_mean.binaryproto',
                         procmode='gpu', shape=(64,3,128,128))

    img,rects = net.propose_and_detect('/home/dl/DVDPL/Caffe/caffe/data/person_clsfy_data/actions3/orgdata/images/1001.jpg')

    
    fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6,6))
    ax.imshow(img)
    for x, y, w, h in rects:
        r = mpatches.Rectangle((x,y),w,h,
                               fill=False,
                               edgecolor='red',
                               linewidth=1)
        ax.add_patch(r)

    plt.savefig('imgdisplay.png')
