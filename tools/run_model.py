caffe_root = '/home/dl/DVDPL/Caffe/caffe/'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
import os
import numpy as np
import argparse as ag
import re
import time


solver = None

class MySolver:
    def __init__(self,net_arch):
        self.solverfile = os.path.join('nets',net_arch,'solver.prototxt')
        self.trainvalfile = os.path.join('nets',net_arch,'trainval.prototxt')
        print self.solverfile
        
    def solve(self,max_iter):
        caffe.set_mode_gpu()
        solver = caffe.get_solver(self.solverfile)
        deltas = np.zeros((200,0))
        i = 0
        timegrain = 200
        for i in  xrange( max_iter+1):
            s = time.clock() 
            solver.step(1)
            e = time.clock()
            deltas[i%timegrain] = e-s


def parseArgs():
    parser = ag.ArgumentParser()
    parser.add_argument('net',help='The name of the network to run')
    parser.add_argument('iters',type=int,help='The number of iterations to train with')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArgs()

    msolver = MySolver(args.net)
    msolver.solve(args.iters)
