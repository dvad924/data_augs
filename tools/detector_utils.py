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
    


def selective_window(img,scale=300,sigma=0.8,mins=10):
    #this is a time bottleneck
    im_lables,regions = selective_search(img,scale=scale,sigma=sigma,min_size=mins);
    true_regions = set([])
    print '# of regions: {}'.format(len(regions))
    for reg in regions:
        if reg['rect'] in true_regions:
            continue
        if reg['size'] < 100:
            continue
        true_regions.add(reg['rect'])
    return true_regions

class Evaluator:
    def __init__(self, file_list):
        if not os.path.exists(file_list):
            return
        self.img_files = []
        with open(file_list) as f:
            for lin in f.readlines():
                fields = lin.split(' ')
                fil = fields[0].strip()
                self.img_files.append(fil)
    
    def set_annotations (self, annots):
        self.data_annots = annots
    def get_file_annotation(self,id):
        return self.data_annots[id]

    
    
        

def area(box):
    return box[2] * box[3]

def box_intersect(box1,box2): #assumes input is given as list or array of 4 values y,x,h,w
    #first calculate intersection
    by_y = lambda x: x[0]
    by_x = lambda x: x[1]
    ly = max(box1,box2,key=by_y)
    sy = min(box1,box2,key=by_y)
    lx = max(box1,box2,key=by_x)
    sx = min(box1,box2,key=by_x)
    h = max(0, (sy[0]+sy[2]) - ly[0])
    w = max(0, (sx[1]+sx[3]) - lx[1])
    return h * w

def box_union(box1,box2): #assumes input given as list or array of 4 elements y,x,h,w
    #by principle of inclusion exclusion
    return area(box1) + area(box2) - box_intersect(box1,box2)

def box_ratio_score(box1,box2):
    return float(box_intersect(box1,box2)*1.0/box_union(box1,box2))

def calc_accuracy(gt_boxes, p_boxes):
    #This is will be a binary accuracy score
    #either the boxes overlap by more than 50% or they fail
    #currently if there are 
    matches = np.zeros(max(len(gt_boxes),len(p_boxes)))
    for p in p_boxes:
        for gt in gt_boxes:#needs to be implemented
            continue
    
class box_merger:
    def merge_boxes(self,boxes):
        clusters = self.cluster_regions(boxes)
        self.bfs(clusters)
        return self.collapse_rects()

    def collapse_rects(self):
        comps = self.ccs.values()
        regions = []
        for component in comps:
            regions.append(self.collapse_rect(component))
        return regions

    def collapse_rect(self,comp):
        sty,stx = min(comp,key=lambda x:x[0])[0],min(comp,key=lambda x:x[1])[1]
        ey,ex = max(comp,key=lambda x:x[0]+x[2]),max(comp,key=lambda x:x[0]+x[3])
        ey = ey[0] + ey[2]
        ex = ex[1] + ex[3]
        return sty,stx,ey-sty,ex-stx

    

    def bfs(self,graph):
        self.ccs = {}
        nodes = set(graph.keys())
        while len(nodes) > 0:
            node = list(nodes)[0]
            if node not in self.ccs.keys():
                self.ccs[node] = set([node])
                self.ccs[node] = self.ccs[node] | self.bfrun(graph,node)
                nodes = nodes - self.ccs[node]
                
    def bfrun(self,graph,start):
        visited = set()
        queue = [start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(graph[vertex] - visited)
        return visited
    
    def cluster_regions(self,regions):
        regions = map(lambda x:(x[0],x[1],x[2],x[3]),regions)
        region_graph = {}
        for x in xrange(0,len(regions)):
            region_graph[regions[x]] = set()
        for i in xrange(0,len(regions)-1):
            for j in xrange(i+1,len(regions)):
                if box_intersect(regions[i],regions[j]) > 0:
                    region_graph[regions[i]].add(regions[j])
                    region_graph[regions[j]].add(regions[i])
        return region_graph
                
                       
if __name__ == '__main__':    
    import pdb
    pdb.set_trace()
    # img = cv2.imread(os.path.join(testdir,'1001.jpg'))
    # stack = sliding_window(img,(138,60),25,25)
    # print len(stack)
    # for im in stack:
    #     cv2.imshow('win1',im)
    #     cv2.waitKey(33)
