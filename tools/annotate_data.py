import xml.etree.ElementTree as ET
import numpy as np
import os
path = 'data/person/annotations'

def _load_annotation(index):
    f = index.split('.')[0]
    print index,f
    fnametxt = os.path.join(path,f+'.txt')
    fnamexml = os.path.join(path,f+'.xml')
    if os.path.isfile(fnametxt):
        return _load_custom_annotation(fnametxt)
    elif os.path.isfile(fnamexml):
        return _load_pascal_annotation(fnamexml)
            
def _load_custom_annotation(fname):
    """
    Load image and bounding boxes info from TXT files in PERSON
    """
    data = []
    assert os.path.exists(fname), 'Error {} does not exist'.format(fname)
    with open(fname) as f:
        for line in f.readlines():
            fields = line.split(' ')
            if fields[0].strip().lower() == 'person':
                data.append(line)
    num_objs = len(data)

    
    return '1' if num_objs > 0 else '0'

def _load_pascal_annotation(fname):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(fname)
    tree = ET.parse(fname)
    objs = tree.findall('object')
    
    # Exclude the samples labeled as difficult
    non_diff_objs = [
        obj for obj in objs if int(obj.find('difficult').text) == 0 and obj.find('name').text.strip().lower() == 'person']
        # if len(non_diff_objs) != len(objs):
        #     print 'Removed {} difficult objects'.format(
        #         len(objs) - len(non_diff_objs))
    objs = non_diff_objs
    num_objs = len(objs)    

    return '1' if num_objs > 0 else '0'


def annotate(f1,f2):
    with open(f1) as fin:
        with open(f2,'w') as fout:
            
            for line in fin.readlines():
                fname = line.strip()
                ann = _load_annotation(fname)
                if not fname.split('_')[0].lower() in ('person','bike','carsgraz'):
                    fout.write('{}.jpg {}\n'.format(fname,ann))
                else:
                    fout.write('{}.png {}\n'.format(fname,ann))
                
if __name__ == '__main__':
    annotate('data/person/assign/train.txt','data/person/assign/trainclas.txt')
    annotate('data/person/assign/test.txt', 'data/person/assign/testclas.txt')
                        
