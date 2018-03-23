import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
from IPython import embed #to debug
import scipy.misc


def get_video_list_new(source,root):

    
    if not os.path.exists(source):
        print("Setting file %s for ucf101 dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                origin_path=line_info[0]
                intermediate_path=origin_path.split('Frame/')[1]
                clip_path=os.path.join('data',new_dir,intermediate_path)             
                duration = int(len(os.listdir(clip_path))/2)
                target = int(line_info[2])
                clips.append('{} {} {}\n'.format(clip_path,duration,target))
    return clips, len(clips)



def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--source',default='dev.txt',type=str,help='set the dataset name, to find the data path')
    parser.add_argument('--data_root',default='./',type=str)
    parser.add_argument('--new_dir',default='flows',type=str)
    args = parser.parse_args()
    return args

if __name__ =='__main__':

    # example: if the data path not setted from args,just manually set them as belows.
    #dataset='ucf101'
    #data_root='/S2/MI/zqj/video_classification/data'
    #data_root=os.path.join(data_root,dataset)

    args=parse_args()
    data_root=args.data_root
    videos_root=data_root
    new_dir=args.new_dir
    source=args.source
    video_list,len_videos=get_video_list_new(source,videos_root)
    
    newfile='new_'+source
    open(newfile,'w').writelines(video_list)
    
    print 'find {} videos.'.format(len_videos)

