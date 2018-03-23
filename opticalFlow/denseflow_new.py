import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
from IPython import embed #to debug
import scipy.misc
import time


def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def save_flows(flows,save_dir,frame_num,bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    save_dir=save_dir.split('Frame/')[1]
    
    #rescale to 0~255 with the bound setting
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    
    if not os.path.exists(os.path.join(data_root,'data',new_dir,save_dir)):
        os.makedirs(os.path.join(data_root,'data',new_dir,save_dir))

    #save the flows
    save_x=os.path.join(data_root,'data',new_dir,save_dir,'flow_x_{:05d}.jpg'.format(frame_num))
    save_y=os.path.join(data_root,'data',new_dir,save_dir,'flow_y_{:05d}.jpg'.format(frame_num))
    flow_x_img=Image.fromarray(flow_x)
    flow_y_img=Image.fromarray(flow_y)
    scipy.misc.imsave(save_x,flow_x_img)
    scipy.misc.imsave(save_y,flow_y_img)
    return 0

def dense_flow(augs):
    '''
    To extract dense_flow images
    :param augs:the detailed augments:
        video_name: the video name which is like: 'v_xxxxxxx',if different ,please have a modify.
        save_dir: the destination path's final direction name.
        step: num of frames between each two extracted frames
        bound: bi-bound parameter
    :return: no returns
    '''
    video_name,save_dir,step,bound=augs
    #video_name=video_name.split('/')[-1]

    
    if not os.path.exists(video_name):
         print 'Could not find image folder! ', video_name
         exit()

    image,prev_image,gray,prev_gray=None,None,None,None
    flow_num=0
    imagelist=os.listdir(video_name)
    for element in range(1,len(imagelist)+1,step):
        if flow_num==0:
            image_path=os.path.join(video_name,"%03d.jpg"%(element))
            prev_image=cv2.imread(image_path)
	    prev_gray=cv2.cvtColor(prev_image,cv2.COLOR_RGB2GRAY)
            flow_num+=1
            continue
	image_path=os.path.join(video_name,"%03d.jpg"%(element))
        current_image=cv2.imread(image_path)
	img1=prev_image[100,100]
	img2=current_image[100,100]
        current_gray=cv2.cvtColor(current_image,cv2.COLOR_RGB2GRAY)
        frame_0=prev_gray
        frame_1=current_gray
        dtvl1=cv2.createOptFlow_DualTVL1()
        flowDTVL1=dtvl1.calc(frame_0,frame_1,None)
        save_flows(flowDTVL1,save_dir,flow_num,bound) #this is to save flows and img.
        prev_gray=current_gray
        prev_image=current_image
        flow_num+=1

def get_video_list():
    video_list=[]
    for cls_names in os.listdir(videos_root):
        cls_path=os.path.join(videos_root,cls_names)
	video_list.append(cls_path)
        #for video_ in os.listdir(cls_path):
            #video_list.append(video_)
    video_list.sort()
    return video_list,len(video_list)

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
                clip_path = os.path.join(root, line_info[0])
                duration = int(line_info[1])
                target = int(line_info[2])
                #item = (clip_path, duration)
                clips.append(clip_path)
		print(clip_path)
    return clips, len(clips)



def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--source',default='dev.txt',type=str,help='set the dataset name, to find the data path')
    parser.add_argument('--data_root',default='./',type=str)
    parser.add_argument('--new_dir',default='flows',type=str)
    parser.add_argument('--num_workers',default=4,type=int,help='num of workers to act multi-process')
    parser.add_argument('--step',default=1,type=int,help='gap frames')
    parser.add_argument('--bound',default=15,type=int,help='set the maximum of optical flow')
    parser.add_argument('--s_',default=0,type=int,help='start id')
    parser.add_argument('--e_',default=1,type=int,help='end id')
    parser.add_argument('--mode',default='run',type=str,help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args

if __name__ =='__main__':

    # example: if the data path not setted from args,just manually set them as belows.
    #dataset='ucf101'
    #data_root='/S2/MI/zqj/video_classification/data'
    #data_root=os.path.join(data_root,dataset)

    args=parse_args()
    data_root='./'
    #videos_root=os.path.join(data_root,'frames')
    videos_root=data_root

    #specify the augments
    num_workers=args.num_workers
    step=args.step
    bound=args.bound
    s_=args.s_
    e_=args.e_
    new_dir=args.new_dir
    mode=args.mode
    #get video list
    #video_list,len_videos=get_video_list()
    #video_list=video_list[s_:e_]
    source=args.source
    video_list,len_videos=get_video_list_new(source,videos_root)

    #len_videos=e_-s_ #min(e_-s_,13320-s_) # if we choose the ucf101
    print 'find {} videos.'.format(len_videos)
    flows_dirs=[video for video in video_list]
    print 'get videos list done! '

    pool=Pool(num_workers)
    if mode=='run':
        pool.map(dense_flow,zip(video_list,flows_dirs,[step]*len(video_list),[bound]*len(video_list)))
    else: #mode=='debug
        dense_flow((video_list[0],flows_dirs[0],step,bound))
