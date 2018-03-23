import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2


if __name__=='__main__':
    source='windows.txt'

    if not os.path.exists(source):
        print("Setting file %s for ucf101 dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            print(source,'haha')
	    print(data,"ll")
	    for line in data:
		print(line)
                line_info = line.split()
                clip_path = line_info[0]
		print(clip_path, "-", line_info[1], "-", line_info[2])
                duration = int(line_info[1])
                target = int(line_info[2])
                item = (clip_path, duration, target)
                clips.append(item)
	    print(line,'jj')

