import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2


def find_classes(dir):
    #classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    #classes.sort()
    #class_to_idx = {classes[i]: i for i in range(len(classes))}
    classes=['not fall','fall']
    class_to_idx=[0,1]
    return classes, class_to_idx

def make_dataset(root, source):

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
                item = (clip_path, duration, target)
                clips.append(item)
    return clips

def ReadSegmentRGB(path, offsets, new_height, new_width, new_length, is_color, name_pattern):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length+1):
            frame_name = name_pattern % (length_id + offset)
            frame_path = path + "/" + frame_name
            cv_img_origin = cv2.imread(frame_path, cv_read_flag)
            if cv_img_origin is None:
               print("Could not load file %s" % (frame_path))
               sys.exit()
               # TODO: error handling here
            if new_width > 0 and new_height > 0:
                # use OpenCV3, use OpenCV2.4.13 may have error
                cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
            else:
                cv_img = cv_img_origin
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            sampled_list.append(cv_img)
    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input

def ReadSegmentFlow(path, offsets, new_height, new_width, new_length, is_color, name_pattern,average_duration):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []

    sampled_id=None
    if average_duration<=6:
        sampled_id=range(1,7)
    elif average_duration>6 and average_duration<12:
        sampled_id=range(average_duration-6+1,average_duration+1)
    else:
        sampled_id=range(average_duration-12,average_duration+1,2)
        
    for length_id in sampled_id:
        frame_name_x = name_pattern % ("x", length_id)
        frame_path_x = path + "/" + frame_name_x
        cv_img_origin_x = cv2.imread(frame_path_x, cv_read_flag)
        frame_name_y = name_pattern % ("y", length_id)
        frame_path_y = path + "/" + frame_name_y
        cv_img_origin_y = cv2.imread(frame_path_y, cv_read_flag)
        if cv_img_origin_x is None or cv_img_origin_y is None:
            print("Could not load file %s or %s" % (frame_path_x, frame_path_y))
            sys.exit()
            # TODO: error handling here
        if new_width > 0 and new_height > 0:
            cv_img_x = cv2.resize(cv_img_origin_x, (new_width, new_height), interpolation)
            cv_img_y = cv2.resize(cv_img_origin_y, (new_width, new_height), interpolation)
        else:
            cv_img_x = cv_img_origin_x
            cv_img_y = cv_img_origin_y
        sampled_list.append(np.expand_dims(cv_img_x, 2))
        sampled_list.append(np.expand_dims(cv_img_y, 2))

    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input


class ucf101(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality,
                 name_pattern=None,
                 is_color=True,
                 num_segments=1,
                 new_length=1,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None,
                 video_transform=None):

        classes, class_to_idx = find_classes(root)
        clips = make_dataset(root, source)

        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))

        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.clips = clips

        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb":
                self.name_pattern = "%03d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "flow_%s_%05d.jpg"

        self.is_color = is_color
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_width = new_width
        self.new_height = new_height

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        path, duration, target = self.clips[index]
        average_duration = int(duration / self.num_segments)
        offsets = []
        for seg_id in range(self.num_segments):
            if self.modality=="rgb":
                oneEighth=int(average_duration*1/8)
                threeFourth=int(average_duration*3/4)
                if self.phase == "train":
                    if average_duration >= 1:
                        offset = random.randint(threeFourth-oneEighth, threeFourth+oneEighth)
                        offsets.append(offset)
                    else:
                        offsets.append(0)
                elif self.phase == "val":
                    if average_duration >= 1:
                        offsets.append(threeFourth)
                    else:
                        offsets.append(0)
                else:
                    print("Only phase train and val are supported.")
            elif self.modality=="flow":
                print("flow mode")
            else:
                print("Only rgb and flow are supported.")

        if self.modality == "rgb":
            clip_input = ReadSegmentRGB(path,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern
                                        )
        elif self.modality == "flow":
            clip_input = ReadSegmentFlow(path,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        average_duration
                                        )
        else:
            print("No such modality %s" % (self.modality))

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)

        return clip_input, target


    def __len__(self):
        return len(self.clips)
