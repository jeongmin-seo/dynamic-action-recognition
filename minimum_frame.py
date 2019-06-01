import os
import numpy as np

kDataRoot = '/home/jeongmin/workspace/data/Action/HMDB51/frames'
kSaveRoot = '/home/jeongmin/workspace/data/Action/HMDB51/vgg_feature'

min_frame = None
max_frame = None
for action_class in os.listdir(kDataRoot):
    kSaveActionRoot = os.path.join(kSaveRoot, action_class)

    for video_num in os.listdir(kSaveActionRoot):
        kSaveVideoRoot = os.path.join(kSaveActionRoot, video_num)

        if not min_frame or min_frame > len(os.listdir(kSaveVideoRoot)):
            min_frame = len(os.listdir(kSaveVideoRoot))

        if not max_frame or max_frame < len(os.listdir(kSaveVideoRoot)):
            max_frame = len(os.listdir(kSaveVideoRoot))


print(min_frame)
print(max_frame)

a = np.load('/home/jeongmin/workspace/data/Action/HMDB51/vgg_feature/brush_hair/1/feature_00001.npy')
print(a.shape)
