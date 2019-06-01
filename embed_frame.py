import os
import numpy as np
import torchvision.models as models
# import vgg
import torchvision.transforms as transforms
import PIL.Image as Image

kDataRoot = '/home/jeongmin/workspace/data/Action/HMDB51/frames'
kSaveRoot = '/home/jeongmin/workspace/data/Action/HMDB51/vgg_feature'


if __name__ == '__main__':

    # vgg16 = models.vgg16(pretrained=True).features
    # vgg16 = vgg.vgg16(pretrained=True,extract_feature=True).eval().cuda()
    model = models.vgg16(pretrained=True).cuda()
    feature = model.features[:]
    flatten = model.classifier[:1]
    for action_class in os.listdir(kDataRoot):
        kDataActionRoot = os.path.join(kDataRoot, action_class)
        kSaveActionRoot = os.path.join(kSaveRoot, action_class)

        if not os.path.isdir(kSaveActionRoot):
            os.mkdir(kSaveActionRoot)

        for video_num in os.listdir(kDataActionRoot):
            kDataVideoRoot = os.path.join(kDataActionRoot, video_num)
            kSaveVideoRoot = os.path.join(kSaveActionRoot, video_num)

            if not os.path.isdir(kSaveVideoRoot):
                os.mkdir(kSaveVideoRoot)

            for file_name in os.listdir(kDataVideoRoot):
                basename, ext = os.path.splitext(file_name)

                if not ext == '.jpg':
                    continue

                frame_num = basename.split('_')[-1]
                save_name = 'feature_' + frame_num + '.npy'

                img = Image.open(os.path.join(kDataVideoRoot, file_name))
                normalize = transforms.Compose([transforms.Resize([224, 224]),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])

                feat = feature(normalize(img).unsqueeze(0).cuda())
                feat = feat.view(feat.size(0), -1)
                flat = flatten(feat)
                np.save(os.path.join(kSaveVideoRoot, save_name), flat.detach().cpu().numpy())
