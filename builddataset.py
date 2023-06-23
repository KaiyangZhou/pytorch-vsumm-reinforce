"""
Preprocessing code for creating video summarisation training dataset files like tvsum.h5 and summe.h5
"""

#Input path of folder with training videoes
input_path = 'video_path'
#.h5 dataset filename(optionally with path to save)
cp_path = 'dataset/training_datasets.h5'

"""Below code uses googlenet model for extracting feature from the videos and KTS for video segmentation like tvsum and summe datasets"""

from __future__ import print_function
import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable
import os
from tqdm import tqdm
import math
import cv2
import numpy as np
import h5py
from PIL import Image

input_videos_folder = input_path #path of input video
h5file_name= cp_path #path of .h5 file

class Rescale(object):
    """Rescale a image to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
    """
    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img

transform = transforms.Compose([Rescale(224, 224),transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#net = models.googlenet(pretrained=True).float().cuda()
net = models.googlenet(pretrained=True).float()
net.eval()
fea_net = nn.Sequential(*list(net.children())[:-2])


def calc_scatters(K):
    #calculating scatter matrix to find covariance
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n+1, n+1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1);
    scatters = np.zeros((n, n))
    diagK2 = np.diag(K2)
    i = np.arange(n).reshape((-1,1))
    j = np.arange(n).reshape((1,-1))
    scatters = (K1[1:].reshape((1,-1))-K1[:-1].reshape((-1,1))
                - (diagK2[1:].reshape((1,-1)) + diagK2[:-1].reshape((-1,1)) - K2[1:,:-1].T - K2[:-1,1:]) / ((j-i+1).astype(float) + (j==i-1).astype(float)))
    scatters[j<i]=0
    return scatters

def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=True,out_scatters=None):
    #finding change points
    m = int(ncp)
    (n, n1) = K.shape
    J = calc_scatters(K)
    if out_scatters != None:
        out_scatters[0] = J
    I = 1e101*np.ones((m+1, n+1))
    I[0, lmin:lmax] = J[0, lmin-1:lmax-1]
    if backtrack:
        p = np.zeros((m+1, n+1), dtype=int)
    for k in range(1,m+1):
        for l in range((k+1)*lmin, n+1):
            tmin = max(k*lmin, l-lmax)
            tmax = l-lmin+1
            c = J[tmin:tmax,l-1].reshape(-1) + I[k-1, tmin:tmax].reshape(-1)
            I[k,l] = np.min(c)
            if backtrack:
                p[k,l] = np.argmin(c)+tmin
    cps = np.zeros(m, dtype=int)
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]
    scores = I[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores

def cpd_auto(K, ncp, vmax, desc_rate=1 ):
    #finding change points based on Kernal temporal segmentation method
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False)
    N = K.shape[0]
    penalties = np.zeros(m+1)
    ncp = np.arange(1, m+1)
    penalties[1:] = (ncp/(2.0*N))*(np.log(float(N)/ncp)+1)
    costs = scores/float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin(K, m_best)
    return (cps, costs)

class generate:
    #encoder
    def __init__(self, video_path, save_path):
        #self.resnet = Model_Resnet()
        self.dataset = {}
        self.video_name=video_path.split('/')[-1]
        self.video_path = video_path
        self.h5_file = h5py.File(save_path, 'w')

        self.video_list = []
        self._set_video_list(video_path)

    def _set_video_list(self, video_path):
        #creating groups for each video
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = os.listdir(video_path)
            self.video_list.sort()
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx+1)] = {}
            self.h5_file.create_group('video_{}'.format(idx+1))

    def _extract_feature(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        #fea.append(fea_net(transform(Image.fromarray(frame)).cuda().unsqueeze(0)).squeeze().detach().cpu())
        frame_features = fea_net(transform(Image.fromarray(frame)).unsqueeze(0)).squeeze().detach().cpu()
        return frame_features

    def _get_change_points(self, video_feat, n_frame, fps):
        # extracting change points and number of frames per segment
        n = n_frame / fps
        m = int(math.ceil(n/2.0))
        K = np.dot(video_feat, video_feat.T)
        change_points, _ = cpd_auto(K, m, 1)
        change_points = np.concatenate(([0],change_points,[n_frame-1]))
        temp_change_points = []
        for idx in range(len(change_points)-1):
            segment = [change_points[idx], change_points[idx+1]-1]
            if idx == len(change_points)-2:
                segment = [change_points[idx], change_points[idx+1]]

            temp_change_points.append(segment)
        change_points = np.array(list(temp_change_points))
        temp_n_frame_per_seg = []
        for change_points_idx in range(len(change_points)):
            n_frame = change_points[change_points_idx][1] - change_points[change_points_idx][0]
            temp_n_frame_per_seg.append(n_frame)
        n_frame_per_seg = np.array(list(temp_n_frame_per_seg))
        return change_points, n_frame_per_seg
    def _save_dataset(self):
        pass

    def gen(self):
        #adding contents to the .h5 file
        for video_idx, video_filename in enumerate(self.video_list):
            video_path = video_filename
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)
            video_basename = os.path.basename(video_path).split('.')[0]
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ratio =  n_frames//320
            frame_list = []
            picks = []
            print(video_filename)
            video_feat = None
            video_feat_for_train = None
            c=0
            user_summary=[]
            for frame_idx in tqdm(range(n_frames-1)):
                success, frame = video_capture.read()
                if success:
                    if frame_idx % ratio == 0:
                        frame_feat = self._extract_feature(frame)
                        picks.append(frame_idx)

                        if video_feat_for_train is None:
                            video_feat_for_train = frame_feat
                        else:
                            video_feat_for_train = np.vstack((video_feat_for_train, frame_feat))
                    if video_feat is None:
                        video_feat = frame_feat
                    else:
                        video_feat = np.vstack((video_feat, frame_feat))
                else:
                    break
            video_capture.release()
            change_points, n_frame_per_seg = self._get_change_points(video_feat, n_frames, fps)

            self.h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat_for_train)
            self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames# number of frames of video
            self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps #frames per second
            self.h5_file['video_{}'.format(video_idx+1)]['video_name'] = self.video_name #name of input video
            self.h5_file['video_{}'.format(video_idx+1)]['change_points'] = change_points # change points(indices of segments)
            self.h5_file['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg #number of frames per segment
#generating .h5 file

h5_gen = generate(input_videos_folder,h5file_name)
h5_gen.gen()
h5_gen.h5_file.close()
print("Dataset at", cp_path)

"""Code to see the contents of created h5 files"""

def h5printR(item, leading = ''):
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')

# Print structure of a `.h5` file
def h5print(filename):
    with h5py.File(filename, 'r') as h:
        print(filename)
        h5printR(h, '  ')
h5print('dataset/training_datasets.h5')

"""After this preprocessing any video(in context of dataset categories) can be summarised by the trained model."""