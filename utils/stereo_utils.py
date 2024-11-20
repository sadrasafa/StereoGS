import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from argparse import Namespace
from utils.graphics_utils import fov2focal
sys.path.append(os.path.join('utils/RAFT-Stereo'))
sys.path.append(os.path.join('utils/RAFT-Stereo', 'core'))
from raft_stereo import RAFTStereo

DEVICE = 'cuda'
args = Namespace(restore_ckpt='models/raftstereo-middlebury.pth', mixed_precision=True, valid_iters=32, 
                 hidden_dims=[128,128,128], corr_implementation='reg_cuda', shared_backbone=False, corr_radius=4,
                 corr_levels=4, n_downsample=2, context_norm='batch', slow_fast_gru=False, n_gru_layers=3)

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class RAFT:
    def __init__(self, path):
        args.restore_ckpt = path
        if 'iraftstereo_rvc' in path:
            args.context_norm = 'instance'
        self.model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        self.model.load_state_dict(torch.load(args.restore_ckpt))
        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()

    def predict_disparity(self, left_image, right_image):
        with torch.no_grad():
            padder = InputPadder(left_image.shape, divis_by=32)
            left_image, right_image = padder.pad(left_image, right_image)
            _, flow_up = self.model(left_image, right_image, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up)
            return -flow_up
    
def disp2depth(disp, baseline, cam):
    assert disp.shape == (1, cam.image_height, cam.image_width)
    focal_length = fov2focal(cam.FoVx, cam.image_width)
    depth = baseline * focal_length / disp
    return depth

