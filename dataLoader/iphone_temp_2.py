import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
import imageio
from torchvision import transforms as T


from .utils import *
import warnings

warnings.filterwarnings("ignore")

import random
import numpy as np

def load_everything(datadir):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(datadir)   
    kept_keys = {
            'data_class',
            'near', 'far',
            'i_train', 'i_val', 'i_test',}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)
    return data_dict

def load_data(datadir):

    K, depths = None, None
    times=None

    data_class=Load_hyper_data(datadir=datadir,
                                use_bg_points=True, add_cam=True)
    data_dict = dict(
        data_class=data_class,
        near=data_class.near, far=data_class.far,
        i_train=data_class.i_train, i_val=data_class.i_test, i_test=data_class.i_test,)
    return data_dict


class Load_hyper_data():
    def __init__(self, 
                 datadir, 
                 ratio=0.5,
                 use_bg_points=False,
                 add_cam=False):
        from .utils import Camera
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)

        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']

        self.add_cam = False
        if len(self.val_id) == 0:
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if
                            (i%4 == 0)])
            self.i_test = self.i_train+2
            self.i_test = self.i_test[:-1,]
        else:
            self.add_cam = True
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(i)
                if id in self.train_id:
                    self.i_train.append(i)
        assert self.add_cam == add_cam
        
        print('self.i_train',self.i_train)
        print('self.i_test',self.i_test)
        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['warp_id'] for i in self.all_img]
        max_time = max(self.all_time)
        self.all_time = [meta_json[i]['warp_id']/max_time for i in self.all_img]
        self.selected_time = set(self.all_time)
        self.ratio = ratio


        # all poses
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            camera = camera.scale(ratio)
            camera.position = camera.position - self.scene_center
            camera.position = camera.position * self.coord_scale
            self.all_cam_params.append(camera)

        self.all_img = [f'{datadir}/rgb/{int(1/ratio)}x/{i}.png' for i in self.all_img]
        self.h, self.w = self.all_cam_params[0].image_shape

        self.use_bg_points = use_bg_points
        if use_bg_points:
            with open(f'{datadir}/points.npy', 'rb') as f:
                points = np.load(f)
            self.bg_points = (points - self.scene_center) * self.coord_scale
            self.bg_points = torch.tensor(self.bg_points).float()
        print(f'total {len(self.all_img)} images ',
                'use cam =',self.add_cam, 
                'use bg_point=',self.use_bg_points)

    def load_idx(self, idx, not_dic=False):

        all_data = self.load_raw(idx)
        if not_dic == True:
            rays_o = all_data['rays_ori']
            rays_d = all_data['rays_dir']
            viewdirs = all_data['viewdirs']
            rays_color = all_data['rays_color']
            return rays_o, rays_d, viewdirs,rays_color
        return all_data
    

    def load_raw(self, idx):
        image = Image.open(self.all_img[idx])
        camera = self.all_cam_params[idx]
        pixels = camera.get_pixel_centers()
        rays_dir = torch.tensor(camera.pixels_to_rays(pixels)).float().view([-1,3])
        rays_ori = torch.tensor(camera.position[None, :]).float().expand_as(rays_dir)
        # rays_color = torch.tensor(np.array(image)).view([-1,3])/255.
        image = torch.tensor(np.array(image))  # (4, h, w)
           
        image = image.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
        image = image[:, :3] * image[:, -1:] + (1 - image[:, -1:])  # blend A to RGB
        
        image = image / 255.0   
        rays_color = image.float()
        return {'rays_ori': rays_ori, 
                'rays_dir': rays_dir, 
                'viewdirs':rays_dir / rays_dir.norm(dim=-1, keepdim=True),
                'rays_color': rays_color, 
                'near': torch.tensor(self.near).float().view([-1]), 
                'far': torch.tensor(self.far).float().view([-1]),}
    
# def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict):

#     # init
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if abs(cfg_model.world_bound_scale - 1) > 1e-9:
#         xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
#         xyz_min -= xyz_shift
#         xyz_max += xyz_shift


class IphoneDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, normalization = None,# 第一行为原本需要的变量（dnerf）
                 ): 
        self.white_bg = False # 只是为了调试过不相关的时候用，需要斟酌是否修改
        self.root_dir = datadir
        self.is_stack = is_stack
        self.data_dict = load_everything(datadir)
        self.data_class = self.data_dict['data_class']
        xyz_min, xyz_max = self.compute_bbox_by_cam_frustrm_hyper(data_class = self.data_dict['data_class'])
  
        self.scene_bbox = torch.stack([xyz_min, xyz_max])
        self.near_far = [self.data_dict['near'], self.data_dict['far']]
        self.h = self.data_dict['data_class'].h
        self.w = self.data_dict['data_class'].w


        # 要对应删改
        if not self.is_stack:
            # 注意下面出来的是训练集的信息,需要调整到一致的数据格式
            rgb_tr, times_flaten, cam_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = self.gather_training_rays_hyper(self.data_dict['data_class'])
            self.all_rays = torch.cat([rays_o_tr, rays_d_tr], dim=1)  
            self.all_rgbs = rgb_tr
            self.time_flatten = times_flaten
            self.cam_tr = cam_tr

        else:
            print('Temp version, ignore the same operations for test dataset!')
            # rgb_tr, times_flaten, cam_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = self.gather_testing_rays_hyper(self.data_dict['data_class'])
            # self.all_rays = torch.cat([rays_o_tr, rays_d_tr], dim=1).reshape(-1, self.h, self.w, 6) 
            # self.all_rgbs = rgb_tr.reshape(-1, self.h, self.w, 3)  # (len(self.meta['frames]),h,w,3)
            # self.time_flatten = times_flaten.reshape(-1, self.h * self.w, 1)
            # self.cam_tr = cam_tr(-1, self.h * self.w, 1)

    def compute_bbox_by_cam_frustrm_hyper(self, data_class):
        print('compute_bbox_by_cam_frustrm: start')
        xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
        xyz_max = -xyz_min
        for i in data_class.i_train:
            rays_o, _, viewdirs,_ = data_class.load_idx(i,not_dic=True)
            pts_nf = torch.stack([rays_o+viewdirs*data_class.near, rays_o+viewdirs*data_class.far])
            xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
            xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
        print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
        print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
        print('compute_bbox_by_cam_frustrm: finish')
        return xyz_min, xyz_max
                 
    def gather_training_rays_hyper(self, data_class):
        now_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N = len(data_class.i_train)*data_class.h*data_class.w
        rgb_tr = torch.zeros([N,3], device=now_device)
        rays_o_tr = torch.zeros_like(rgb_tr)
        rays_d_tr = torch.zeros_like(rgb_tr)
        viewdirs_tr = torch.zeros_like(rgb_tr)
        times_tr = torch.ones([N,1], device=now_device)
        cam_tr = torch.ones([N,1], device=now_device)
        imsz = []
        top = 0
        for i in data_class.i_train:
            rays_o, rays_d, viewdirs,rgb = data_class.load_idx(i,not_dic=True)
            n = rgb.shape[0]
            if data_class.add_cam:
                cam_tr[top:top+n] = cam_tr[top:top+n]*data_class.all_cam[i]
            times_tr[top:top+n] = times_tr[top:top+n]*data_class.all_time[i]
            rgb_tr[top:top+n].copy_(rgb)
            rays_o_tr[top:top+n].copy_(rays_o.to(now_device))
            rays_d_tr[top:top+n].copy_(rays_d.to(now_device))
            viewdirs_tr[top:top+n].copy_(viewdirs.to(now_device))
            imsz.append(n)
            top += n
        assert top == N
        return rgb_tr, times_tr,cam_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz 

    def gather_testing_rays_hyper(self,  data_class):
        now_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N = len(data_class.i_test)*data_class.h*data_class.w
        rgb_tr = torch.zeros([N,3], device=now_device)   
        rays_o_tr = torch.zeros_like(rgb_tr)
        rays_d_tr = torch.zeros_like(rgb_tr)
        viewdirs_tr = torch.zeros_like(rgb_tr)
        times_tr = torch.ones([N,1], device=now_device)
        cam_tr = torch.ones([N,1], device=now_device)
        imsz = []
        top = 0
        for i in data_class.i_test:
            rays_o, rays_d, viewdirs,rgb = data_class.load_idx(i,not_dic=True)
            n = rgb.shape[0]
            if data_class.add_cam:
                cam_tr[top:top+n] = cam_tr[top:top+n]*data_class.all_cam[i]
            times_tr[top:top+n] = times_tr[top:top+n]*data_class.all_time[i]
            rgb_tr[top:top+n].copy_(rgb)
            rays_o_tr[top:top+n].copy_(rays_o.to(now_device))
            rays_d_tr[top:top+n].copy_(rays_d.to(now_device))
            viewdirs_tr[top:top+n].copy_(viewdirs.to(now_device))
            imsz.append(n)
            top += n
        assert top == N
        return rgb_tr, times_tr,cam_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz  