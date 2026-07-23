import cv2 as cv
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from tqdm import tqdm
from pyhocon import ConfigFactory

from .ray_utils import *  # 先融合了blender和Tenso4Ddataset里面导入的，后续再删
# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K # intrinsics规模(4, 4)，前三行三列为K矩阵/第三行第三列元素的结果

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose() # 这里的pose其实是世界到相机坐标系的逆矩阵，即表示的矩阵是相机坐标系到世界坐标系
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class DynamicDataset(Dataset):
    def __init__(self,  conf_path, mode='train', case='CASE_NAME', downsample=1.0, is_stack=False, N_vis=-1, normalization = None):
        super(Dataset, self).__init__()
        
        print('Load data: Begin')
        self.device = torch.device('cuda')
        f = open(self.conf_path)
        conf_text = f.read() # 具体的参数信息都读取出来存到该变量中
        conf_text = conf_text.replace('CASE_NAME', case) # 把所有的'CASE_NAME'都替换为case变量的内容
        f.close()
        
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        conf = self.conf['dataset']

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name') # 'cameras_sphere.npz'
        self.object_cameras_name = conf.get_string('object_cameras_name') # 'cameras_sphere.npz'

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True) # True
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1) # 1.1
        self.near = conf.get_float('near', default=-1) # -1
        self.far = conf.get_float('far', default=-1) # -1
        self.n_frames = conf.get_int('n_frames', default=128) # 128

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis) # 400
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0 # 归一化
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0 # (400, 1024, 1024, 3)

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)] # 400个变换矩阵
        self.fid_list = [torch.LongTensor(np.array([camera_dict['fid_%d' % idx]])) for idx in range(self.n_images)]
        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)] # 400个变换矩阵

        self.intrinsics_all = []
        self.pose_all = []
        self.proj_all = []
        # zip函数用于将多个可迭代对象（如列表、元组等）“压缩”在一起，使得它们的元素成对或成组地配对
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4] # (3, 4)
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
            self.proj_all.append(torch.from_numpy(P).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3],从numpy数组转换到torch张量
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.errors = self.masks[:, :, :, :1].clone()
        self.errors = F.interpolate(self.errors.permute(0, 3, 1, 2), (self.images.shape[1] // 8, self.images.shape[2] // 8), mode='bilinear')
        self.errors = F.max_pool2d(self.errors, 7, stride=1, padding=3)
        self.errors = self.errors.permute(0, 2, 3, 1) # torch.Size([400, 128, 128, 1])
        self.radius = torch.zeros(self.masks.shape[0], self.masks.shape[2], self.masks.shape[1], 1) # 初始化，记录每个图像的半径，[n_images, W, H, 1]
        
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0] # tensor(867.5587, device='cuda:0')
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.proj_all = torch.stack(self.proj_all).to(self.device) # torch.Size([400, 3, 4])
        self.H, self.W = self.images.shape[1], self.images.shape[2] # 1024 1024
        self.image_pixels = self.H * self.W 
        self.fid_all = torch.stack(self.fid_list).to(self.device)
        self.time_emb_list = (self.fid_all / self.n_frames * 2) - 0.95 # torch.Size([400, 1])

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0'] # (4, 4)
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0] # array([-1.00999999, -1.01      , -1.01      ])
        self.object_bbox_max = object_bbox_max[:3, 0] # array([1.01000001, 1.01      , 1.01      ])
        self.process_radius()

        print('Load data: End')
        
    def process_radius(self):
        for img_idx in tqdm(range(self.images.shape[0])):
            tx = torch.linspace(0, self.W - 1, self.W, device=self.device) # torch.Size([1024])
            ty = torch.linspace(0, self.H - 1, self.H, device=self.device) # torch.Size([1024])
            pixels_x, pixels_y = torch.meshgrid(tx, ty) # torch.Size([1024, 1024]),torch.Size([1024, 1024])
            p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
            rays_v = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3 完成像素坐标系到相机中心坐标系的变换
            rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3 完成相机中心坐标系到世界坐标系的变换
            dx = torch.sqrt(torch.sum((rays_v[:-1, :, :] - rays_v[1:, :, :]) ** 2, dim=-1)) # rays_v[:-1, :, :]：去掉最后一行 rays_v[1:, :, :]：去掉第一行的射线方向向量。以达到错位相减的效果
            dx = torch.cat([dx, dx[-2:-1, :]], dim=0) # torch.Size([1024, 1024])
            # Cut the distance in half, and then round it out so that it's
            # halfway between inscribed by / circumscribed about the pixel.
            radii = dx[..., None] * 2 / np.sqrt(12) # 2 / np.sqrt(12) 是一个常数因子，用于标准化和调整半径的大小，使其介于像素内切圆和外接圆之间。这种调整可以确保计算出的半径既不过大也不过小，适合实际应用。
            self.radius[img_idx] = radii.detach().cpu()   # W H 3

            