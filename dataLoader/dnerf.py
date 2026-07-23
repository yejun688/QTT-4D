import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
import imageio
from torchvision import transforms as T


from .ray_utils import *


class DNeRFDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, normalization = None):
        
        self.normalization = normalization # 默认是None
        self.N_vis = N_vis # -1
        self.root_dir = datadir  # 数据的文件夹
        self.split = split # split的种类，依据种类划分数据集
        self.is_stack = is_stack  
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.downsample=downsample
        
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0,6.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3) # tensor([[[0., 0., 0.]]])
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3) # tensor([[[1.5000, 1.5000, 1.5000]]])



    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh # 800，800
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh 根据是否downsample去调整焦距


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3) (800,800,3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True) # 每个directions，三维张量正则化
        # 定义相机的内参矩阵
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()
        
        # === Precompute region_id map per image ===
        M = 8  # e.g. 8 → 8×8=64 regions

        region_ids = torch.zeros(400, 400, dtype=torch.long)
        h_per_region = 400 // M
        w_per_region = 400 // M

        for h_temp in range(400):
            for w_temp in range(400):
                region_h = h_temp // h_per_region
                region_w = w_temp // w_per_region
                region_ids[h_temp, w_temp] = region_h * M + region_w

        # Flatten once; then for each image, repeat this
        region_ids_flat = region_ids.view(-1)
        
        self.all_region_ids = []
        # 为timenet预备
        # self.image_temps = []
        self.time_temps = []
        # self.pose_temps = []

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.all_times = [] # timenet前置
        # self.all_imgs = [] # timenet前置
        # self.all_poses = [] # timenet前置
        # self.downsample=1.0

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        self.num_img = 0
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#
            self.num_img += 1

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            # pose = np.array(frame['transform_matrix']) # dnerf中没有做
             # timenet前置
            # self.pose_temps.append(np.array(frame['transform_matrix'])) 
            # c2w矩阵
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]

            # 将图像读取为numpy数组,为timenet前序处理做准备
            img = Image.open(image_path)
            # self.image_temps.append(imageio.imread(image_path))

            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS) # 根据是否下采样调整图像
            img = self.transform(img)  # (4, h, w)
           
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            if self.normalization and self.normalization != 'None':
                img = normalize_image(img, self.normalization)
            
            self.all_rgbs += [img] 


            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3),从c2w得到原点，转换为世界坐标系后得到方向
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

            # 读取时间戳
            self.time_temps += [frame['time']]
            
            if not self.is_stack:
                self.all_region_ids += [region_ids_flat.clone()]
         # timenet前置
        # self.image_temps = (np.array(self.image_temps) / 255.).astype(np.float32) # keep all 4 channels (RGBA) (50, 800, 800, 4)
        self.time_temps = np.array(self.time_temps).astype(np.float32) # (50,)
        # self.pose_temps = np.array(self.pose_temps).astype(np.float32) # (50, 4, 4)

        self.poses = torch.stack(self.poses)
         # timenet前置
        # self.all_imgs.append(self.image_temps)
        self.all_times.append(self.time_temps)
        # self.all_poses.append(self.pose_temps)
         # timenet前置
        # self.all_imgs = np.concatenate(self.all_imgs, 0) # (50, 800, 800, 4)
        self.all_times = np.concatenate(self.all_times, 0) # (50,)
        # self.all_poses = np.concatenate(self.all_poses, 0) # (50, 4, 4)

        # self.all_imgs = self.all_imgs[...,:3]*self.all_imgs[...,-1:] + (1.-self.all_imgs[...,-1:]) # (50, 800, 800, 3)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 6)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            
            self.all_region_ids = torch.cat(self.all_region_ids, 0)

#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        self.transform = T.ToTensor()
    # 通过将内参矩阵与外参矩阵的逆矩阵相乘，计算得到的投影矩阵，用于将世界坐标转换为相机坐标系中的像素坐标    
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample


def normalize_image(image, normalization):
    if normalization == 'mean_std':
        mean = torch.tensor([0.8415, 0.8415, 0.8415])
        std = torch.tensor([0.3014, 0.3014, 0.3014])
        mean = mean.view(1, 3, 1)
        std = std.view(1, 3, 1)
        normalized_image = (image - mean) / std
    elif normalization == 'min_max':
        min_val = torch.tensor([0.0, 0.0, 0.0])
        max_val = torch.tensor([1.0, 1.0, 1.0])
        min_val = min_val.view(1, 3, 1)
        max_val = max_val.view(1, 3, 1)
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        normalized_image = image

    return normalized_image