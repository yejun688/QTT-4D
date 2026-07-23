import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from collections import defaultdict
from PIL import Image
import imageio
from torchvision import transforms as T
import shutil


from .ray_utils import *
from .iphone_helpers import *

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

class IphoneDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, normalization = None):
        self.normalization = normalization # 默认是None
        self.N_vis = N_vis # -1
        self.root_dir = datadir  # 数据的文件夹
        self.split = split # split的种类，依据种类划分数据集
        self.is_stack = is_stack  
        self.img_wh = (int(720/downsample),int(960/downsample))
        self.define_transforms()
        
                # 指定文件路径
        extra_json_path = os.path.join(self.root_dir, "extra.json") 

        # 读取 JSON 文件
        with open(extra_json_path, "r") as f:
            extra_data = json.load(f)

        scene_bbox = torch.tensor(extra_data["bbox"])
        # self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.downsample=downsample
        
        self.split_dir = osp.join(self.root_dir, 'create_split')
        
        if self.split == 'train':
            self.create_splits(self.root_dir)
            self.read_train_meta()
        else:
            self.read_test_meta()
        
        # self.define_proj_mat()
        
                # 指定文件路径
        scene_json_path = os.path.join(self.root_dir, "scene.json") 

        # 读取 JSON 文件
        with open(scene_json_path, "r") as f:
            scene_data = json.load(f)

        self.white_bg = False
        self.near_far = [scene_data["near"] / scene_data["scale"], scene_data["far"] / scene_data["scale"]]
        
        self.center = torch.tensor(scene_data["center"])# tensor([[[0., 0., 0.]]])
        
        self.scene_bbox = scene_bbox * scene_data["scale"] + self.center
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3) # tensor([[[1.5000, 1.5000, 1.5000]]])
        
    def create_splits(self, root_dir):
        """
    根据 splits 目录中的 train.json 和 val.json，将对应的相机 JSON 文件和图像 PNG 文件
    复制到 create_split 目录下的 images, test_images, cameras, test_cameras 文件夹中。

    参数：
        root_dir (str): Dycheck/apple 目录的路径，包含 camera, rgb/1x, splits 文件夹。
    """
        # 目标输出文件夹
        output_dir = os.path.join(root_dir, "create_split")
        os.makedirs(output_dir, exist_ok=True)

        # 创建目标目录
        images_dir = os.path.join(output_dir, "images")
        test_images_dir = os.path.join(output_dir, "test_images")
        cameras_dir = os.path.join(output_dir, "cameras")
        test_cameras_dir = os.path.join(output_dir, "test_cameras")

        for d in [images_dir, test_images_dir, cameras_dir, test_cameras_dir]:
            os.makedirs(d, exist_ok=True)

        # 读取 train.json 和 val.json
        splits = {
            "train": os.path.join(root_dir, "splits", "train.json"),
            "val": os.path.join(root_dir, "splits", "val.json")
        }

        frame_lists = {}

        for split, path in splits.items():
            if not os.path.exists(path):
                print(f"警告: {path} 文件不存在，跳过该分割集")
                continue
            with open(path, "r") as f:
                data = json.load(f)
                frame_lists[split] = data.get("frame_names", [])  # 读取 frame_names 列表

        # 设置源数据目录
        camera_src_dir = os.path.join(root_dir, "camera")  # 相机 JSON 目录
        rgb_src_dir = os.path.join(root_dir, "rgb", "1x")  # PNG 图像目录

        # 处理 train.json
        if "train" in frame_lists:
            self._move_files(frame_lists["train"], camera_src_dir, cameras_dir, ".json")
            self._move_files(frame_lists["train"], rgb_src_dir, images_dir, ".png")

         # 处理 val.json
        if "val" in frame_lists:
            self._move_files(frame_lists["val"], camera_src_dir, test_cameras_dir, ".json")
            self._move_files(frame_lists["val"], rgb_src_dir, test_images_dir, ".png")

        print("文件分类完成！")


    def _move_files(self, frame_list, src_dir, dest_dir, ext):
        """
    将 frame_list 中的文件（扩展名为 ext）从 src_dir 复制到 dest_dir。

    参数：
        frame_list (list): 需要处理的 frame 名称列表
        src_dir (str): 源文件所在的目录
        dest_dir (str): 目标存储目录
        ext (str): 文件扩展名，例如 ".json" 或 ".png"
    """
        for frame in frame_list:
            file_name = f"{frame}{ext}"
            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
            else:
                print(f"警告: {src_path} 不存在，跳过")
           
    @torch.no_grad
    def load_rgb(self, rgb_dirname="images"): # 注意，train的时候传images，test的时候传test_images
        # img_dir = osp.join(self.ws, rgb_dirname)
        img_dir = osp.join(self.root_dir, rgb_dirname)
        img_npz = img_dir + ".npz"
        if osp.exists(img_npz):
            images = np.load(img_npz)["images"]  # ! in [0,255]
            images = torch.from_numpy(images).float() / 255.0  # T,H,W,3
            img_names = [f"{i:05d}" for i in range(images.shape[0])]
        elif osp.exists(img_dir):
            img_fns = [
                f
                for f in os.listdir(img_dir)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
            img_fns.sort()
            img_names = [osp.splitext(f)[0] for f in img_fns]
            images = [imageio.imread(osp.join(img_dir, img_fn)) for img_fn in img_fns]
            images = torch.Tensor(np.stack(images)) / 255.0  # T,H,W,3
        else:
            raise ValueError(f"Cannot find images in {img_dir}")
        # # assign
        # self.frame_names = img_names
        # images = images[..., :3] # torch.Size([475, 480, 360, 3])
        # self.register_gradfree_buffer("rgb", images.detach())
        return images
        
    def read_train_meta(self):

        # with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
        #     self.meta = json.load(f)

        w, h = self.img_wh # 720 ,960
        
        
        (
        gt_training_cam_T_wi, # torch.Size([475, 4, 4])
        gt_testing_cam_T_wi_list, # two cameras torch.Size([212, 4, 4]) torch.Size([320, 4, 4])
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov, # 52.071839030522504
        gt_testing_fov_list, # [52.5902636880661, 52.41911396036227]
        gt_training_cxcy_ratio,
        gt_testing_cxcy_ratio_list,
        ) = load_iphone_gt_poses(src = self.split_dir, t_subsample = 1)
        
        # self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        # self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh 根据是否downsample去调整焦距
                
        self.focal = 0.5 * w / np.tan(np.deg2rad(0.5 * gt_training_fov))
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3) (960,720,3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True) # 每个directions，三维张量正则化
        # 定义相机的内参矩阵
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()
        
        # 为timenet预备
        # self.image_temps = []
        self.time_temps = []
        # self.pose_temps = []

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_times = [] # timenet前置
        # self.all_imgs = [] # timenet前置
        # self.all_poses = [] # timenet前置
        # self.downsample=1.0
        
        camera_dir = osp.join(self.split_dir, "cameras")
        image_dir = osp.join(self.split_dir, "images")

        camera_files = sorted([f for f in os.listdir(camera_dir) if f.endswith('.json')])
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        # 确保camera和image数量匹配
        assert len(camera_files) == len(image_files), "Camera and image counts do not match!"
        
        time_step = 1.0 / len(camera_files)
        
        self.num_img = len(camera_files)
        
        count = 1

        # 遍历cameras和images
        for cam_file, img_file in tqdm(zip(camera_files, image_files), total=len(camera_files), desc="Loading Data"):
            # 读取camera文件
            cam_path = os.path.join(camera_dir, cam_file)
            with open(cam_path, 'r') as f:
                camera_data = json.load(f)

            # 提取pose相关数据
            orientation = np.array(camera_data['orientation'])  # 3x3 旋转矩阵
            position = np.array(camera_data['position'])        # 位置向量 (3,)
        
            # 生成4x4的c2w矩阵 (相机到世界坐标)
            c2w = np.eye(4)
            c2w[:3, :3] = orientation.T  # 旋转矩阵转置，从世界到相机变为相机到世界
            c2w[:3, 3] = position        # 平移向量
            c2w = torch.FloatTensor(c2w)
            self.poses += [c2w]

            # 读取对应的图像文件
            img_path = os.path.join(image_dir, img_file)
            self.image_paths += [img_path]
            
            img = Image.open(img_path)

            # 对图像进行下采样
            if self.downsample != 1.0:
                img_wh = (int(img.width * self.downsample), int(img.height * self.downsample))
                img = img.resize(img_wh, Image.LANCZOS)
                
            img = self.transform(img)  # (4, h, w)
           
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            
            img = img.float()

            self.all_rgbs.append(img)
            
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3),从c2w得到原点，转换为世界坐标系后得到方向
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            
            self.time_temps += [count * time_step]
            
            count += 1

        # img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        # idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        # self.num_img = 0
        # for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#
        #     self.num_img += 1

        #     frame = self.meta['frames'][i]
        #     pose = np.array(frame['transform_matrix']) @ self.blender2opencv
        #     # pose = np.array(frame['transform_matrix']) # dnerf中没有做
        #      # timenet前置
        #     # self.pose_temps.append(np.array(frame['transform_matrix'])) 
        #     # c2w矩阵
        #     c2w = torch.FloatTensor(pose)
        #     self.poses += [c2w]

        #     image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
        #     self.image_paths += [image_path]

        #     # 将图像读取为numpy数组,为timenet前序处理做准备
        #     img = Image.open(image_path)
        #     # self.image_temps.append(imageio.imread(image_path))

        #     if self.downsample!=1.0:
        #         img = img.resize(self.img_wh, Image.LANCZOS) # 根据是否下采样调整图像
        #     img = self.transform(img)  # (4, h, w)
           
        #     img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
        #     img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

        #     if self.normalization and self.normalization != 'None':
        #         img = normalize_image(img, self.normalization)
            
        #     self.all_rgbs += [img] 


        #     rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3),从c2w得到原点，转换为世界坐标系后得到方向
        #     self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        #     # 读取时间戳
        #     self.time_temps += [frame['time']]
         # timenet前置
        # self.image_temps = (np.array(self.image_temps) / 255.).astype(np.float32) # keep all 4 channels (RGBA) (50, 800, 800, 4)
        self.time_temps = np.array(self.time_temps).astype(np.float32) # (475,)
        # self.pose_temps = np.array(self.pose_temps).astype(np.float32) # (50, 4, 4)

        self.poses = torch.stack(self.poses)
         # timenet前置
        # self.all_imgs.append(self.image_temps)
        self.all_times.append(self.time_temps)
        # self.all_poses.append(self.pose_temps)
         # timenet前置
        # self.all_imgs = np.concatenate(self.all_imgs, 0) # (50, 800, 800, 4)
        self.all_times = np.concatenate(self.all_times, 0) # (475,)
        # self.all_poses = np.concatenate(self.all_poses, 0) # (50, 4, 4)

        # self.all_imgs = self.all_imgs[...,:3]*self.all_imgs[...,-1:] + (1.-self.all_imgs[...,-1:]) # (50, 800, 800, 3)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 6)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def read_test_meta(self):

        # with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
        #     self.meta = json.load(f)

        w, h = self.img_wh # 720 ,960
        
        (
        gt_training_cam_T_wi, # torch.Size([475, 4, 4])
        gt_testing_cam_T_wi_list, # two cameras torch.Size([212, 4, 4]) torch.Size([320, 4, 4])
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov, # 52.071839030522504
        gt_testing_fov_list, # [52.5902636880661, 52.41911396036227]
        gt_training_cxcy_ratio,
        gt_testing_cxcy_ratio_list,
        ) = load_iphone_gt_poses(src = self.split_dir, t_subsample = 1)
        
        # self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        # self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh 根据是否downsample去调整焦距

        self.focal_1 = 0.5 * w / np.tan(np.deg2rad(0.5 * gt_testing_fov_list[0]))
        self.focal_2 = 0.5 * w / np.tan(np.deg2rad(0.5 * gt_testing_fov_list[1]))
            
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions_1 = get_ray_directions(h, w, [self.focal_1,self.focal_1])  # (h, w, 3) (960,720,3)
        self.directions_1 = self.directions_1 / torch.norm(self.directions_1, dim=-1, keepdim=True) # 每个directions，三维张量正则化
        self.directions_2 = get_ray_directions(h, w, [self.focal_2,self.focal_2])  # (h, w, 3) (960,720,3)
        self.directions_2 = self.directions_2 / torch.norm(self.directions_2, dim=-1, keepdim=True) # 每个directions，三维张量正则化
        # 定义相机的内参矩阵
        self.intrinsics_1 = torch.tensor([[self.focal_1,0,w/2],[0,self.focal_1,h/2],[0,0,1]]).float()
        self.intrinsics_2 = torch.tensor([[self.focal_2,0,w/2],[0,self.focal_2,h/2],[0,0,1]]).float()       
        # 为timenet预备
        # self.image_temps = []
        self.time_temps = []
        # self.pose_temps = []

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_times = [] # timenet前置
        # self.all_imgs = [] # timenet前置
        # self.all_poses = [] # timenet前置
        # self.downsample=1.0
        
        camera_dir = osp.join(self.split_dir, "test_cameras")
        image_dir = osp.join(self.split_dir, "test_images")
        camera_files = sorted([f for f in os.listdir(camera_dir) if f.endswith('.json')])
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        test_1_time_boundary = max(gt_testing_tids_list[0]) + 1
        test_2_time_boundary = max(gt_testing_tids_list[1]) + 1
        # camera_counts = self.count_camera_files(camera_dir)
        former_boundary = len(gt_testing_tids_list[0]) # 212
        
        time_step_1 = 1.0 / test_1_time_boundary
        time_step_2 = 1.0 / test_2_time_boundary
        
        count = 1

        # 遍历cameras和images
        for cam_file, img_file in tqdm(zip(camera_files, image_files), total=len(camera_files), desc="Loading Data"):
            # 读取camera文件
            cam_path = os.path.join(camera_dir, cam_file)
            with open(cam_path, 'r') as f:
                camera_data = json.load(f)
                
            # 提取pose相关数据
            orientation = np.array(camera_data['orientation'])  # 3x3 旋转矩阵
            position = np.array(camera_data['position'])        # 位置向量 (3,)
        
            # 生成4x4的c2w矩阵 (相机到世界坐标)
            c2w = np.eye(4)
            c2w[:3, :3] = orientation.T  # 旋转矩阵转置，从世界到相机变为相机到世界
            c2w[:3, 3] = position        # 平移向量
            c2w = torch.FloatTensor(c2w)
            self.poses += [c2w]

            # 读取对应的图像文件
            img_path = os.path.join(image_dir, img_file)
            self.image_paths += [img_path]
            
            img = Image.open(img_path)

            # 对图像进行下采样
            if self.downsample != 1.0:
                img_wh = (int(img.width * self.downsample), int(img.height * self.downsample))
                img = img.resize(img_wh, Image.LANCZOS)
                
            img = self.transform(img)  # (4, h, w)
           
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            img = img.float()  # (H*W, 3)

            self.all_rgbs.append(img)
            
            if count <= former_boundary:
                rays_o, rays_d = get_rays(self.directions_1, c2w)  # both (h*w, 3),从c2w得到原点，转换为世界坐标系后得到方向
                self.time_temps += [(gt_testing_tids_list[0][count - 1] + 1) / test_1_time_boundary]
            else:
                rays_o, rays_d = get_rays(self.directions_2, c2w)
                self.time_temps += [(gt_testing_tids_list[1][count - former_boundary - 1] + 1) / test_2_time_boundary]
                
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            
            count += 1

        # img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        # idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        # self.num_img = 0
        # for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#
        #     self.num_img += 1

        #     frame = self.meta['frames'][i]
        #     pose = np.array(frame['transform_matrix']) @ self.blender2opencv
        #     # pose = np.array(frame['transform_matrix']) # dnerf中没有做
        #      # timenet前置
        #     # self.pose_temps.append(np.array(frame['transform_matrix'])) 
        #     # c2w矩阵
        #     c2w = torch.FloatTensor(pose)
        #     self.poses += [c2w]

        #     image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
        #     self.image_paths += [image_path]

        #     # 将图像读取为numpy数组,为timenet前序处理做准备
        #     img = Image.open(image_path)
        #     # self.image_temps.append(imageio.imread(image_path))

        #     if self.downsample!=1.0:
        #         img = img.resize(self.img_wh, Image.LANCZOS) # 根据是否下采样调整图像
        #     img = self.transform(img)  # (4, h, w)
           
        #     img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
        #     img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

        #     if self.normalization and self.normalization != 'None':
        #         img = normalize_image(img, self.normalization)
            
        #     self.all_rgbs += [img] 


        #     rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3),从c2w得到原点，转换为世界坐标系后得到方向
        #     self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        #     # 读取时间戳
        #     self.time_temps += [frame['time']]
         # timenet前置
        # self.image_temps = (np.array(self.image_temps) / 255.).astype(np.float32) # keep all 4 channels (RGBA) (50, 800, 800, 4)
        self.time_temps = np.array(self.time_temps).astype(np.float32) # (475,)
        # self.pose_temps = np.array(self.pose_temps).astype(np.float32) # (50, 4, 4)

        self.poses = torch.stack(self.poses)
         # timenet前置
        # self.all_imgs.append(self.image_temps)
        self.all_times.append(self.time_temps)
        # self.all_poses.append(self.pose_temps)
         # timenet前置
        # self.all_imgs = np.concatenate(self.all_imgs, 0) # (50, 800, 800, 4)
        self.all_times = np.concatenate(self.all_times, 0) # (475,)
        # self.all_poses = np.concatenate(self.all_poses, 0) # (50, 4, 4)

        # self.all_imgs = self.all_imgs[...,:3]*self.all_imgs[...,-1:] + (1.-self.all_imgs[...,-1:]) # (50, 800, 800, 3)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 6)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)  torch.Size([532, 691200, 6])
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
    
    def count_camera_files(self, camera_dir):
        """
        统计每种相机的数据文件数量
        :param camera_dir: 包含相机数据的文件夹路径
        :return: 每种相机的文件数量统计
        """
        # 初始化计数字典
        camera_counts = defaultdict(int)

        # 遍历文件夹中的所有文件
        for file_name in os.listdir(camera_dir):
            # 过滤掉非 JSON 文件
            if not file_name.endswith('.json'):
                continue

            # 提取文件前缀（相机 ID）
            camera_id = file_name.split('_')[0]  # 如 "1_00417.json" 的前缀是 "1"

            # 累加对应相机的计数
            camera_counts[camera_id] += 1

        return dict(camera_counts)

            
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