import concurrent.futures
import gc
import glob
import os

import json
import cv2
import numpy as np
import torch
from iopath.common.file_io import NativePathHandler, PathManager
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from scipy.spatial.transform import Rotation

from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender
from .pose_utils_gear import (
    average_poses_gear,
    center_poses_with,
    correct_poses_bounds,
    create_rotating_spiral_poses,
    create_spiral_poses,
    interpolate_poses,
)

from .ray_utils_gear import (
    get_ndc_rays_fx_fy,
    get_pixels_for_image,
    get_ray_directions_K,
    get_rays,
    sample_images_at_xy,
)

def perspective_to_fisheye(points, K, radial_distortion):
    return cv2.fisheye.undistortPoints(
        points[:, None], K, np.array([radial_distortion[0], radial_distortion[1], 0.0, 0.0]).astype(np.float32)
    )
    
def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


# def process_video(video_data_save, video_path, img_wh, downsample, transform):
#     """
#     Load video_path data to video_data_save tensor.
#     """
#     video_frames = cv2.VideoCapture(video_path)
#     count = 0
#     while video_frames.isOpened() and count < 50:
#         ret, video_frame = video_frames.read()
#         if ret:
#             video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
#             video_frame = Image.fromarray(video_frame)
#             if downsample != 1.0:
#                 img = video_frame.resize(img_wh, Image.LANCZOS)
#             img = transform(img)
#             video_data_save[count] = img.view(3, -1).permute(1, 0)
#             count += 1
#         else:
#             break
#     video_frames.release()
#     print(f"Video {video_path} processed.")
#     return None
def process_video(video_data_save, video_path, img_wh, downsample, transform):
    """
    Load video_path data to video_data_save tensor.
    Reads frames from 50 to 99 (skipping the first 50).
    """
    video_frames = cv2.VideoCapture(video_path)
    
    frame_idx = 0  # 记录当前处理的帧索引
    count = 0  # 记录存储的帧数
    
    while video_frames.isOpened() and count < 50:
        ret, video_frame = video_frames.read()
        if not ret:
            break  # 读取失败，退出
        
        if frame_idx < 50:
            frame_idx += 1  # 先跳过前 50 帧
            continue
        
        # 从第 50 帧开始处理
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        video_frame = Image.fromarray(video_frame)
        
        if downsample != 1.0:
            video_frame = video_frame.resize(img_wh, Image.LANCZOS)
        
        img = transform(video_frame)
        video_data_save[count] = img.view(3, -1).permute(1, 0)
        
        count += 1  # 记录已经存储的帧数
        frame_idx += 1  # 记录当前处理到的帧索引

    video_frames.release()
    print(f"Video {video_path} processed, frames 50-99 extracted.")
    return None


# define a function to process all videos
def process_videos(videos, skip_index, img_wh, downsample, transform, num_workers=1):
    """
    A multi-threaded function to load all videos fastly and memory-efficiently.
    To save memory, we pre-allocate a tensor to store all the images and spawn multi-threads to load the images into this tensor.
    """
    all_imgs = torch.zeros(len(videos) - 1, 50, img_wh[-1] * img_wh[-2], 3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # start a thread for each video
        current_index = 0
        futures = []
        for index, video_path in enumerate(videos):
            # skip the video with skip_index (eval video)
            if index == skip_index:
                continue
            else:
                future = executor.submit(
                    process_video,
                    all_imgs[current_index],
                    video_path,
                    img_wh,
                    downsample,
                    transform,
                )
                futures.append(future)
                current_index += 1
    return all_imgs


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


class ImmersiveDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=True
        # cal_fine_bbox=False,
        # N_vis=-1,
        # time_scale=1.0,
        # scene_bbox_min=[-1.0, -1.0, -1.0],
        # scene_bbox_max=[1.0, 1.0, 1.0],
        # N_random_pose=1000,
        # bd_factor=0.75,
        # eval_step=1,
        # eval_index=0,
        # sphere_scale=1.0,
    ):
        self.pmgr = PathManager()
        self.pmgr.register_handler(NativePathHandler())
        
         # 把这几个变量挪到下面
        self.sphere_scale=1.0
        self.N_random_pose=1000
        self.cal_fine_bbox=False

        self.img_wh = (
            int(2560 / downsample),
            int(1920 / downsample),
        )  # According to the neural 3D paper, the default resolution is 1024x768
        self.root_dir = datadir
        self.dataset_name = os.path.basename(self.root_dir)
        self.split = split
        self.downsample = 2560 / self.img_wh[0]
        self.is_stack = is_stack
        self.N_vis = 5 # 参考Hexplane的config
        self.time_scale = 1.0
        # 参考Hexplane的config
        scene_bbox_min = [-2.0, -2.0, -2.0]
        scene_bbox_max = [2.0, 2.0, 2.0]
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])

        self.world_bound_scale = 1.1
        self.bd_factor = 0.75
        self.eval_step = 1
        self.eval_index = 0
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()

        self.near = 0.0
        self.far = 1.0
        self.near_far = [self.near, self.far]  # NDC near far is [0, 1.0]
        self.white_bg = False
        self.ndc_ray = True
        self.depth_data = False
        self.use_ndc = True
        self.correct_poses = True
        
        self.read_meta()
        self.load_meta()
        print("meta data loaded")
        
    def read_meta(self):
        W, H = self.img_wh

        # Load meta
        with self.pmgr.open(os.path.join(self.root_dir, "models.json"), "r") as f:
            self.meta = json.load(f)

        # Populate vars
        self.video_paths = []
        self.intrinsics = []
        self.distortions = []
        self.poses = []
        self.focals = []
        val_idx = []
        for idx, camera in enumerate(self.meta):

            # DEBUGGING
            # if idx >= 10:
            #     break

            # Path
            self.video_paths.append(os.path.join(self.root_dir, camera["name"] + ".mp4"))

            # Intrinsics
            width_factor = self.img_wh[0] / 2560.0
            height_factor = self.img_wh[1] / 1920.0

            K = np.eye(3)
            K = np.array(
                [
                    [camera["focal_length"] * width_factor, 0.0, camera["principal_point"][0] * width_factor],
                    [0.0, camera["focal_length"] * height_factor, camera["principal_point"][1] * height_factor],
                    [0.0, 0.0, 1.0],
                ]
            )
            
            self.focals.append([camera["focal_length"] * width_factor, camera["focal_length"] * width_factor])
            
            self.intrinsics.append(K)

            # Distortion
            radial_distortion = np.array(camera["radial_distortion"])
            self.distortions.append(radial_distortion[:2])

            # Pose
            R = Rotation.from_rotvec(camera["orientation"]).as_matrix()
            T = np.array(camera["position"])

            pose = np.eye(4)
            pose[:3, :3] = R.T
            pose[:3, -1] = T

            pose_pre = np.eye(4)
            pose_pre[1, 1] *= -1
            pose_pre[2, 2] *= -1
            pose = pose_pre @ pose @ pose_pre

            if camera["name"] == "camera_0001":
                val_idx = idx
                center_pose = pose[None, :3, :4]
            
            self.poses.append(pose[:3, :4])

        self.images_per_frame = len(self.video_paths) # 45
        self.total_num_views = len(self.video_paths) # 45
        # self.intrinsics = np.stack([self.intrinsics for i in range(self.num_frames)]).reshape(-1, 3, 3) # (2250, 3, 3)
        # self.distortions = np.stack([self.distortions for i in range(self.num_frames)]).reshape(-1, 2) # (2250, 2)
        # self.poses = np.stack([self.poses for i in range(self.num_frames)]).reshape(-1, 3, 4) # (2250, 3, 4)
        self.K = self.intrinsics[0]
        
        # # Times
        # self.times = np.tile(np.linspace(0, 1, self.num_frames)[..., None], (1, self.images_per_frame))  # 复制多少次，每个video是0到1
        # self.times = self.times.reshape(-1)

        # self.camera_ids = np.tile(
        #     np.linspace(0, self.images_per_frame - 1, self.images_per_frame)[None, :], (self.num_frames, 1) 
        # )
        # self.camera_ids = self.camera_ids.reshape(-1)
        ## Bounds, common for all scenes
        if self.dataset_name in ["01_Welder"]:
            self.near = 0.25
            self.far = 6.0
            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])

        if self.dataset_name in ["02_Flames"]:
            self.near = 1.0
            self.far = 10.0
            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        if self.dataset_name in ["04_Truck"]:
            self.near = 0.5
            self.far = 10.0
            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        elif self.dataset_name in ["05_Horse"]:
            self.near = 0.5
            self.far = 45.0
            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        elif self.dataset_name in ["07_Car"]:
            self.near = 0.5
            self.far = 50.0
            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        elif self.dataset_name in ["09_Alexa_Meade_Exhibit"]:
            self.near = 0.5
            self.far = 30.0
            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        elif self.dataset_name in ["10_Alexa_Meade_Face_Paint_1"]:
            self.near = 0.25
            self.far = 6.0
            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([0.5, self.far])
        elif self.dataset_name in ["11_Alexa_Meade_Face_Paint_2"]:
            self.near = 0.25
            self.far = 6.0
            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([0.5, self.far])
        elif self.dataset_name in ["12_Cave"]:
            self.near = 0.5
            self.far = 20.0
            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        else:
            self.near = 0.5
            self.far = 10.0
            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        ## Correct poses, bounds
        poses = np.copy(self.poses)

        if self.use_ndc or self.correct_poses:
            self.poses, self.poses_avg = center_poses_with(poses, center_pose)

        self.near = self.bounds.min() * 0.95
        self.far = self.bounds.max() * 1.05
        
        self.near_far = [0.0, 1.0]
        self.near_fars = [self.near, self.far]
        ## Holdout validation images
        # val_indices = []

        # if len(self.val_set) > 0:
        #     val_indices += [frame * self.images_per_frame + val_idx for frame in range(self.num_frames)] # 长度为50，每隔45
        #     # val_indices += [frame * self.num_frames + 0 for frame in range(self.images_per_frame)] # 长度为45

        
        # train_indices = [i for i in range(len(self.poses)) if i not in val_indices]
        
        # self.valid_list = val_indices
        # self.train_list = train_indices
        
        # if self.val_all:
        #     val_indices = [i for i in train_indices]  # noqa

        # if self.split == "val" or self.split == "test":
        #     if not self.val_all and len(self.val_set) > 0:
        #         self.video_paths = [self.video_paths[val_idx]]
                

        #     self.intrinsics = self.intrinsics[val_indices]
        #     self.camera_ids = self.camera_ids[val_indices]
        #     self.distortions = self.distortions[val_indices]
        #     self.poses = self.poses[val_indices]
        #     self.times = self.times[val_indices]
            
        # elif self.split == "train":
        #     if not self.val_all and len(self.val_set) > 0:
        #         self.video_paths = [self.video_paths[i] for i in range(len(self.video_paths)) if i != val_idx]

        #     self.intrinsics = self.intrinsics[train_indices] # (2200, 3, 3)
        #     self.camera_ids = self.camera_ids[train_indices] # (2200,)
        #     self.distortions = self.distortions[train_indices] # (2200, 2)
        #     self.poses = self.poses[train_indices]
        #     self.times = self.times[train_indices]  # (2200,)
            

        # self.num_images = len(self.poses) # 2200  ;50
        # self.images_per_frame = len(self.video_paths)
        
    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # Read poses and video file paths.
        # poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        # poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        # self.near_fars = poses_arr[:, -2:] # (N_cams, 2)
        
        # 新注释掉使用 self.video_paths
        # videos = glob.glob(os.path.join(self.root_dir, "camera_*.mp4"))
        # videos = sorted(videos)
        # videos = [video for video in videos if "camera_0003.mp4" not in video]
        videos = self.video_paths
        
        # assert len(videos) == poses_arr.shape[0]
        
        # 准备valid spiral的位姿态前置
        # H, W, focal = poses[0, :, -1]
        # focal = focal / self.downsample
        # self.focal = [focal, focal]
        # poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # poses, pose_avg = center_poses(
        #     poses, self.blender2opencv
        # )  # Re-center poses so that the average is near the center. (N_cams, 3, 4) (4, 4)

        # near_original = self.near_fars.min()
        # scale_factor = near_original * 0.75
        # self.near_fars /= (
        #     scale_factor  # rescale nearest plane so that it is at z = 4/3.
        # )
        # poses[..., 3] /= scale_factor

        # # Sample N_views poses for validation - NeRF-like camera trajectory.
        # N_views = 120
        # self.val_poses = get_spiral(poses, self.near_fars, N_views=N_views) # (120, 4, 4)

        W, H = self.img_wh
        # self.directions = torch.tensor(
        #     get_ray_directions_blender(H, W, self.focal)
        # )  # (H, W, 3)

        if self.split == "train":
            # Loading all videos from this dataset requires around 50GB memory, and stack them into a tensor requires another 50GB.
            # To save memory, we allocate a large tensor and load videos into it instead of using torch.stack/cat operations.
            all_times = []
            all_rays = []
            count = 50

            for index in range(0, len(videos)):
                if (
                    index == self.eval_index
                ):  # the eval_index(0 as default) is the evaluation one. We skip evaluation cameras.
                    continue
                
                # self.directions = torch.tensor(
                # get_ray_directions_blender(H, W, self.focal[index])
                # )  # (H, W, 3)
                
                video_times = torch.tensor([i / (count - 1) for i in range(count)])
                all_times += [video_times]

                # rays_o, rays_d = get_rays(
                #     self.directions, torch.FloatTensor(self.poses[index])
                # )  # both (h*w, 3)
                rays_o, rays_d = self.get_coords(index)
                # rays_o, rays_d = ndc_rays_blender(H, W, focal, 1.0, rays_o, rays_d)
                # rays_o, rays_d = ndc_rays_blender(H, W, self.focals[index][0], 1.0, rays_o, rays_d)
                all_rays += [torch.cat([rays_o, rays_d], 1)]
                print(f"video {index} is loaded")
                gc.collect()

            # load all video images
            all_imgs = process_videos(
                videos,
                self.eval_index,
                self.img_wh,
                self.downsample,
                self.transform,
                num_workers=8,
            )
            all_times = torch.stack(all_times, 0) # torch.Size([44, 50])
            all_rays = torch.stack(all_rays, 0) # torch.Size([44, 307200, 6])
            # breakpoint()
            print("stack performed")
            N_cam, N_time, N_rays, C = all_imgs.shape # 44， 50， 307200， 3
            print(all_imgs.shape)
            self.image_stride = N_rays # 786432
            self.cam_number = N_cam # 44
            self.time_number = N_time # 50
            self.all_rgbs = all_imgs # torch.Size([44, 50, 307200, 3])
            self.all_times = all_times.view(N_cam, N_time, 1) # torch.Size([44, 50, 1])
            self.all_rays = all_rays.reshape(N_cam, N_rays, 6) # torch.Size([44, 307200, 6])
            self.all_times = self.time_scale * (self.all_times * 2.0 - 1.0) # torch.Size([44, 50, 1])
            self.global_mean_rgb = torch.mean(all_imgs, dim=1)

        else:
            index = self.eval_index
            video_imgs = []
            video_frames = cv2.VideoCapture(videos[index])
            while video_frames.isOpened():
                ret, video_frame = video_frames.read()
                if ret:
                    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                    video_frame = Image.fromarray(video_frame)
                    if self.downsample != 1.0:
                        img = video_frame.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img)
                    video_imgs += [img.view(3, -1).permute(1, 0)]
                else:
                    break
            video_imgs = torch.stack(video_imgs, 0)
            video_times = torch.tensor(
                [i / (len(video_imgs) - 1) for i in range(len(video_imgs))]
            )
            video_imgs = video_imgs[0 :: self.eval_step]
            video_times = video_times[0 :: self.eval_step]
            rays_o, rays_d = get_rays(
                self.directions, torch.FloatTensor(self.poses[index])
            )  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, self.focals[index][0], 1.0, rays_o, rays_d)
            all_rays = torch.cat([rays_o, rays_d], 1)
            gc.collect()
            N_time, N_rays, C = video_imgs.shape
            self.image_stride = N_rays # 786432
            self.time_number = N_time # 300
            self.all_rgbs = video_imgs.view(-1, N_rays, 3) # torch.Size([300, 768, 1024, 3])
            self.all_rays = all_rays # torch.Size([786432, 6])
            self.all_times = video_times
            self.all_rgbs = self.all_rgbs.view(
                -1, *self.img_wh[::-1], 3
            )  # (len(self.meta['frames]),h,w,3)
            self.all_times = self.time_scale * (self.all_times * 2.0 - 1.0)
            
    def get_coords(self, idx):
        # if self.split != "train" or self.split == "render":
        #     camera_id = 1
        # else:
        #     camera_id = self.camera_ids[idx]

        if self.split != "render":
            K = torch.FloatTensor(self.intrinsics[idx])
            distortion = self.distortions[idx]
        else:
            K = torch.FloatTensor(self.intrinsics[0])
            K[0, 0] *= 0.75
            K[1, 1] *= 0.75
            distortion = None

        c2w = torch.FloatTensor(self.poses[idx])

        # Undistort
        if distortion is not None:
            directions = get_ray_directions_K(self.img_wh[1], self.img_wh[0], K, centered_pixels=True).view(-1, 3)
            directions = perspective_to_fisheye(
                np.array(directions[..., :2]).astype(np.float32),
                np.eye(3).astype(np.float32),
                distortion.astype(np.float32),
            )[:, 0]
            directions = np.concatenate(
                [directions[..., 0:1], directions[..., 1:2], -np.ones_like(directions[..., -1:])], -1
            )

            directions = torch.tensor(directions)
            directions = torch.nn.functional.normalize(directions, dim=-1)
        else:
            directions = get_ray_directions_K(self.img_wh[1], self.img_wh[0], K, centered_pixels=True).view(-1, 3)

        # Convert to world space
        rays_o, rays_d = get_rays(directions, c2w)

        # Convert to NDC
        if self.use_ndc:
            rays_o, rays_d = self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
        else:
            rays = torch.cat([rays_o, rays_d], dim=-1)

        # Return
        return rays_o, rays_d
    
    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], 1, rays) # 注意这里之前失传self.near
    
    def __len__(self):
        if self.split == "train" and self.is_stack is True:
            return self.cam_number * self.time_number
        else:
            return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            if self.is_stack:
                cam_idx = idx // self.time_number
                time_idx = idx % self.time_number
                sample = {
                    "rays": self.all_rays[cam_idx],
                    "rgbs": self.all_rgbs[cam_idx, time_idx],
                    "time": self.all_times[cam_idx, time_idx]
                    * torch.ones_like(self.all_rays[cam_idx][:, 0:1]),
                }

            else:
                sample = {
                    "rays": self.all_rays[
                        idx // (self.time_number * self.image_stride),
                        idx % (self.image_stride),
                    ],
                    "rgbs": self.all_rgbs[idx],
                    "time": self.all_times[
                        idx // (self.time_number * self.image_stride),
                        idx
                        % (self.time_number * self.image_stride)
                        // self.image_stride,
                    ]
                    * torch.ones_like(self.all_rgbs[idx][:, 0:1]),
                }

        else:  # create data for each image separately
            if self.is_stack:
                sample = {
                    "rays": self.all_rays,
                    "rgbs": self.all_rgbs[idx],
                    "time": self.all_times[idx]
                    * torch.ones_like(self.all_rays[:, 0:1]),
                }

            else:
                sample = {
                    "rays": self.all_rays[idx % self.image_stride],
                    "rgbs": self.all_rgbs[idx],
                    "time": self.all_times[idx // self.image_stride]
                    * torch.ones_like(self.all_rays[:, 0:1]),
                }

        return sample

    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def get_val_rays(self):
        val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        rays_all = []  # initialize list to store [rays_o, rays_d]

        for i in range(val_poses.shape[0]):
            c2w = torch.FloatTensor(val_poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            if self.ndc_ray:
                W, H = self.img_wh
                rays_o, rays_d = ndc_rays_blender(
                    H, W, self.focal[0], 1.0, rays_o, rays_d
                )
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)