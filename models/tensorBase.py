# Based on TensoRF code from https://github.com/apchenstu/TensoRF
# Modifications and/or extensions have been made for specific purposes in this project.

import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
import math

def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)
        print('ALPHA GRID MASK')
        print("alpha_volume.shape", alpha_volume.shape)
        print("gridSize", self.gridSize)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


# class MLPRender_Fea(torch.nn.Module): # 要改一下，加入时间编码
#     def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
#         super(MLPRender_Fea, self).__init__()
#         times_dim = 30
#         self.in_mlpC = 2*4*3 + 2*feape*inChanel + 3 + inChanel + times_dim + 2*10*3 + 3
#         self.viewpe = viewpe # 2（tineuvox是4）
#         self.feape = feape # 2 
#         self.viewpe = 4
#         self.pospe = 10 # tineuvox是10）
#         layer1 = torch.nn.Linear(self.in_mlpC, featureC)
#         layer2 = torch.nn.Linear(featureC, featureC)
#         layer3 = torch.nn.Linear(featureC,3)

#         self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
#         torch.nn.init.constant_(self.mlp[-1].bias, 0)

#     def forward(self, pts, viewdirs, features, times_feature):
#         # indata = [features, viewdirs]
#         indata = [times_feature, pts, viewdirs, features]
#         if self.pospe > 0:
#             indata += [positional_encoding(pts, self.pospe)]
#         if self.viewpe > 0:
#             indata += [positional_encoding(viewdirs, self.viewpe)]
#         if self.feape > 0:
#             indata += [positional_encoding(features, self.feape)]
#         mlp_in = torch.cat(indata, dim=-1)
#         rgb = self.mlp(mlp_in)
#         rgb = torch.sigmoid(rgb)

#         return rgb
    
#     def num_params(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLPRender_Fea(torch.nn.Module): # 要改一下，加入时间编码
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128, add_cam = False, timenet_output = 17, fused = None):
        super(MLPRender_Fea, self).__init__()
        self.add_cam = add_cam
        self.fused = fused
        if self.fused:
            if self.add_cam:
                self.in_mlpC =  2*4*3 + 3 + 256 + timenet_output  
            else: # 2 * view * 3 + 2*2* app_dim + 3 + app_dim + 256
                self.in_mlpC = 2*4*3  + 3 + 256
            self.viewpe = viewpe # 2（tineuvox是4）
            self.feape = feape # 2 
            self.viewpe = 4
            self.pospe = 10 # tineuvox是10）
            self.feature_linears = nn.Linear(256, 256)
            layer1 = torch.nn.Linear(self.in_mlpC, featureC)
            layer2 = torch.nn.Linear(featureC, featureC)
            layer3 = torch.nn.Linear(featureC,3)

            self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
            torch.nn.init.constant_(self.mlp[-1].bias, 0)
        else:
            if self.add_cam:
                self.in_mlpC =  2*4*3 + 2*feape*inChanel + 3 + inChanel + 256 + timenet_output
            else:
                self.in_mlpC = 2*4*3 + 2*feape*inChanel + 3 + inChanel + 256
            self.viewpe = viewpe # 2（tineuvox是4）
            self.feape = feape # 2 
            self.viewpe = 4
            self.pospe = 10 # tineuvox是10）
            self.feature_linears = nn.Linear(256, 256)
            layer1 = torch.nn.Linear(self.in_mlpC, featureC)
            layer2 = torch.nn.Linear(featureC, featureC)
            layer3 = torch.nn.Linear(featureC,3)

            self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
            torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, input_h, viewdirs, features = None, camera_feature = None):
        # indata = [features, viewdirs]
        feature = self.feature_linears(input_h)
        if self.fused:
            if self.add_cam:
                indata = [feature, camera_feature, viewdirs]
            else:
                indata = [feature, viewdirs] # 256 + 3 
            
            if self.viewpe > 0:
                indata += [positional_encoding(viewdirs, self.viewpe)] # 24
        else:
            if self.add_cam:
                indata = [feature, camera_feature, viewdirs, features]
            else:
                indata = [feature, viewdirs, features] # 256 + 3 + 27
            
            if self.viewpe > 0:
                indata += [positional_encoding(viewdirs, self.viewpe)] # 24
            if self.feape > 0:
                indata += [positional_encoding(features, self.feape)] #  2 * 2* 27
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
# 对时间和位置编码进行拼接并送入deformation网络出新的偏移点
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_views=3, input_ch_time=9, skips=[],):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self._time, self._time_out = self.create_net()

    def create_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch
            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)
        return net_final(h)

    def forward(self, input_pts, ts):
        dx = self.query_time(input_pts, ts, self._time, self._time_out)
        input_pts_orig = input_pts[:, :3] # 原始三维的点

        # # 打印 input_pts[:, :3] 的极端值
        # print(f"input_pts[:, :3] min: {torch.min(input_pts_orig).item()}, max: {torch.max(input_pts_orig).item()}")

        # # 打印 dx 的极端值
        # print(f"dx min: {torch.min(dx).item()}, max: {torch.max(dx).item()}")

        out=input_pts_orig + dx # 偏移量加上原始的三维点
        return out


class TimeNet(nn.Module):
    def __init__(self, times_ch, timenet_output, timenet_width=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(times_ch, timenet_width),
            nn.ReLU(),
            nn.Linear(timenet_width, timenet_output)
        )

    def forward(self, x):
        return self.fc(x)
    
class FeatureNet(nn.Module):
    def __init__(self, input_dim, featurenet_width, featurenet_depth):
        super().__init__()
        self.fea = nn.Sequential(
            nn.Linear(input_dim, featurenet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(featurenet_width, featurenet_width), nn.ReLU(inplace=True))
                for _ in range(featurenet_depth-1)
            ],
            )
    def forward(self, x):
        return self.fea(x)
    
class DensityNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.dens = nn.Linear(input_dim, 1)
        self.dens = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.dens(x)
    
class CameraNet(nn.Module):
    def __init__(self, timebase_pe, viewbase_pe, timenet_output):
        super().__init__()
        times_ch = 2*timebase_pe+1
        views_ch = 3+3*viewbase_pe*2+timenet_output # 重新更新此变量用于后续rgbnet处理hyper形式
        self.camnet = nn.Sequential(
        nn.Linear(times_ch, 256), nn.ReLU(inplace=True),
        nn.Linear(256, timenet_output))
    
    def forward(self, x):
        return self.camnet(x)
    
class TimeSpaceFuser(nn.Module):
    def __init__(self, dim_xyz, dim_time, dim_hidden):
        super().__init__()
        self.q_proj = nn.Linear(dim_xyz, dim_hidden)   # 输入：φ(xyz)，输出：[N, dim_hidden]
        self.k_proj = nn.Linear(dim_time, dim_hidden)    # 输入：φ(time)，输出：[N, dim_hidden]
        self.v_proj = nn.Linear(dim_time, dim_time)        # 输入：φ(time)，输出：[N, dim_time]
        self.out_proj = nn.Linear(dim_time, dim_time)      # 保持输出维度与 φ(time) 一致

        self.gate_net = nn.Sequential(
            nn.Linear(dim_xyz + dim_time, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_time),
            nn.Sigmoid()
        )

    def forward(self, phi_xyz, phi_time):
        # Cross Attention 部分：让每个空间点查询时间特征
        q = self.q_proj(phi_xyz)                         # [N, dim_hidden]
        k = self.k_proj(phi_time)                        # [N, dim_hidden]
        v = self.v_proj(phi_time)                        # [N, dim_time]
        # 计算简单的点乘注意力（可以换成标准 multi-head attention）
        attn_weights = torch.softmax((q * k).sum(dim=-1, keepdim=True) / math.sqrt(q.size(-1)), dim=0)  # [N, 1]
        time_attn = attn_weights * v                     # [N, dim_time]

        # Gate 部分：利用拼接后的特征生成每个维度的融合权重
        gate_input = torch.cat([phi_xyz, phi_time], dim=-1)  # [N, dim_xyz + dim_time]
        gate = self.gate_net(gate_input)                # [N, dim_time]
        
        fused = gate * time_attn + (1 - gate) * phi_time  # [N, dim_time]
        return self.out_proj(fused)                     # [N, dim_time]

class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0, # 下面一行是新增的参数,记得在train的时候传入是否camera的指示变量,以及viewbase
                    timebase_pe = 8, voxel_dim = 6, gridbase_pe = 2, net_width = 256, defor_depth = 3, posbase_pe = 10, alpha_init = 0.001, add_cam = False, viewbase_pe = 4,
                    fea2denseAct = 'softplus', 
                    max_rank = 256, max_rank_density = -1, max_rank_appearance = -1, use_TTNF_sampling = False,
                    compression_alg = "compress_all", name = "TensorBase", should_shrink = False,
                    canonization = "None", fused = False, init_scale = 0.01, is_tensor_ring = True, **kargs
                    ):
        super(TensorBase, self).__init__()
        # 新增关于处理hyper数据集的指示变量
        self.add_cam = add_cam

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device=device
        self.name = name
        
        self.init_scale = init_scale

        self.fused = fused
        print("Using Fused Sigma and Appearance Features::", fused)

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio
        
        self.is_tensor_ring = is_tensor_ring

        # TT args
        self.max_rank = max_rank # 256 
        self.use_TTNF_sampling = use_TTNF_sampling # 0
        self.compression_alg = compression_alg # 'compress_all'
        self.canonization = canonization # 'None'
        self.should_shrink = should_shrink # 0
        self.max_rank_density = max_rank_density  # 200
        self.max_rank_appearance = max_rank_appearance # 280

        self.gridSize = gridSize # [16, 16, 16]

        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.time_poc = torch.FloatTensor([(2**i) for i in range(timebase_pe)]).to(device) # tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128.])
        self.pos_poc = torch.FloatTensor([(2**i) for i in range(posbase_pe)]).to(device) # tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
        self.grid_poc = torch.FloatTensor([(2**i) for i in range(gridbase_pe)]).to(device) # tensor([1., 2.], device='cuda:0')

                # determine the density bias shift
        self.alpha_init = alpha_init # 0.001
        self.act_shift = np.log(1/(1-alpha_init) - 1) # -6.906754778648465
        print('TiNeuVox: set density bias shift to', self.act_shift)

        self.init_svd_volume(gridSize[0], device)
        # 新模块
        self.init_time_net(timebase_pe, voxel_dim, gridbase_pe)
        self.init_feature_net(voxel_dim, gridbase_pe, posbase_pe, net_width, featurenet_depth = 2)
        self.init_density_net(net_width)
        self.init_defor_net(net_width, defor_depth, posbase_pe, voxel_dim, gridbase_pe)
        
        # 空间为主查询更贴合的时间特征
        self.init_space_time_attention(63, 30, 64)
        
        if self.add_cam:
            self.init_camera_net(timebase_pe, viewbase_pe, self.timenet_output)
        
        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_space_time_attention(self, dim_xyz, dim_time, dim_hidden):       
        self.time_space_attention = TimeSpaceFuser(dim_xyz, dim_time, dim_hidden).to(self.device)
        print('space_time_attention', self.time_space_attention)
        
    def init_camera_net(self, timebase_pe, viewbase_pe, timenet_output):
        self.views_ch = 3+3*viewbase_pe*2+timenet_output
        self.camera_net = CameraNet(timebase_pe, viewbase_pe, timenet_output).to(self.device)
        print('camnet', self.camera_net)
        
    
    def init_time_net(self, timebase_pe, voxel_dim, gridbase_pe):
        times_ch = 2*timebase_pe+1
        self.timenet_output = voxel_dim+voxel_dim*2*gridbase_pe

        self.timenet = TimeNet(times_ch, self.timenet_output).to(self.device)

        print("times_ch", times_ch, "timenet_output", self.timenet_output)


    def init_defor_net(self, net_width, defor_depth, posbase_pe, voxel_dim, gridbase_pe):
        timenet_output = voxel_dim+voxel_dim*2*gridbase_pe

        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=3+3*posbase_pe*2, input_ch_time=timenet_output).to(self.device)
        
        print("net_width", net_width, "defor_depth", defor_depth, "input_ch", 3+3*posbase_pe*2, "input_ch_time", timenet_output)

    def init_feature_net(self, voxel_dim, gridbase_pe, posbase_pe, featurenet_width, featurenet_depth):
        # grid_dim = voxel_dim*3+voxel_dim*3*2*gridbase_pe 
        if self.fused:
            input_dim = 28+self.timenet_output+0+0+3+3*posbase_pe*2
            self.feature_net = FeatureNet(input_dim, featurenet_width, featurenet_depth).to(self.device)
        else:
            grid_dim = 5
            input_dim = grid_dim+self.timenet_output+0+0+3+3*posbase_pe*2
            self.feature_net = FeatureNet(input_dim, featurenet_width, featurenet_depth).to(self.device)

        print('input_dim', input_dim, 'featurenet_width', featurenet_width, ' featurenet_depth',  featurenet_depth)

    def init_density_net(self, input_dim):
        self.density_net = DensityNet(input_dim).to(self.device)

        print('input_dim', input_dim)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea': # lego进的是这个
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC, self.add_cam, self.timenet_output, self.fused).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1)) # aabb tensor([[-1.5000, -1.5000, -1.5000],[ 1.5000,  1.5000,  1.5000]], device='cuda:0')        
        print("grid size", gridSize) # [16, 16, 16]
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize # 归一化到[-1,1]里面
        
        
        self.gridSize = torch.tensor(gridSize, dtype=torch.long, device=self.device)
        print("aabbSize: ", self.aabbSize) # tensor([3., 3., 3.], device='cuda:0')
        print("gridSize: ",gridSize) # [16, 16, 16]
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize) # sampling step size:  tensor(0.1000, device='cuda:0')
        print("sampling number: ", self.nSamples)  # sampling number:  52
    
        self.radius = (self.gridSize-1) * 0.5  # [7.5,7.5,7.5]
    def scale_xyz(self, xyz_sampled):
        #return (xyz_sampled+1) * 0.5 * (self.gridSize-1)
        return xyz_sampled * self.radius + self.radius
    
    
    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        res = (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1
        return res

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001, lr_init_featurenet = 0.0008, lr_init_deformation_net = 0.0006, lr_init_density_net = 0.0008, lr_init_time_net = 0.0008):
        pass

    def get_kwargs(self):

        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
            
        if self.name == "TensorTT":
            if self.fused:
                skeleton = self.vox_fused.skeleton
                ckpt.update({'skeleton':skeleton})
                
            else:
                skeleton_rgb = self.vox_rgb.skeleton
                skeleton_sigma = self.vox_sigma.skeleton
                ckpt.update({'skeleton_rgb':skeleton_rgb})
                ckpt.update({'skeleton_sigma':skeleton_sigma})
                

        torch.save(ckpt, path)
        

    def load(self, ckpt):
        print('========> Loading model from checkpoint ...')
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        
        if self.name == "TensorTT": # if the model is TensorTT then load the skeleton of the TT
            if self.fused:
                # Convert list of parameters to a dictionary
                params_vox_fused = {int(i): p for i, p in enumerate(param_value for param_name, param_value in ckpt['state_dict'].items() if 'vox_fused' in param_name)}
                self.vox_fused.load_tn(params_vox_fused, ckpt['skeleton'])
            else:
                # Convert lists of parameters to dictionaries
                params_vox_rgb = {int(i): p for i, p in enumerate(param_value for param_name, param_value in ckpt['state_dict'].items() if 'vox_rgb' in param_name)}
                params_vox_sigma = {int(i): p for i, p in enumerate(param_value for param_name, param_value in ckpt['state_dict'].items() if 'vox_sigma' in param_name)}
                self.vox_rgb.load_tn(params_vox_rgb, ckpt['skeleton_rgb'])
                self.vox_sigma.load_tn(params_vox_sigma, ckpt['skeleton_sigma'])

        self.load_state_dict(ckpt['state_dict'])


    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        # 新增ray_id！！！！
        ray_id = torch.arange(rays_o.shape[0], device=rays_o.device).unsqueeze(1).repeat(1, N_samples)

        return rays_pts, interpx, ~mask_outbbox, ray_id

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize # 用于缩小采样点位置的系数，使其控制在盒内
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far) # 非near和far中间的都被剔除了
        rng = torch.arange(N_samples)[None].float() # 0 到 N_samples-1 的采样索引。[None] 将一维张量转换为二维张量，使其形状变为 (1, N_samples)，根据是否在训练阶段，添加随机噪声
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]]) # 生成一个与 rng 的第一个维度匹配的随机噪声张量，并将其添加到 rng 中，增加采样点的随机性。
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step) # 每条光线的取样点位置的方向系数确定
        
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        # 新增ray_id！！！！
        ray_id = torch.arange(rays_o.shape[0], device=rays_o.device).unsqueeze(1).repeat(1, N_samples)

        return rays_pts, interpx, ~mask_outbbox, ray_id


    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize # gridSize in TT starts from [32,32,32] --> [64,64,64] (...) --> [256,256,256]

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples # self.aabb [-1.5,-1.5,-1.5] [1.5,1.5,1.5]

        # 添加调试信息：打印 dense_xyz 的统计信息
        print(f"[getDenseAlpha] dense_xyz 范围: {dense_xyz.min().item()} 到 {dense_xyz.max().item()}")

        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))

        # 添加调试信息：打印 alpha 的统计信息
        print(f"[getDenseAlpha] alpha 最大值: {alpha.max().item()}, 最小值: {alpha.min().item()}, 非零值数量: {(alpha > 0).sum().item()}")
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        print('========> Updating alpha mask ...')
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        print('alpha.shape', alpha.shape)
        print(f"[updateAlphaMask] alpha.max(): {alpha.max().item()}, alpha.min(): {alpha.min().item()}")
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        print('dense_xyz.shape', dense_xyz.shape)
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]
        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        # self.alphaMask_thres = 1e-6
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0
        
        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)
        print(f"alpha shape: {alpha.shape}, max alpha value: {alpha.max()}, min alpha value: {alpha.min()}")

        valid_xyz = dense_xyz[alpha>0.5]
        print(f"valid_xyz shape: {valid_xyz.shape}")
        xyz_min = valid_xyz.amin(0) # this gives one value for each dim (x,y,z) .e.g. [-1.5,-1.5,-1.4] and [1.5,1.5,1.4]
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha) # number of voxels occupied by the object
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        print('===============================')
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, num_train_images, width, all_rays, all_rgbs, times, N_samples=256, chunk=10240*5, bbox_only=False, wandb=None):
        print('========> filtering rays ...')
        tt = time.time()

        # 确保 times 是一个 torch.Tensor
        if not isinstance(times, torch.Tensor):
            times = torch.tensor(times, dtype=torch.float32)
        
        # 计算所有射线的总数量 N
        N = torch.tensor(all_rays.shape[:-1]).prod() # tensor(64000000)
        times_tr = torch.ones([N,1])
        global_image_index = 0  # 全局图像索引

        # 修改 times_tr 对应的值
        n = num_train_images
        single_img_length = width ** 2
        for i in range(n):
                start_idx = i * single_img_length
                end_idx = (i + 1) * single_img_length
                times_tr[start_idx:end_idx] = times[i]

        mask_filtered = []
        # 将所有射线分成块，每块大小为 chunk
        # torch.arange(N)先生成形状为 (64000000,) 的张量，包含值 [0, 1, 2, ..., 63999999]
        # chunk = 10240 * 5，即 51200，结果是一个包含若干个小张量的元组，每个小张量的形状为 (51200,)
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            # idx_chunk提供了当前块的索引，选取出来并将当前块的射线移动到指定设备上
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only: # only first iteration 
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d) # 将射线方向中为0的值替换为一个很小的数值 1e-6，以避免除零错误。
                rate_a = (self.aabb[1] - rays_o) / vec # 计算每条射线各个方向上离理论最大值的t
                rate_b = (self.aabb[0] - rays_o) / vec # 计算每条射线各个方向上离理论最小值的t，此时t是多个三维张量组成的
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min # 当t_max <= t_min擦着过去或者没在box内

            else:   
                xyz_sampled, _,_,_= self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False) 
                alphas = self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1])
                mask_inbbox = (alphas > 0).any(-1) # gets all rays that have more than 0 as alpha value

            mask_filtered.append(mask_inbbox.cpu())
            
        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
        sum_mask_filtered = torch.sum(mask_filtered)
        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {sum_mask_filtered / N}, num_rays_now: {sum_mask_filtered}')
        if wandb is not None:
            wandb.log({'ray mask ratio': torch.sum(mask_filtered) / N, 'num_rays_now': torch.sum(mask_filtered)})
            
        print('===============================')
        return all_rays[mask_filtered], all_rgbs[mask_filtered], times_tr[mask_filtered]
    
    @torch.no_grad()
    def filtering_rays_immersive(self, all_rays, all_rgbs, all_times, N_samples=256, chunk=5120*5, bbox_only=False, wandb=None):
        print('========> filtering rays ...')  # 原本是 10240 * 5
        tt = time.time()

        # # 确保 times 是一个 torch.Tensor
        # if not isinstance(times, torch.Tensor):
        #     times = torch.tensor(times, dtype=torch.float32)
        
        # # 计算所有射线的总数量 N
        N = torch.tensor(all_rays.shape[:-1]).prod() 
        # times_tr = torch.ones([N,1])
        # global_image_index = 0  # 全局图像索引

        # # 修改 times_tr 对应的值
        # n = num_train_images
        # single_img_length = width * length
        # for i in range(n):
        #         start_idx = i * single_img_length
        #         end_idx = (i + 1) * single_img_length
        #         times_tr[start_idx:end_idx] = times[i]

        mask_filtered = []
        # 将所有射线分成块，每块大小为 chunk
        # torch.arange(N)先生成形状为 (64000000,) 的张量，包含值 [0, 1, 2, ..., 63999999]
        # chunk = 10240 * 5，即 51200，结果是一个包含若干个小张量的元组，每个小张量的形状为 (51200,)
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            # idx_chunk提供了当前块的索引，选取出来并将当前块的射线移动到指定设备上
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only: # only first iteration 
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d) # 将射线方向中为0的值替换为一个很小的数值 1e-6，以避免除零错误。
                rate_a = (self.aabb[1] - rays_o) / vec # 计算每条射线各个方向上离理论最大值的t
                rate_b = (self.aabb[0] - rays_o) / vec # 计算每条射线各个方向上离理论最小值的t，此时t是多个三维张量组成的
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min # 当t_max <= t_min擦着过去或者没在box内

            else:   
                xyz_sampled, _,_,_= self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False) 
                alphas = self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1])
                mask_inbbox = (alphas > 0).any(-1) # gets all rays that have more than 0 as alpha value

            mask_filtered.append(mask_inbbox.cpu())
            
        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
        sum_mask_filtered = torch.sum(mask_filtered)
        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {sum_mask_filtered / N}, num_rays_now: {sum_mask_filtered}')
        if wandb is not None:
            wandb.log({'ray mask ratio': torch.sum(mask_filtered) / N, 'num_rays_now': torch.sum(mask_filtered)})
            
        print('===============================')
        return all_rays[mask_filtered], all_rgbs[mask_filtered], all_times[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)

        # 添加调试信息：打印 alpha_mask 的统计信息
        print(f"[compute_alpha] alpha_mask 总数: {alpha_mask.sum().item()} / {alpha_mask.numel()}") 

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            if self.name == "TensorTT" or self.name == "TensorRing":
                xyz_sampled = self.scale_xyz( xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature) # TODO check threshold for softplus again
            sigma[alpha_mask] = validsigma
            
            # 添加调试信息：打印 sigma 的统计信息
            print(f"[compute_alpha] sigma 最大值: {sigma.max().item()}, 最小值: {sigma.min().item()}, 平均值: {sigma.mean().item()}")

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])
        
        # 添加调试信息：打印 alpha 的统计信息
        print(f"[compute_alpha] alpha 最大值: {alpha.max().item()}, 最小值: {alpha.min().item()}, 平均值: {alpha.mean().item()}")
        return alpha
    
    def poc_fre(self, input_data, poc_buf):
        input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
        input_data_sin = input_data_emb.sin()
        input_data_cos = input_data_emb.cos()
        input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
        return input_data_emb

    def rescale(self, ray_pts_delta):
        # 将输出重新映射到范围 [-1.5, 1.5] 中
        points_min = torch.min(ray_pts_delta, dim=0, keepdim=True)[0]  # 获取最小值
        points_max = torch.max(ray_pts_delta, dim=0, keepdim=True)[0]  # 获取最大值
        epsilon = 1e-6
        # 正确映射到 [-1.5, 1.5] 范围
        points_rescale_normalized = (ray_pts_delta - points_min) / (points_max - points_min + epsilon)  # 将范围映射到 [0, 1]
        points_rescale = 2.0 * self.aabb[1][0] * (points_rescale_normalized - 0.5)  # 将 [0, 1] 映射到 [-1.5, 1.5]
        return points_rescale

    def forward(self, rays_chunk, times_chunk, cameras_chunk = None, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid, ray_id = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            # 获取采样后的点，方向系数以及是否采样点是否在盒内的判断，新增ray_id！！！！ torch.Size([4096, 55, 3]);torch.Size([4096, 55]); torch.Size([4096, 55]);torch.Size([4096, 55])
            xyz_sampled, z_vals, ray_valid, ray_id = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1) # 错位相减，获取系数之差，再填补最后一个位置 torch.Size([4096, 55])
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape) # torch.Size([4096, 55, 3])
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            # ray_invalid[ray_valid] |= (~AlphaGridMask)
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device) # 表示除了最后一个维度之外的所有维度,torch.Size([4096, 55])
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device) # 返回 xyz_sampled 的前两个维度,并在最后添加一个大小为 3 的维度,torch.Size([4096, 55, 3])

        h_feature_temp = torch.zeros((*xyz_sampled.shape[:2], 256), device=xyz_sampled.device) # 111
        
        if self.add_cam==True:
            cam_emb = self.poc_fre(cameras_chunk, self.time_poc)
            cams_feature = self.camera_net(cam_emb)
            cams_feature = cams_feature.unsqueeze(1).expand(-1, xyz_sampled.size(1), -1) # torch.Size([4096, 55, 30])
        
        times_emb = self.poc_fre(times_chunk, self.time_poc)
        times_feature = self.timenet(times_emb) # torch.Size([4096, 30])，所有射线的时间特征
        
        xyz_unshifted = xyz_sampled.clone()
        if ray_valid.any():
            # xyz_sampled = self.normalize_coord(xyz_sampled) 原本的
            # points_to_sample = xyz_sampled[ray_valid]
            points_to_sample = xyz_sampled[ray_valid] # torch.Size([116084, 3])


            ray_id_to_sample = ray_id[ray_valid] # torch.Size([116084])

            rays_pts_emb = self.poc_fre(points_to_sample, self.pos_poc) # torch.Size([116084, 63])
            # ---------------------------------------------------------------------------------
            # ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature[ray_id_to_sample]) # torch.Size([116084, 3])
            # ---------------------------------------------------------------------------------
            
            fused_time = self.time_space_attention(rays_pts_emb, times_feature[ray_id_to_sample])  # [N, 30]
            ray_pts_delta = self.deformation_net(rays_pts_emb, fused_time)          # [N, 3]

            #-----------------------------------------------------------
            # 更新 dists 中ray_valid部分用以考虑偏移后的采样点间的距离变化
            # 初始化与 xyz_sampled 形状相同的 ray_pts_delta_full
            ray_pts_delta_full = torch.zeros_like(xyz_sampled)  # 形状：[num_rays, num_samples, 3]

            # 将变形后的点赋值回原始位置
            ray_pts_delta_full[ray_valid] = ray_pts_delta  # ray_pts_delta 是 deformation_net 的输出

            # 计算每条光线上相邻点之间的差异
            delta_diff = ray_pts_delta_full[:, 1:] - ray_pts_delta_full[:, :-1]  # 形状：[num_rays, num_samples - 1, 3]

            # 创建一个掩码，标记相邻的两个点都是有效的情况
            valid_dists_mask = ray_valid[:, 1:] & ray_valid[:, :-1]  # 形状：[num_rays, num_samples - 1]

            # 计算相邻点之间的距离（未应用掩码）
            dists_valid_unmasked = torch.norm(delta_diff, dim=-1)  # 形状：[num_rays, num_samples - 1]

            # 使用掩码，将无效的位置的距离设为零，避免就地修改
            dists_valid = dists_valid_unmasked * valid_dists_mask.float()

            # 填充距离以匹配原始的采样数量
            dists_new = torch.cat([dists_valid, torch.zeros((dists_valid.shape[0], 1), device=dists_valid.device)], dim=1)  # 形状：[num_rays, num_samples]

            # 更新 dists 中对应的位置，避免就地修改
            dists_clone = dists.clone()  # 克隆 dists，避免修改原始张量
            dists_clone[ray_valid] = dists_new[ray_valid]
            dists = dists_clone  # 更新 dists 引用

            # 更新 xyz_sampled 中的有效位置为偏移后的点，避免就地修改 
            xyz_sampled_clone = xyz_sampled.clone()
            xyz_sampled_clone[ray_valid] = points_to_sample
            xyz_sampled = xyz_sampled_clone

            points_to_sample = self.normalize_coord(ray_pts_delta)

            # 对更新后的 xyz_sampled 进行归一化处理，用于后面应用app_mask（原思路，但要更新偏移后的点）
            xyz_sampled = self.normalize_coord(xyz_sampled)

            if self.name == "TensorTT" or self.name == "TensorRing":
                points_to_sample = self.scale_xyz(points_to_sample) # 都是7.5+
            if self.fused:
                # ####################################### 原本的修改路线
                # # handling the fused case
                # rgb_valid, sigma_valid = self.compute_features(points_to_sample)
                
                # # Tin
                # sigma_feature = sigma_feature.unsqueeze(1) # 查一下需不需要
                # vox_feature_flatten_emb = self.poc_fre(sigma_feature, self.grid_poc)
                # h_feature = self.feature_net(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature[ray_id_to_sample]), -1))
                # h_feature_temp[ray_valid] = h_feature
                # density_result = self.density_net(h_feature)
                # density_result = density_result.squeeze(-1)
                # validsigma = self.feature2density(density_result)
                
                # # validsigma = self.feature2density(sigma_valid)
                # sigma[ray_valid] = validsigma
                # # i want rgb_features full which has same first dim as rgb but last dim is payload_dim
                # rgb_features_full = torch.zeros((*xyz_sampled.shape[:2], self.app_dim), device=xyz_sampled.device)
                # rgb_features_full[ray_valid] = rgb_valid
                # ###########################################
                
                ############################################适配Tineuvox的修改路线
                                # handling the fused case
                vox_feature_flatten = self.compute_features(points_to_sample)
                
                # Tin
                # vox_feature_flatten_emb = self.poc_fre(vox_feature_flatten, self.grid_poc) # 不位置编码了，按PuTT所说
                 # ---------------------------------------------------------------------------------
                # h_feature = self.feature_net(torch.cat((vox_feature_flatten, rays_pts_emb, times_feature[ray_id_to_sample]), -1))
                h_feature = self.feature_net(torch.cat((vox_feature_flatten, rays_pts_emb, fused_time), -1))
                h_feature_temp[ray_valid] = h_feature
                density_result = self.density_net(h_feature)
                density_result = density_result.squeeze(-1)
                validsigma = self.feature2density(density_result)
                
                # validsigma = self.feature2density(sigma_valid)
                sigma[ray_valid] = validsigma
                # # i want rgb_features full which has same first dim as rgb but last dim is payload_dim
                # rgb_features_full = torch.zeros((*xyz_sampled.shape[:2], self.app_dim), device=xyz_sampled.device)
                # rgb_features_full[ray_valid] = rgb_valid

            else:
                rgb_valid = None
                
                # Code for computing sigma and rgb separately
                sigma_feature = self.compute_densityfeature(points_to_sample) # 对于sigma来说，在特定表示方法下表示即可,torch.Size([116084])

                sigma_feature = sigma_feature.unsqueeze(1)  # 将输入从 [116084] 转换为 [116084, 1]
                # 新模块
                vox_feature_flatten_emb = self.poc_fre(sigma_feature, self.grid_poc)
                h_feature = self.feature_net(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature[ray_id_to_sample]), -1)) # [116084,256]

                h_feature_temp[ray_valid] = h_feature # 222

                density_result = self.density_net(h_feature)
                density_result = density_result.squeeze(-1)  # 使其维度变为 [116084]

                validsigma = self.feature2density(density_result)

                # validsigma = self.feature2density(sigma_feature)
                sigma[ray_valid] = validsigma # 将validsigma赋给sigma中ray_valid对应的部分

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale) # 要求dists和sigma同维度，出来的weights也是[4096, 55]

        app_mask = weight > self.rayMarch_weight_thres

        # 传times_feature[ray_id_time]
        ray_id_target = ray_id[app_mask] # torch.Size([73787])
        weight_direct_use = weight[app_mask]

        h_feature_input = h_feature_temp[app_mask] # 333
        
        xyz_unshifted = self.normalize_coord(xyz_unshifted)
        points_unshifted = xyz_unshifted[app_mask]

        if app_mask.any():
            points_to_sample = xyz_sampled[app_mask] # 这里要保证 xyz_sampled和weight一致

            if self.name == "TensorTT" or self.name == "TensorRing":
                points_to_sample = self.scale_xyz(points_to_sample)
                points_unshifted = self.scale_xyz(points_unshifted)
            if self.fused:
                ####################################### 原本的修改路线
                # # Use the previously computed rgb_features_full for the appearance features
                # valid_rgbs_filtered = rgb_features_full[app_mask] 

                # valid_rgbs = self.renderModule(points_to_sample, viewdirs[app_mask], valid_rgbs_filtered)

                # rgb[app_mask] = valid_rgbs
                ###########################################
                if self.add_cam:
                    valid_rgbs = self.renderModule(h_feature_input, viewdirs[app_mask], None, cams_feature[app_mask])
                else:
                    valid_rgbs = self.renderModule(h_feature_input, viewdirs[app_mask])
                rgb[app_mask] = valid_rgbs
            else:
                # Compute the appearance features when not fused
                app_features = self.compute_appfeature(points_to_sample) # torch.Size([73787, 27])
                # valid_rgbs = self.renderModule(points_to_sample, viewdirs[app_mask], app_features, times_feature[ray_id_time])
                # valid_rgbs = self.renderModule(points_unshifted, viewdirs[app_mask], app_features, times_feature[ray_id_time])
                
                if self.add_cam:
                    valid_rgbs = self.renderModule(h_feature_input, viewdirs[app_mask], app_features, cams_feature[app_mask])
                else:    
                    valid_rgbs = self.renderModule(h_feature_input, viewdirs[app_mask], app_features)
                rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1) # torch.Size([4096])
        rgb_map = torch.sum(weight[..., None] * rgb, -2) # torch.Size([4096, 3])

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        rgb_map = rgb_map.clamp(0,1) # torch.Size([4096, 3])

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]
        
        if is_train:
            return rgb_map, depth_map, acc_map ,valid_rgbs, ray_id_target, weight_direct_use# rgb, sigma, alpha, weight, bg_weight
        else:
            return rgb_map, depth_map, None, None, None, None