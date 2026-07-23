import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb

def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

def get_rays_of_a_view(H, W, K, c2w, ndc, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, mode=mode) 
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs

def get_mask(self, rays_o, rays_d, near, far, stepsize, **render_kwargs): # 两个文件看看是不是一样的效果，改成putt里面的
        '''Check whether the rays hit the geometry or not'''
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox]] = 1
        return hit.reshape(shape)

def compute_times_tr(rgb_tr_ori, times_list, train_poses, HW, Ks, ndc, model, render_kwargs):
    print('Computing times_tr...')
    assert len(rgb_tr_ori) == len(train_poses) == len(Ks) == len(HW) == len(times_list)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device

    # 计算所有图像的总像素数量
    N = sum(img.shape[0] * img.shape[1] for img in rgb_tr_ori)

    # 初始化 times_tr，全为1
    times_tr = torch.ones([N, 1], device=DEVICE)

    # 将 times_list 转换为 PyTorch 张量并调整形状
    times = torch.tensor(times_list, dtype=torch.float32, device=DEVICE)
    times = times.unsqueeze(-1)  # 调整形状为 [num_images, 1]

    top = 0
    for c2w, img, (H, W), K, time_one in zip(train_poses, rgb_tr_ori, HW, Ks, times):
        assert img.shape[:2] == (H, W)

        # 获取当前视图的光线（假设 get_rays_of_a_view 已定义）
        rays_o, rays_d, _ = get_rays_of_a_view(H=H, W=W, K=K, c2w=c2w, ndc=ndc)

        # 初始化掩码
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)

        # 计算掩码
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.get_mask(
                rays_o=rays_o[i:i+CHUNK],
                rays_d=rays_d[i:i+CHUNK],
                **render_kwargs
            ).to(DEVICE)

        # 有效像素的数量
        n = mask.sum()

        # 更新 times_tr，对于有效像素，乘以对应的 time_one
        times_tr[top:top + n] *= time_one

        # 更新索引 top
        top += n

    # 截断 times_tr，仅保留有效像素部分
    times_tr = times_tr[:top]

    print('Finished computing times_tr.')
    return times_tr



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

timebase_pe = 8
times_ch = 2*timebase_pe+1
timenet = TimeNet(times_ch, timenet_output=10) # timenet_output暂时未确定

# 解决times_poc的声明问题
register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)])) # 继承nn.Module的函数在里面声明

# 解决times_sel的声明问题
rgb_tr_ori = images.to('cpu' if cfg.data.load2gpu_on_the_fly else device)
N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
rgb_tr = torch.zeros([N,3], device=DEVICE)
train_N_rand = 4096
index_generator = batch_indices_generator(len(rgb_tr), train_N_rand)
batch_index_sampler = lambda:next(index_generator)
sel_i = batch_index_sampler()

times_flaten = compute_times_tr(rgb_tr_ori, times_list, train_poses, HW, Ks, ndc, model, render_kwargs)

times_sel = times_flaten[sel_i]

times_emb = poc_fre(times_sel, times_poc)

times_emb = timenet(times_emb)

