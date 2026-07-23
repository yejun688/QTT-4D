# Based on TensoRF code from https://github.com/apchenstu/TensoRF
# Modifications and/or extensions have been made for specific purposes in this project.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from tqdm.auto import tqdm
from opt import config_parser
import wandb
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from models.tensorTT import TensorTT
from models.tensoRF import TensorVMSplit, TensorCP

from dataLoader import dataset_dict
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

renderer = OctreeRender_trilinear_fast

@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    model = eval(args.model_name)(**kwargs)
    model.load(ckpt)

    alpha,_ = model.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=model.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    #args.downsample_train = 8
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs = update_kwargs(kwargs, args)
    kwargs.update({'device': device})

    model = eval(args.model_name)( max_rank_density = args.max_rank_density, max_rank_appearance = args.max_rank_appearance, **kwargs )
    print("model", model)

    model.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,model, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all_v2', exist_ok=True)
        evaluation(test_dataset,model, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,model, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def GM_function(x, gamma):
    return x**2 / (x**2 + gamma**2)


def GM_Resi(x, y, gamma):
    return 1 / 3 * torch.norm(GM_function(x - y, gamma), p=1, dim=-1)

        
def sample_ndc_data(train_dataset, iteration, is_immersive):
    if is_immersive :
        global_mean = train_dataset.global_mean_rgb.to(device)
        # hierarchical sampling from dyNeRF: hierachical sampling involves three stages of samplings.
        if iteration <= 40000:
            # Stage 1: randomly sample a single image from an arbitrary camera.
            # And sample a batch of rays from all the rays of the image based on the difference of global median and local values.
            # Stage 1 only samples key-frames, which is the frame every self.cfg.data.key_f_num frames.
            cam_i = np.random.choice(train_dataset.all_rgbs.shape[0]) # 先随机选出一个camera视角
            index_i = np.random.choice(
                train_dataset.all_rgbs.shape[1] // 4
            )
            rgb_train = (
                train_dataset.all_rgbs[cam_i, index_i * 4]
                .view(-1, 3)
                .to(device)
            )
            rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
            frame_time = train_dataset.all_times[
                cam_i, index_i * 4
            ]
            # Calcualte the probability of sampling each ray based on the difference of global median and local values.
            probability = GM_Resi(
                rgb_train, global_mean[cam_i], 0.001
            )
            select_inds = torch.multinomial(
                probability, 4096
            ).to(rays_train.device)
            rays_train = rays_train[select_inds]
            rgb_train = rgb_train[select_inds]
            frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time 
        elif (iteration<= 25000 + 40000):
            # Stage 2: basically the same as stage 1, but samples all the frames instead of only key-frames.
            cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
            index_i = np.random.choice(train_dataset.all_rgbs.shape[1])
            rgb_train = (
                train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(device)
            )
            rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
            frame_time = train_dataset.all_times[cam_i, index_i]
            probability = GM_Resi(
                rgb_train, global_mean[cam_i], 0.02
            )
            select_inds = torch.multinomial(
                probability, 4096
            ).to(rays_train.device)
            rays_train = rays_train[select_inds]
            rgb_train = rgb_train[select_inds]
            frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time           
        else:
            # Stage 3: randomly sample one frame and sample a batch of rays from the sampled frame.
            # TO sample a batch of rays from this frame, we calcualate the value changes of rays compared to nearby timesteps, and sample based on the value changes.
            cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
            N_time = train_dataset.all_rgbs.shape[1]
            # Sample two adjacent time steps within a range of 5 frames.
            index_i = np.random.choice(N_time)
            index_2 = np.random.choice(
                min(N_time, index_i + 5) - max(index_i - 5, 0)
            ) + max(index_i - 5, 0)
            rgb_train = (
                train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(device)
            )
            rgb_ref = (
                train_dataset.all_rgbs[cam_i, index_2].view(-1, 3).to(device)
            )
            rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
            frame_time = train_dataset.all_times[cam_i, index_i]
            # Calcualte the temporal difference between the two frames as sampling probability.
            probability = torch.clamp(
                1 / 3 * torch.norm(rgb_train - rgb_ref, p=1, dim=-1),
                min=0.1,
            )
            select_inds = torch.multinomial(
                probability, 4096
            ).to(rays_train.device)
            rays_train = rays_train[select_inds]
            rgb_train = rgb_train[select_inds]
            frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
    else:
        global_mean = train_dataset.global_mean_rgb.to(device)
         # hierarchical sampling from dyNeRF: hierachical sampling involves three stages of samplings.
        if iteration <= 40000:
            # Stage 1: randomly sample a single image from an arbitrary camera.
            # And sample a batch of rays from all the rays of the image based on the difference of global median and local values.
            # Stage 1 only samples key-frames, which is the frame every self.cfg.data.key_f_num frames.
            cam_i = np.random.choice(train_dataset.all_rgbs.shape[0]) # 先随机选出一个camera视角
            index_i = np.random.choice(
                train_dataset.all_rgbs.shape[1] // 30
            )
            rgb_train = (
                train_dataset.all_rgbs[cam_i, index_i * 30]
                .view(-1, 3)
                .to(device)
            )
            rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
            frame_time = train_dataset.all_times[
                cam_i, index_i * 30
            ]
            # Calcualte the probability of sampling each ray based on the difference of global median and local values.
            probability = GM_Resi(
                rgb_train, global_mean[cam_i], 0.001
            )
            select_inds = torch.multinomial(
                probability, 4096
            ).to(rays_train.device)
            rays_train = rays_train[select_inds]
            rgb_train = rgb_train[select_inds]
            frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time 
        elif (iteration<= 25000 + 40000):
            # Stage 2: basically the same as stage 1, but samples all the frames instead of only key-frames.
            cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
            index_i = np.random.choice(train_dataset.all_rgbs.shape[1])
            rgb_train = (
                train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(device)
            )
            rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
            frame_time = train_dataset.all_times[cam_i, index_i]
            probability = GM_Resi(
                rgb_train, global_mean[cam_i], 0.02
            )
            select_inds = torch.multinomial(
                probability, 4096
            ).to(rays_train.device)
            rays_train = rays_train[select_inds]
            rgb_train = rgb_train[select_inds]
            frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time           
        else:
            # Stage 3: randomly sample one frame and sample a batch of rays from the sampled frame.
            # TO sample a batch of rays from this frame, we calcualate the value changes of rays compared to nearby timesteps, and sample based on the value changes.
            cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
            N_time = train_dataset.all_rgbs.shape[1]
            # Sample two adjacent time steps within a range of 25 frames.
            index_i = np.random.choice(N_time)
            index_2 = np.random.choice(
                min(N_time, index_i + 25) - max(index_i - 25, 0)
            ) + max(index_i - 25, 0)
            rgb_train = (
                train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(device)
            )
            rgb_ref = (
                train_dataset.all_rgbs[cam_i, index_2].view(-1, 3).to(device)
            )
            rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
            frame_time = train_dataset.all_times[cam_i, index_i]
            # Calcualte the temporal difference between the two frames as sampling probability.
            probability = torch.clamp(
                1 / 3 * torch.norm(rgb_train - rgb_ref, p=1, dim=-1),
                min=0.1,
            )
            select_inds = torch.multinomial(
                probability, 4096
            ).to(rays_train.device)
            rays_train = rays_train[select_inds]
            rgb_train = rgb_train[select_inds]
            frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
    return rays_train, rgb_train, frame_time

def reconstruction(args):
    torch.autograd.set_detect_anomaly(True)
    
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    # test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsample_list = args.upsample_list # [250, 500, 2000,4000]
    if upsample_list is None or len(upsample_list) == 0:  # 在这一次调试中显然是不会执行upsample_list为1
        upsample_list = [args.n_iters+1]
    update_AlphaMask_list = args.update_AlphaMask_list  # [2000, 4000]
    if update_AlphaMask_list is None or len(update_AlphaMask_list) == 0:   # 在这一次调试中显然是不会执行update_AlphaMask_list为1
        update_AlphaMask_list = [args.n_iters+1]
    n_lamb_sigma = args.n_lamb_sigma # None
    n_lamb_sh = args.n_lamb_sh # None
    
    if args.add_timestamp: 
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}' # './log/PuTT-upsampling_12MB_blender/ship'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True) 
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)

    
    if args.i_wandb:
        wandb.init(
            project='ICML_dynamic',
            entity="rekkles2023",
            config=args,
            name=args.expname,
            dir=logfolder,
            force=True,  # makes the user provide wandb online credentials instead of running offline
        )
        wandb.tensorboard.patch(
            save=False,  # copies tb files into cloud and allows to run tensorboard in the cloud
            pytorch=True,
        )
        slurm_job_id = os.environ.get('SLURM_JOB_ID', 'default_run_id')
        # log the run id to wandb
        print("logging slurm_job_id to wandb", slurm_job_id)
        wandb.log({'slurm_job_id': slurm_job_id})


    
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device) # tensor([[-1.5000, -1.5000, -1.5000],[ 1.5000,  1.5000,  1.5000]], device='cuda:0')
    reso_cur = N_to_reso(args.N_voxel_init, aabb, model_name = args.model_name)
    print("reso_cur", reso_cur)
    print("aabb", aabb)
    # nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
    nSamples = cal_n_samples(reso_cur,args.step_ratio)
    nSamples = min(nSamples, args.nSamples)

    print("nSamples", nSamples)
    summary_writer.add_scalar('train/nSamples', nSamples, global_step=0) # add scalar nSamples 

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs = update_kwargs(kwargs, args, near_far, n_lamb_sigma, n_lamb_sh, reso_cur)
        kwargs.update({'device':device})
        model = eval(args.model_name)(**kwargs)
        model.load(ckpt)
    else:
        model = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct,
                    add_cam = args.add_cam,
                    use_TTNF_sampling = args.use_TTNF_sampling , rayMarch_weight_thres = args.rm_weight_mask_thre, fused = args.fused,
                    max_rank = args.max_rank, canonization = args.canonization, compression_alg = args.compression_alg, name = args.model_name,
                    is_tensor_ring = args.is_tensor_ring, init_scale = args.init_scale,
                    max_rank_density = args.max_rank_density, max_rank_appearance = args.max_rank_appearance, should_shrink = args.should_shrink,
                    ) # 在这里创建的模型为依照TensorTT类创建的对象

    compression_info = model.get_compression_values()
    tb_add_scalars(summary_writer, 'stats',compression_info, global_step=0)
    print("compression info", compression_info)

    grad_vars = model.get_optparam_groups(args.lr_init, args.lr_init_shader, args.lr_init_featurenet, args.lr_init_deformation_net, args.lr_init_density_net, args.lr_init_time_net)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    if args.lr_min_factor > 0:
        lr_min = args.lr_init * args.lr_min_factor
    else:
        lr_min = 0

    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    warmup_steps = -1

    #linear in logrithmic space
    if args.model_name != "TensorTT":
        N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsample_list)+1))).long()).tolist()[1:]
    else:
        final_res = np.round(args.N_voxel_final**(1/3)) # 256
        init_res = int(args.N_voxel_init**(1/3)) # 15
        factor = int(np.log2(final_res/init_res)) # 4
        N_voxel_list = [(args.N_voxel_init * 8 ** i) for i in range(1,factor+1)]
        print("Voxel_list: ", N_voxel_list) # [32768, 262144, 2097152, 16777216]

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]
    
    if not ndc_ray:
        if args.is_hyper:
            allrays, allrgbs ,times_flaten, allcams = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.time_flatten, train_dataset.cam_tr
        
            print("allrays", allrays.shape) 
            # log number of rays
            summary_writer.add_scalar('num_rays_now', allrays.shape[0], global_step=0)
        else:
            # 读取所有数据
            allrays, allrgbs ,alltimes = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_times
    
            print("allrays", allrays.shape) # allrays torch.Size([64000000, 6])
            # log number of rays
            summary_writer.add_scalar('num_rays_now', allrays.shape[0], global_step=0)
            
            if args.is_immersive:
                width, length = train_dataset.img_wh
                num_train_imgs = allrays.shape[0] / (width * length)
                print(num_train_imgs)
                num_train_imgs = int(num_train_imgs)
            else:
                num_train_imgs = train_dataset.num_img # 记录训练集个数
                width = train_dataset.img_wh[0]

            if not args.ndc_ray: # 光线：torch.Size([31938490, 6]) 颜色：torch.Size([31938490, 3]) 时间：torch.Size([31938490, 1])
                if args.is_immersive:
                    # allrays, allrgbs, times_flaten = model.filtering_rays_immersive(allrays, allrgbs, alltimes, bbox_only=True)
                    allrays, allrgbs, times_flaten = allrays, allrgbs ,alltimes
                    print('pass filtering')
                else:
                    allrays, allrgbs, times_flaten = model.filtering_rays(num_train_imgs, width, allrays, allrgbs, alltimes, bbox_only=True) # 筛选存在采样点在盒内的射线

        trainingSampler_random = SimpleSampler(allrays.shape[0], args.batch_size)
        if args.is_resampling:
            weighted_sampler = None
            tracker = RegionLossTracker(num_regions=64, momentum=0.9, device=device)
            
    Ortho_reg_weight = args.Ortho_weight # 0.0
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital # 1e-05
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app # 0.0，0.0
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    
    # Manually set to empty list if None
    upsample_ranks_iterations = args.upsample_ranks_iterations if args.upsample_ranks_iterations is not None else []
    upsample_ranks_app = args.upsample_ranks_app if args.upsample_ranks_app is not None else []
    upsample_ranks_density = args.upsample_ranks_density if args.upsample_ranks_density is not None else []
    print("upsample_ranks_iterations", upsample_ranks_iterations)
    print("upsample_ranks_app", upsample_ranks_app)
    print("upsample_ranks_density", upsample_ranks_density)

    # tqdm 是一个 Python 库，用于在循环中显示进度条
    # miniters每隔多少次迭代才刷新进度条 为10；参数指定进度条输出的位置。这里设置为标准输出 sys.stdout，即在终端中显示进度条。
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        if ndc_ray:
            rays_train, rgb_train, frame_time = sample_ndc_data(
                train_dataset, iteration, args.is_immersive
            ) # torch.Size([4096, 6]),torch.Size([4096, 3]),torch.Size([4096, 1]) 全是一帧的时间注意看看另两条分支出来的时间
            rays_train, rgb_train, frame_time = rays_train.to(device), rgb_train.to(device), frame_time.to(device)
            rgb_map, alphas_map, depth_map, weights, uncertainty, alphainv_last, valid_rgbs, ray_id_target, weight_direct_use = renderer(rays_train, frame_time, model, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True, is_hyper = args.is_hyper)
        else:
            if args.is_resampling:
                if iteration < 30000:
                    ray_idx = trainingSampler_random.nextids() # 每次随机生成4096个索引
                else:
                    if trainingSampler_weighted is None:
                        # 第一次切换：用现有区域 loss 生成概率
                        init_probs = tracker.get_probs(tau=0.2)
                        trainingSampler_weighted = WeightedRaySampler(
                            region_ids=train_dataset.all_region_ids,
                            region_probs=init_probs,
                            batch_size=args.batch_size
                        )
                        ray_idx = trainingSampler_weighted.nextids()
            else:
                ray_idx = trainingSampler_random.nextids() # 每次随机生成4096个索引
                
            if args.is_hyper:
                rays_train, rgb_train, times_train, cams_train= allrays[ray_idx].to(device), allrgbs[ray_idx].to(device), times_flaten[ray_idx].to(device), allcams[ray_idx].to(device)
                rgb_map, alphas_map, depth_map, weights, uncertainty, alphainv_last, valid_rgbs, ray_id_target, weight_direct_use = renderer(rays_train, times_train, model, cams_train, chunk=args.batch_size,
                                    N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True, is_hyper = args.is_hyper)
            else:
                rays_train, rgb_train, times_train= allrays[ray_idx].to(device), allrgbs[ray_idx].to(device), times_flaten[ray_idx].to(device)
                # torch.Size([4096, 6]),torch.Size([4096, 3]),torch.Size([4096, 1])
                #rgb_map, alphas_map, depth_map, weights, uncertainty，在本文件第23行renderer = OctreeRender_trilinear_fast
                rgb_map, alphas_map, depth_map, weights, uncertainty, alphainv_last, valid_rgbs, ray_id_target, weight_direct_use = renderer(rays_train, times_train, model, chunk=args.batch_size,
                                    N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)
        
        if args.is_resampling:
            per_ray_loss = torch.mean((rgb_map - rgb_train)**2, dim=-1)
            region_ids_batch = train_dataset.all_region_ids[ray_idx].to(device)
            tracker.update(region_ids_batch, per_ray_loss)
            # —— 阶段性更新权重 —— 
            if iteration in {50000, 70000}:
                new_probs = tracker.get_probs(tau=0.2)
                trainingSampler_weighted.update_probs(new_probs)
            loss = per_ray_loss.mean()
        else:
            loss = torch.mean((rgb_map - rgb_train) ** 2) # mse,在一个batch下，每个采样点的mse做好后取平均值

        # # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = model.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0 and args.fused == None:
            loss_reg_L1 = model.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            # if is TT
            if args.model_name != "TensorTT":
                loss_tv = model.TV_loss_density(tvreg) * TV_weight_density
            else:
                loss_tv = model.TV_loss(density_only = True) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            if args.model_name != "TensorTT":
                loss_tv = model.TV_loss_app(tvreg)*TV_weight_app
            else:
                loss_tv = model.TV_loss(density_only = False) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        # 引入TiNeuVox新loss
        # loss_1
        pout = alphainv_last.clamp(1e-6, 1-1e-6) # torch.Size([4096])
        entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
        total_loss += 0.001 * entropy_last_loss
        summary_writer.add_scalar('train/entropy_last_loss', 0.001 * entropy_last_loss.detach().item(), global_step=iteration)

        # loss_2
        rgbper = (valid_rgbs - rgb_train[ray_id_target]).pow(2).sum(-1) # torch.Size([446200])
        rgbper_loss = (rgbper * weight_direct_use.detach()).sum() / 4096
        total_loss += 0.01 * rgbper_loss
        # print(total_loss)
        summary_writer.add_scalar('train/rgbper_loss', 0.01 * rgbper_loss.detach().item(), global_step=iteration)
            

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)
        summary_writer.add_scalar('train/LR', optimizer.param_groups[0]['lr'], global_step=iteration)

        if iteration <= warmup_steps:
            summary_writer.add_scalar("Warmup", lr_warmup_scheduler.get_last_lr()[0], global_step=iteration)
            lr_warmup_scheduler.step()


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0 or iteration-1 in upsample_list and args.plot_after_upsampling:
            PSNRs_test = evaluation(test_dataset,model, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
        if iteration-1 in upsample_list and args.plot_after_upsampling:
            PSNRs_test = evaluation(test_dataset,model, args, renderer, f'{logfolder}/imgs_vis/', N_vis=2,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)

        if iteration in update_AlphaMask_list:
            new_aabb = model.updateAlphaMask(reso_cur)
            if iteration == update_AlphaMask_list[0]:
                if args.should_shrink:
                    model.shrink(new_aabb)
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)
            elif not args.ndc_ray: #  and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                # allrays,allrgbs = model.filtering_rays(allrays,allrgbs, wandb=wandb if args.i_wandb else None)
                allrays, allrgbs, times_flaten = model.filtering_rays(allrays, allrgbs, alltimes, bbox_only=True, wandb=wandb if args.i_wandb else None)
                trainingSampler_random = SimpleSampler(allrgbs.shape[0], args.batch_size)
                
                
        if iteration in upsample_ranks_iterations:

            rank_app_before, rank_density_before = model.get_max_ranks()
            
            rank_app = upsample_ranks_app[ upsample_ranks_iterations.index(iteration) ]
            rank_density = upsample_ranks_density[ upsample_ranks_iterations.index(iteration) ]                        
            model.upsample_volume_ranks(rank_app, rank_density)
            
            print("upsampling ranks from", rank_app_before, rank_density_before, "to", rank_app, rank_density)
            if args.model_name in ["TensorTT" , "TensorVMSplit", "TensorCP", "TensorVM"]:
                compression_info = model.get_compression_values()
                print("compression info", compression_info)
                tb_add_scalars(summary_writer, 'stats',compression_info, global_step=0)
            
            
            # index of iteration in upsample_ranks_iterations
            curr_lr = optimizer.param_groups[0]['lr'] * args.rank_upsampling_lr_factor
            curr_lr_shader = optimizer.param_groups[2]['lr'] * args.rank_upsampling_lr_factor
            
            curr_lr = min(curr_lr, args.lr_init)
            curr_lr_shader = min(curr_lr_shader, args.lr_init_shader) 
            
            grad_vars = model.get_optparam_groups(curr_lr, curr_lr_shader)

            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
            

            if args.model_name in ["TensorTT" , "TensorVMSplit", "TensorCP", "TensorVM"]:
                compression_info = model.get_compression_values()
                print("compression info", compression_info)
                tb_add_scalars(summary_writer, 'stats',compression_info, global_step=0)

            if args.warmup_steps > 0:
                warmup_steps = args.warmup_steps
                lr_warmup_scheduler = linear_warmup_lr_scheduler(optimizer, warmup_steps)
                warmup_steps = args.warmup_steps + iteration
                
            model.train()

        if iteration in upsample_list:
            print("=========> upsampling ...")
            model.eval()
            if args.model_name == "TensorVMSplit" and args.should_shrink:
                n_voxels = N_voxel_list.pop(0)
                reso_cur = N_to_reso(n_voxels, model.aabb, model_name = args.model_name)
            elif args.model_name != "TensorTT":
                n_voxels = N_voxel_list.pop(0)
                # take the third root of of the number of voxels
                reso = int(n_voxels ** (1/3))
                reso_cur = [reso, reso, reso]
            else:
                reso_cur = [x * 2 for x in reso_cur]   

            model.upsample_volume_grid(reso_cur)
            nSamples = model.nSamples # updated in upsample_volume_grid
            nSamples = min(nSamples, args.nSamples)
            model.nSamples = nSamples
            
            print("nSamples", nSamples)
            summary_writer.add_scalar('train/nSamples', nSamples, global_step=0)
            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = model.get_optparam_groups(args.lr_init*lr_scale, args.lr_init_shader*lr_scale, args.lr_init_featurenet*lr_scale, args.lr_init_deformation_net*lr_scale, args.lr_init_density_net*lr_scale, args.lr_init_time_net*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

            compression_info = model.get_compression_values()
            print("compression info", compression_info)
            tb_add_scalars(summary_writer, 'stats',compression_info, global_step=0)

            if args.warmup_steps > 0:
                warmup_steps = args.warmup_steps
                lr_warmup_scheduler = linear_warmup_lr_scheduler(optimizer, warmup_steps)
                warmup_steps = args.warmup_steps + iteration
            
            # take model back to gpu
            model.train()
            print('===============================')


        # if lr is less than lr_min, set lr_factor to 1
        if optimizer.param_groups[0]['lr'] < lr_min:
            lr_factor = 0.999999 # make it very small
            print("lr_factor set to 0.999999")

    torch.cuda.empty_cache()
    
    summary_writer.add_scalar('train/nSamples', model.nSamples, global_step=0)
        
    save_path = f'{logfolder}/{args.expname}.th'
    model.save(save_path)


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,model, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,model, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,summary_writer=summary_writer)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,model, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        
    print("model", model)
    

def log_gradients(model):
    grad_log = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
                    # Wandb does not accept NoneType, so we check if the gradient is not None
            grad_log[f"grad_{name}"] = wandb.Histogram(param.grad.cpu().numpy()) if param.grad is not None else 0

    wandb.log(grad_log)


def update_kwargs(kwargs, args, near_far=None, n_lamb_sigma=None, n_lamb_sh=None, reso_cur=None):
    if near_far is None:
        near_far = kwargs['near_far']
    if n_lamb_sigma is None:
        n_lamb_sigma = kwargs['density_n_comp']
    if n_lamb_sh is None:
        n_lamb_sh = kwargs['appearance_n_comp']
    if reso_cur is None:
        reso_cur = args.N_voxel_final

    kwargs.update({
    'reso_cur': reso_cur,
    'device': device,
    'density_n_comp': n_lamb_sigma,
    'appearance_n_comp': n_lamb_sh,
    'app_dim': args.data_dim_color,
    'near_far': near_far,
    'shadingMode': args.shadingMode,
    'alphaMask_thres': args.alpha_mask_thre,
    'density_shift': args.density_shift,
    'distance_scale': args.distance_scale,
    'pos_pe': args.pos_pe,
    'view_pe': args.view_pe,
    'fea_pe': args.fea_pe,
    'featureC': args.featureC,
    'step_ratio': args.step_ratio,
    'fea2denseAct': args.fea2denseAct,
    'use_TTNF_sampling': args.use_TTNF_sampling,
    'rayMarch_weight_thres': args.rm_weight_mask_thre,
    'fused': args.fused,
    'max_rank': args.max_rank,
    'canonization': args.canonization,
    'compression_alg': args.compression_alg,
    'name': args.model_name
        })
    return kwargs

if __name__ == '__main__':


    args = config_parser()
    torch.set_default_dtype(torch.float32)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:       
        reconstruction(args)