import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, times, tensorf, cameras=None, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda', is_hyper=False):

    rgbs, alphas, depth_maps, weights, uncertainties= [], [], [], [], []
    N_rays_all = rays.shape[0] #4096
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        if is_hyper:
            rays_chunk, times_chunk, cameras_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk], times[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device), cameras[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device) #更新索引块对应的rays
            # 下面直接进tensorbase的forward
            rgb_map, depth_map, acc_map, valid_rgbs, ray_id_target, weight_direct_use = tensorf(rays_chunk, times_chunk, cameras_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
            
        else:
            rays_chunk, times_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk], times[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device) #更新索引块对应的rays
            # 下面直接进tensorbase的forward
            rgb_map, depth_map, acc_map, valid_rgbs, ray_id_target, weight_direct_use = tensorf(rays_chunk, times_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)


    
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None, acc_map, valid_rgbs, ray_id_target, weight_direct_use

# @torch.no_grad()
# def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
#                white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda', summary_writer=None):
#     PSNRs, rgb_maps, depth_maps = [], [], []
#     ssims,l_alex,l_vgg=[],[],[]
#     os.makedirs(savePath, exist_ok=True)
#     os.makedirs(savePath+"/rgbd", exist_ok=True)

#     try:
#         tqdm._instances.clear()
#     except Exception:
#         pass

#     near_far = test_dataset.near_far
#     img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
#     idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))


#     for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

#         W, H = test_dataset.img_wh
#         rays = samples.view(-1,samples.shape[-1])

#         rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
#                                         ndc_ray=ndc_ray, white_bg = white_bg, device=device)

#         rgb_map = rgb_map.clamp(0.0, 1.0)

#         rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

#         depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
#         if len(test_dataset.all_rgbs):
#             gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
#             loss = torch.mean((rgb_map - gt_rgb) ** 2)
#             PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

#             if compute_extra_metrics:
#                 ssim = rgb_ssim(rgb_map, gt_rgb, 1)
#                 # l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
#                 l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
#                 ssims.append(ssim)
#                 # l_alex.append(l_a)
#                 l_vgg.append(l_v)

#         rgb_map = (rgb_map.numpy() * 255).astype('uint8')
#         # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
#         rgb_maps.append(rgb_map)
#         depth_maps.append(depth_map)
#         if savePath is not None:
#             imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
#             rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
#             imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

#     imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
#     imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

#     if PSNRs:
#         psnr = np.mean(np.asarray(PSNRs))
#         if compute_extra_metrics:
#             ssim = np.mean(np.asarray(ssims))
#             # l_a = np.mean(np.asarray(l_alex))
#             l_v = np.mean(np.asarray(l_vgg))
#             np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, 99999999, l_v]))
#             if summary_writer is not None:
#                 summary_writer.add_scalar('test/psnr', psnr, 100000)
#                 summary_writer.add_scalar('test/ssim', ssim, 100000)
#                 summary_writer.add_scalar('test/l_vgg', l_v, 100000)
#         else:
#             np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


#     return PSNRs

@torch.no_grad()
def evaluation(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda', summary_writer=None, all=False, test=True):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass
    
    if args.is_hyper:
        near_far = test_dataset.near_far
        if test: # test和all这两个指示变量还未添加
            if all:
                idx = test_dataset.data_class.i_test
            else:
                idx = test_dataset.data_class.i_test[::16] # 以步长为16采样
        else:
            if all:
                idx = test_dataset.data_class.i_train
            else:
                idx = test_dataset.data_class.i_train[::16]
                
        for i in tqdm(idx):
            W = test_dataset.w
            H = test_dataset.h
            
            rays_o, rays_d, viewdirs,rgb_gt = test_dataset.data_class.load_idx(i, not_dic=True)
            
            rays = torch.cat([rays_o, rays_d], dim=1).to(device)
            times = test_dataset.data_class.all_time[i]*torch.ones_like(rays_o[:,0:1])
            cam_one = test_dataset.data_class.all_cam[i]*torch.ones_like(rays_o[:,0:1])
            
            # 调用渲染器，并传入 times
            rgb_map, _, depth_map, _, _, _, _, _, _,= renderer(
                rays, times, tensorf, cam_one, chunk=4096, N_samples=N_samples,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device, is_hyper = args.is_hyper
            )

            rgb_map = rgb_map.clamp(0.0, 1.0)

            rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

            depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
            
            gt_rgb = rgb_gt.view(H, W, 3)  ### 注意检查一下索引
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                # l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                # l_alex.append(l_a)
                l_vgg.append(l_v)

            rgb_map = (rgb_map.numpy() * 255).astype('uint8')
            # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            rgb_maps.append(rgb_map)
            depth_maps.append(depth_map)
            if savePath is not None:
                imageio.imwrite(f'{savePath}/{prtx}{i:03d}.png', rgb_map)
                # imageio.imwrite(f'{savePath}/{prtx}{"_".join(map(str, i))}.png', rgb_map)
                rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
                imageio.imwrite(f'{savePath}/rgbd/{prtx}{i:03d}.png', rgb_map)
                # imageio.imwrite(f'{savePath}/rgbd/{prtx}{"_".join(map(str, i))}.png', rgb_map)

        imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
        imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

        if PSNRs:
            psnr = np.mean(np.asarray(PSNRs))
            if compute_extra_metrics:
                ssim = np.mean(np.asarray(ssims))
                # l_a = np.mean(np.asarray(l_alex))
                l_v = np.mean(np.asarray(l_vgg))
                np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, 99999999, l_v]))
                if summary_writer is not None:
                    summary_writer.add_scalar('test/psnr', psnr, 100000)
                    summary_writer.add_scalar('test/ssim', ssim, 100000)
                    summary_writer.add_scalar('test/l_vgg', l_v, 100000)
            else:
                np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

        return PSNRs

    elif args.ndc_ray:
        near_far = test_dataset.near_far
        img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
        idxs = list(range(0, len(test_dataset), img_eval_interval))

        for idx in tqdm(idxs):
            data = test_dataset[idx]
            samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]
            depth = None

            W, H = test_dataset.img_wh
            rays = samples.view(-1, samples.shape[-1]).to(device)
            times = sample_times.view(-1, sample_times.shape[-1]).to(device)

            # 调用渲染器，并传入 times
            rgb_map, _, depth_map, _, _, _, _, _, _,= renderer(
                rays, times, tensorf, chunk=4096, N_samples=N_samples,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device
            )

            rgb_map = rgb_map.clamp(0.0, 1.0)

            rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

            depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
            if len(test_dataset.all_rgbs):
                gt_rgb = gt_rgb.view(H, W, 3)
                loss = torch.mean((rgb_map - gt_rgb) ** 2)
                PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

                if compute_extra_metrics:
                    ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                    # l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                    l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                    ssims.append(ssim)
                    # l_alex.append(l_a)
                    l_vgg.append(l_v)

            rgb_map = (rgb_map.numpy() * 255).astype('uint8')
            # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            rgb_maps.append(rgb_map)
            depth_maps.append(depth_map)
            if savePath is not None:
                imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
                rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
                imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

        imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
        imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

        if PSNRs:
            psnr = np.mean(np.asarray(PSNRs))
            if compute_extra_metrics:
                ssim = np.mean(np.asarray(ssims))
                # l_a = np.mean(np.asarray(l_alex))
                l_v = np.mean(np.asarray(l_vgg))
                np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, 99999999, l_v]))
                if summary_writer is not None:
                    summary_writer.add_scalar('test/psnr', psnr, 100000)
                    summary_writer.add_scalar('test/ssim', ssim, 100000)
                    summary_writer.add_scalar('test/l_vgg', l_v, 100000)
            else:
                np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

        return PSNRs

    else:
        # 下面原本putt的合并到else中
        near_far = test_dataset.near_far
        img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
        idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))

        for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
            W, H = test_dataset.img_wh
            rays = samples.view(-1, samples.shape[-1])

            # 移动 rays 到设备上
            rays = rays.to(device)

            # 获取当前图像的时间
            time = test_dataset.all_times[idxs[idx]]
            # 创建与射线对应的时间张量
            times = torch.ones((rays.shape[0], 1), device=device) * time

            # 调用渲染器，并传入 times
            rgb_map, _, depth_map, _, _, _, _, _, _,= renderer(
                rays, times, tensorf, chunk=4096, N_samples=N_samples,
                ndc_ray=ndc_ray, white_bg=white_bg, device=device
            )

            rgb_map = rgb_map.clamp(0.0, 1.0)

            rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

            depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
            if len(test_dataset.all_rgbs):
                gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
                loss = torch.mean((rgb_map - gt_rgb) ** 2)
                PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

                if compute_extra_metrics:
                    ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                    # l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                    l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                    ssims.append(ssim)
                    # l_alex.append(l_a)
                    l_vgg.append(l_v)

            rgb_map = (rgb_map.numpy() * 255).astype('uint8')
            # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            rgb_maps.append(rgb_map)
            depth_maps.append(depth_map)
            if savePath is not None:
                imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
                rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
                imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

        imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
        imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

        if PSNRs:
            psnr = np.mean(np.asarray(PSNRs))
            if compute_extra_metrics:
                ssim = np.mean(np.asarray(ssims))
                # l_a = np.mean(np.asarray(l_alex))
                l_v = np.mean(np.asarray(l_vgg))
                np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, 99999999, l_v]))
                if summary_writer is not None:
                    summary_writer.add_scalar('test/psnr', psnr, 100000)
                    summary_writer.add_scalar('test/ssim', ssim, 100000)
                    summary_writer.add_scalar('test/l_vgg', l_v, 100000)
            else:
                np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

        return PSNRs





@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

