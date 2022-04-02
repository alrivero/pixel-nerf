# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import imp
import sys
import os
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train"))
)

import warnings
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util
import numpy as np
import torch.nn.functional as F
import torch
import tqdm
from dotmap import DotMap
from random import randint
from torchvision.transforms.functional_tensor import crop
from data.AppearanceDataset import AppearanceDataset


def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--appdir", "-DA", type=str, default=None, help="Appearance Dataset directory"
    )
    parser.add_argument(
        "--appearance_format",
        "-FA",
        type=str,
        default=None,
        help="Appearance format, eth3d (only for now)",
    )
    parser.add_argument(
        "--app_ind", "-I", type=int, default=0, help="Index of image to be used for appearance harmonization"
    )
    parser.add_argument(
        "--load_app_encoder",
        action="store_true",
        default=None,
        help="Load an appearance encoder's weights",
    )
    parser.add_argument(
        "--freeze_app_enc",
        action="store_true",
        default=None,
        help="Freeze appearance encoder weights and only train MLP",
    )
    parser.add_argument(
        "--freeze_f1",
        action="store_true",
        default=None,
        help="Freeze first multi-view network weights and only train later MLP",
    )
    parser.add_argument(
        "--refencdir", "-DRE", type=str, default=None, help="Reference encoder directory (used for loss)"
    )
    parser.add_argument(
        "--ray_type",
        "-RT",
        type=str,
        default=None,
        help="Which kind of ray smapling to do, either rand or patch",
    )
    parser.add_argument(
        "--app_scale", "-AS", type=float, default=1.0, help="The scale of app scenes (FLESH OUT IDK)"
    )
    return parser


args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
device = util.get_cuda(args.gpu_id[0])

app_size = None
app_size_h = conf.get_int("data.app_data.img_size_h", None)
app_size_w = conf.get_int("data.app_data.img_size_w", None)
if (app_size_h is not None and app_size_w is not None):
    app_size = (app_size_h, app_size_w)

dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)
dset_app = AppearanceDataset(args.appdir, "train", image_size=app_size, img_ind=args.app_ind)
print(
    "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp)
)

net = make_model(
    conf["model"],
    stop_encoder_grad=args.freeze_enc,
    stop_app_encoder_grad=args.freeze_app_enc,
    stop_f1_grad=args.freeze_f1,
).to(device=device)


if args.freeze_enc:
    print("Encoder frozen")
    net.encoder.eval()

if args.freeze_app_enc:
    print("Appearance encoder weights frozen")
    net.app_encoder.eval()

renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp,).to(
    device=device
)

# Parallize
render_par = renderer.bind_parallel(net, args.gpu_id).eval()
nviews = list(map(int, args.nviews.split()))


class PixelNeRF_ATrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
            self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        # Loss configuration for appearance specific losses
        self.lambda_density = conf.get_float("loss.lambda_density")
        self.lambda_ref = conf.get_float("loss.lambda_ref")
        print(
            "lambda density {} and reference {}".format(self.lambda_density, self.lambda_ref)
        )
        density_loss_conf = conf["loss.density"]
        self.density_app_crit = loss.get_density_loss(density_loss_conf)
        self.ref_app_crit = loss.ReferenceColorLoss(conf, args.refencdir).to(device=device)

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(
                    torch.load(self.renderer_state_path, map_location=device)
                )

        self.z_near = dset.z_near
        self.z_far = dset.z_far

        self.use_bbox = args.no_bbox_step > 0

        self.ray_type = args.ray_type
        self.pass_setup = self.patch_pass_setup if self.ray_type == "patch" else self.rand_pass_setup

        self.appearance_img = dset_app[args.app_ind]["images"].unsqueeze(0).to(device=device)
        self.ref_app_crit.encode_targets(self.appearance_img)

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def rand_pass_setup(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = data["focal"]  # (SB)
        all_c = data.get("c")  # (SB)

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)

        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []

        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            c = None
            if "c" in data:
                c = data["c"][obj_idx]
            if curr_nviews > 1:
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )
            images_0to1 = images * 0.5 + 0.5

            cam_rays = util.gen_rays(
                poses, W, H, focal, self.z_near, self.z_far, c=c
            )  # (NV, H, W, 8)
            rgb_gt_all = images_0to1
            rgb_gt_all = (
                rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (NV, H, W, 3)

            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, args.ray_batch_size)
                pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
            else:
                pix_inds = torch.randint(0, NV * H * W, (args.ray_batch_size,))

            rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3)
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
                device=device
            )  # (ray_batch_size, 8)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(device)
        src_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        all_bboxes = all_poses = all_images = None

        net.encode(
            src_images,
            src_poses,
            all_focals.to(device=device),
            c=all_c.to(device=device) if all_c is not None else None,
        )

        return src_images, all_rays, all_rgb_gt

    def patch_pass_setup(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
        all_focals = data["focal"]  # (SB)
        all_c = data.get("c")  # (SB)

        all_rgb_gt = []
        all_rays = []

        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        for obj_idx in range(SB):
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            c = None
            if "c" in data:
                c = data["c"][obj_idx]
            if curr_nviews > 1:
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )
            images_0to1 = images * 0.5 + 0.5

            cam_rays = util.gen_rays(
                poses, W, H, focal, self.z_near, self.z_far, c=c
            ).permute(0, 3, 1, 2)
            rgb_gt_all = images_0to1
            rgb_gt_all = (
                rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            ).reshape(NV, 3, H, W)

            Hs = int(H * args.app_scale)
            Ws = int(W * args.app_scale)
            i = randint(0, H - Hs)
            j = randint(0, W - Ws)

            rgb_gt = crop(rgb_gt_all, i, j, Hs, Ws)
            rays = crop(cam_rays, i, j, Hs, Ws)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(device)
        src_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)
        all_rays = util.batched_index_select_nd(all_rays, image_ord).reshape(SB, -1, 8)
        all_rgb_gt = util.batched_index_select_nd(all_rgb_gt, image_ord).reshape(SB, -1, 3)

        all_poses = all_images = None

        net.encode(
            src_images,
            src_poses,
            all_focals.to(device=device),
            c=all_c.to(device=device) if all_c is not None else None,
        )

        return src_images, all_rays, all_rgb_gt

    def reg_pass(self, all_rays):
        return DotMap(render_par(all_rays, want_weights=True, app_pass=False))

    def app_pass(self, app_imgs, all_rays):
        # Appearance encoder encoding
        net.app_encoder.encode(app_imgs)
        render_dict = DotMap(render_par(all_rays, want_weights=True, app_pass=True))

        return render_dict

    def nerf_loss(self, app_render_dict, all_rgb_gt, loss_dict):
        # Compute our standard PixelNeRF loss
        coarse_app = app_render_dict.coarse
        fine_app = app_render_dict.fine
        using_fine_app = len(fine_app) > 0

        rgb_loss = self.rgb_coarse_crit(coarse_app.rgb, all_rgb_gt) * self.lambda_coarse
        loss_dict["ac"] = rgb_loss.item()
        if using_fine_app:
            fine_loss = self.rgb_fine_crit(fine_app.rgb, all_rgb_gt)
            rgb_loss = rgb_loss + fine_loss * self.lambda_fine
            loss_dict["af"] = fine_loss.item() * self.lambda_fine
        
        return rgb_loss

    def app_loss(self, src_images, app_render_dict, reg_render_dict, loss_dict):

        # Compute SSH reference encoder loss for appearance pass and density (depth) regularization
        coarse_reg = reg_render_dict.coarse
        fine_reg = reg_render_dict.fine

        coarse_app = app_render_dict.coarse
        fine_app = app_render_dict.fine
        using_fine_app = len(fine_app) > 0

        if using_fine_app:
            density_app_loss = self.density_app_crit(fine_reg.depth.detach(), fine_app.depth)
        else:
            density_app_loss = self.density_app_crit(coarse_reg.depth.detach(), coarse_app.depth)
        density_app_loss *= self.lambda_density
        loss_dict["ad"] = density_app_loss.item()

        # We need to reshape our color data into image patches to feed reference encoder
        B, _, D, H, W = src_images.shape
        Hs = int(H * args.app_scale)
        Ws = int(W * args.app_scale)
        if using_fine_app:
            app_rgb = fine_app.rgb.reshape(-1, D, Hs, Ws)
            app_rgb = F.interpolate(app_rgb, size=(H, W), mode="area")
            ref_app_loss = self.ref_app_crit(app_rgb) * self.lambda_ref
        else:
            app_rgb = coarse_app.rgb.reshape(-1, D, Hs, Ws)
            app_rgb = F.interpolate(app_rgb, size=(H, W), mode="area")
            ref_app_loss = self.ref_app_crit(app_rgb) * self.lambda_ref
        loss_dict["ar"] = ref_app_loss.item()

        return density_app_loss + ref_app_loss

    def calc_losses(self, data, app_data, is_train=True, global_step=0):
        # Do some setup to establish rays and view images
        src_images, all_rays, all_rgb_gt = self.pass_setup(data, is_train, global_step)

        # Render out the scene normally using pretrained F2
        reg_render_dict = self.reg_pass(all_rays)

        # Render out our scene using appearance encoding and trainable F2
        app_render_dict = self.app_pass(app_data, all_rays)

        loss_dict = {}
        
        # Compute our standard NeRF losses and losses associated with appearance encoder
        nerf_loss = self.nerf_loss(app_render_dict, all_rgb_gt, loss_dict)
        app_loss = self.app_loss(src_images, app_render_dict, reg_render_dict, loss_dict)

        # Compute our standard NeRF loss
        loss = nerf_loss + app_loss

        if is_train:
            loss.backward()
        loss_dict["t"] = loss.item()

        return loss_dict

    def train_step(self, data, app_data, global_step):
        return self.calc_losses(data, app_data, is_train=True, global_step=global_step) 

    def eval_step(self, data, app_data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, app_data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        app_images = self.appearance_img
        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx : batch_idx + 1]  # (1)
        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=c
        )  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        view_dest = np.random.randint(0, NV - curr_nviews)
        for vs in range(curr_nviews):
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        # set renderer net to eval mode
        renderer.eval()
        source_views = (
            images_0to1[views_src]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .reshape(-1, H, W, 3)
        )

        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H, W, 8)
            test_images = images[views_src]  # (NS, 3, H, W)
            net.encode(
                test_images.unsqueeze(0),
                poses[views_src].unsqueeze(0),
                focal.to(device=device),
                c=c.to(device=device) if c is not None else None,
            )
            net.app_encoder.encode(app_images)
            test_rays = test_rays.reshape(1, H * W, -1)
            render_dict = DotMap(render_par(test_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0

            alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
            rgb_coarse_np = coarse.rgb[0].cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)

            if using_fine:
                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_list = [
            *source_views,
            gt,
            depth_coarse_cmap,
            rgb_coarse_np,
            alpha_coarse_cmap,
        ]

        app_images_0to1 = app_images * 0.5 + 0.5  # (NV, 3, H, W)
        Wa = app_images.shape[-1]
        app_gt = app_images_0to1[batch_idx].permute(1, 2, 0).cpu().numpy().reshape(H, Wa, 3)

        vis_list.append(app_gt)

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            vis_list = [
                *source_views,
                gt,
                depth_fine_cmap,
                rgb_fine_np,
                alpha_fine_cmap,
            ]

            app_images_0to1 = app_images * 0.5 + 0.5  # (NV, 3, H, W)
            Wa = app_images.shape[-1]
            app_gt = app_images_0to1[batch_idx].permute(1, 2, 0).cpu().numpy().reshape(H, Wa, 3)
            vis_list.append(app_gt)

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        renderer.train()
        return vis, vals

    def start(self):
        def fmt_loss_str(losses):
            return "loss " + (" ".join(k + ":" + str(losses[k]) for k in losses))

        def data_loop(dl):
            """
            Loop an iterable infinitely
            """
            while True:
                for x in iter(dl):
                    yield x

        test_data_iter = data_loop(self.test_data_loader)
        step_id = self.start_iter_id

        progress = tqdm.tqdm(bar_format="[{rate_fmt}] ")
        for epoch in range(self.num_epochs):
            self.writer.add_scalar(
                "lr", self.optim.param_groups[0]["lr"], global_step=step_id
            )

            batch = 0
            for _ in range(self.num_epoch_repeats):
                for data in self.train_data_loader:
                    losses = self.train_step(data, self.appearance_img, global_step=step_id)
                    loss_str = fmt_loss_str(losses)
                    if batch % self.print_interval == 0:
                        print(
                            "E",
                            epoch,
                            "B",
                            batch,
                            loss_str,
                            " lr",
                            self.optim.param_groups[0]["lr"],
                        )

                    if batch % self.eval_interval == 0:
                        test_data = next(test_data_iter)
                        self.net.eval()
                        with torch.no_grad():
                            test_losses = self.eval_step(test_data, self.appearance_img, global_step=step_id)
                        self.net.train()
                        test_loss_str = fmt_loss_str(test_losses)
                        self.writer.add_scalars("train", losses, global_step=step_id)
                        self.writer.add_scalars(
                            "test", test_losses, global_step=step_id
                        )
                        print("*** Eval:", "E", epoch, "B", batch, test_loss_str, " lr")

                    if batch % self.save_interval == 0 and (epoch > 0 or batch > 0):
                        print("saving")
                        if self.managed_weight_saving:
                            self.net.save_weights(self.args)
                        else:
                            torch.save(
                                self.net.state_dict(), self.default_net_state_path
                            )
                        torch.save(self.optim.state_dict(), self.optim_state_path)
                        if self.lr_scheduler is not None:
                            torch.save(
                                self.lr_scheduler.state_dict(), self.lrsched_state_path
                            )
                        torch.save({"iter": step_id + 1}, self.iter_state_path)
                        self.extra_save_state()

                    if batch % self.vis_interval == 0:
                        print("generating visualization")
                        if self.fixed_test:
                            test_data = next(iter(self.test_data_loader))
                        else:
                            test_data = next(test_data_iter)
                        
                        # Render out the scene
                        self.net.eval()
                        with torch.no_grad():
                            vis, vis_vals = self.vis_step(
                                test_data, self.appearance_img, global_step=step_id
                            )
                        if vis_vals is not None:
                            self.writer.add_scalars(
                                "vis", vis_vals, global_step=step_id
                            )
                        self.net.train()
                        if vis is not None:
                            import imageio

                            vis_u8 = (vis * 255).astype(np.uint8)
                            imageio.imwrite(
                                os.path.join(
                                    self.visual_path,
                                    "{:04}_{:04}_vis.png".format(epoch, batch),
                                ),
                                vis_u8,
                        )
                    if (
                        batch == self.num_total_batches - 1
                        or batch % self.accu_grad == self.accu_grad - 1
                    ):
                        # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                        self.optim.step()
                        self.optim.zero_grad()

                    self.post_batch(epoch, batch)
                    step_id += 1
                    batch += 1
                    progress.update(1)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
trainer = PixelNeRF_ATrainer()
trainer.start()
