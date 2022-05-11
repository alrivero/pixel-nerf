import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import torch
import numpy as np
import util
from data import get_split_dataset
from scipy.interpolate import CubicSpline
import tqdm
from data.AppearanceDataset import AppearanceDataset
from contrib.model.unet_tile_se_norm import StyleEncoder


def extra_args(parser):
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="64",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=40,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    parser.add_argument(
        "--appearance_format",
        "-FA",
        type=str,
        default=None,
        help="Appearance format, eth3d (only for now)",
    )
    parser.add_argument(
        "--appdir", "-DA", type=str, default=None, help="Appearance Dataset directory"
    )
    parser.add_argument(
        "--app_set_ind", "-IA", type=int, default=0, help="Index of image to be used for appearance harmonization"
    )
    parser.add_argument(
        "--app_ind", "-IM", type=int, default=0, help="Index of image to be used for appearance harmonization"
    )
    parser.add_argument(
        "--refencdir", "-DRE", type=str, default=None, help="Reference encoder directory (used for loss)"
    )
    parser.add_argument(
        "--save_dir", "-SD", type=str, default=None, help="Directory to save encodings"
    )
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

device = util.get_cuda(args.gpu_id[0])

dset, _, _ = get_split_dataset(
    args.dataset_format, args.datadir, training=False
)

data = dset[args.subset]
data_path = data["path"]
print("Data instance loaded:", data_path)

images = data["images"]  # (NV, 3, H, W)

poses = data["poses"]  # (NV, 4, 4)
focal = data["focal"]
if isinstance(focal, float):
    # Dataset implementations are not consistent about
    # returning float or scalar tensor in case of fx=fy
    focal = torch.tensor(focal, dtype=torch.float32)
focal = focal[None]

c = data.get("c")
if c is not None:
    c = c.to(device=device).unsqueeze(0)

NV, _, H, W = images.shape

# Get the distance from camera to origin
z_near = dset.z_near
z_far = dset.z_far

# Reference encoder used across network
ref_encoder = StyleEncoder(4, 3, 32, 512, norm="BN", activ="relu", pad_type='reflect').to(device=device)
ref_encoder.load_state_dict(torch.load(args.refencdir))

print("Generating rays")
dtu_format = hasattr(dset, "sub_format") and dset.sub_format == "dtu"

dset_app = AppearanceDataset(args.appdir, "train", image_size=(300, 600)) # SET IMAGE SIZE
app_imgs = dset_app[args.app_set_ind][args.app_ind].unsqueeze(0).to(device=device)

if dtu_format:
    print("Using DTU camera trajectory")
    # Use hard-coded pose interpolation from IDR for DTU

    t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
    pose_quat = torch.tensor(
        [
            [0.9698, 0.2121, 0.1203, -0.0039],
            [0.7020, 0.1578, 0.4525, 0.5268],
            [0.6766, 0.3176, 0.5179, 0.4161],
            [0.9085, 0.4020, 0.1139, -0.0025],
            [0.9698, 0.2121, 0.1203, -0.0039],
        ]
    )
    n_inter = args.num_views // 5
    args.num_views = n_inter * 5
    t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
    scales = np.array([2.0, 2.0, 2.0, 2.0, 2.0]).astype(np.float32)

    s_new = CubicSpline(t_in, scales, bc_type="periodic")
    s_new = s_new(t_out)

    q_new = CubicSpline(t_in, pose_quat.detach().cpu().numpy(), bc_type="periodic")
    q_new = q_new(t_out)
    q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
    q_new = torch.from_numpy(q_new).float()

    render_poses = []
    for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
        new_q = new_q.unsqueeze(0)
        R = util.quat_to_rot(new_q)
        t = R[:, :, 2] * scale
        new_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        new_pose[:, :3, :3] = R
        new_pose[:, :3, 3] = t
        render_poses.append(new_pose)
    render_poses = torch.cat(render_poses, dim=0)
else:
    print("Using default (360 loop) camera trajectory")
    if args.radius == 0.0:
        radius = (z_near + z_far) * 0.5
        print("> Using default camera radius", radius)
    else:
        radius = args.radius

    # Use 360 pose sequence from NeRF
    render_poses = torch.stack(
        [
            util.pose_spherical(angle, args.elevation, radius)
            for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
        ],
        0,
    )  # (NV, 4, 4)

render_rays = util.gen_rays(
    render_poses,
    W,
    H,
    focal * args.scale,
    z_near,
    z_far,
    c=c * args.scale if c is not None else None,
).to(device=device)
# (NV, H, W, 8)
bounding_radius = torch.tensor(args.radius).to(device=device)
render_rays = render_rays.reshape(1, -1, 8)
render_patches = util.sample_spherical_enc_patches(render_rays, bounding_radius, app_imgs, 223)

with torch.no_grad():
    print("Encoding", args.num_views * H * W, "rays")
    all_encs = []
    for patches in tqdm.tqdm(
        torch.split(render_patches, args.ray_batch_size, dim=0)
    ):
        batch_encs = ref_encoder(patches)
        all_encs.append(batch_encs)
    all_encs = torch.cat(all_encs)

    torch.save(all_encs, args.save_dir)