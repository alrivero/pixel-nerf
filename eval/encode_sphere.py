import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import torch
import util
import tqdm
from data.AppearanceDataset import AppearanceDataset
from contrib.model.unet_tile_se_norm import StyleEncoder
from contrib.model.PatchEncoder import PatchEncoder

def extra_args(parser):
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
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
    parser.add_argument(
        "--sphere_subdiv",
        "-S",
        type=int,
        default=200,
        help="Level of subdivision used for sphere points",
    )
    parser.add_argument(
        "--batch_size",
        "-B",
        type=int,
        default=100,
        help="# of patches to encode per iteration",
    )
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

device = util.get_cuda(args.gpu_id[0])

# Reference encoder used across network
ref_encoder = StyleEncoder(4, 3, 32, 512, norm="BN", activ="relu", pad_type='reflect').to(device=device)
ref_encoder.load_state_dict(torch.load(args.refencdir))
patch_encoder = PatchEncoder(ref_encoder)

dset_app = AppearanceDataset(args.appdir, "train", image_size=(2048, 4096)) # SET IMAGE SIZE
app_imgs = dset_app[args.app_set_ind][args.app_ind].unsqueeze(0).to(device=device)

sphere_verts = util.uv_sphere(args.radius, args.sphere_subdiv).to(device=device)
sphere_verts = sphere_verts.reshape(-1, 3)

radius = torch.tensor(args.radius).unsqueeze(-1).to(device=device)

print("Encoding Patches...")
with torch.no_grad():
    all_encs = []
    for rays in tqdm.tqdm(
        torch.split(sphere_verts, 1, dim=0)
    ):
        uv_env = util.rays_blinn_newell_uv(rays[None], radius, app_imgs, 223)
        enc_patches = util.uv_to_rgb_patches(app_imgs, uv_env, 223)
        batch_encs = patch_encoder(enc_patches)
        all_encs.append(batch_encs.to(device="cpu"))
    all_encs = torch.cat(all_encs)

    torch.save(all_encs, args.save_dir)