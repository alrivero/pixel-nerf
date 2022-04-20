import os
import torch
import torch.nn.functional as F
import glob
import random
from PIL import Image
from util import get_image_to_tensor_balanced
from torchvision.transforms import Resize


class AppearanceDataset(torch.utils.data.Dataset):

    """
    Dataset consisting of images of a scene taken at different angles 
    Meant to be used with appearance encoder
    """
    def __init__(
        self,
        path,
        stage="train",
        list_prefix="new_",
        image_size=None,
        sub_format="eth3d",
        scale_focal=True,
        max_imgs=100000,
        z_near=1.2,
        z_far=4.0,
    ):
        super().__init__()
        self.base_path = path
        assert os.path.exists(self.base_path)

        # Get all directories in main directory
        cats = [x for x in glob.glob(os.path.join(path, "*")) if os.path.isdir(x)]

        if stage == "train":
            file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        elif stage == "val":
            file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        elif stage == "test":
            file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        all_objs = []
        for file_list in file_lists:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        self.all_objs = all_objs
        self.stage = stage

        self.image_to_tensor = get_image_to_tensor_balanced()
        print(
            "Loading Appearance dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
            "type:",
            sub_format,
        )

        # NOTE: Right now, no intrisic or extrinsic camera information is being used here!
        # Add it later!
        self.max_imgs = max_imgs
        self.lindisp = False
        self.resize = Resize(image_size)
    
    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        _, root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob.glob(os.path.join(root_dir, "images", "dslr_images_undistorted", "*"))
            if (x.endswith(".JPG") or x.endswith(".PNG"))
        ]
        rgb_paths = sorted(rgb_paths)
        
        # Get images from this directory
        all_imgs = []
        for _, rgb_path in enumerate(rgb_paths):
            img = Image.open(rgb_path)
            img_tensor = self.image_to_tensor(img)
            img_tensor  = self.resize(img_tensor)

            all_imgs.append(img_tensor)
        all_imgs = torch.stack(all_imgs)

        return all_imgs
