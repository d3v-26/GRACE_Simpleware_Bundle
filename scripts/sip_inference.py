import os
import torch
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR
from monai.transforms import Compose, Spacingd, Orientationd, ScaleIntensityRanged, Resize
from monai.data import MetaTensor
from scipy.io import savemat


def run_inference(image_np):
    model = UNETR(
        in_channels=1,
        out_channels=12,
        img_size=(64,64,64),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
        proj_type="perceptron",
    )

    raise NotImplementedError("TBD")