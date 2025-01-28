import torch
import nibabel as nib
from monai.data import MetaTensor
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Spacingd, Orientationd, ScaleIntensityRanged


def run_inference(input_path):
    model_path = "../model.pth"
    spatial_size = (64, 64, 64)
    a_min_value = 0
    a_max_value = 255
    num_gpu = 1
    dataparallel = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = UNETR(
        in_channels=1,
        out_channels=12,
        img_size=spatial_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
        proj_type="perceptron",
    )
    if dataparallel:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

    model = model.to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    input_img = nib.load(input_path)
    image_data = input_img.get_fdata()

    # Convert to MetaTensor for MONAI compatibility
    meta_tensor = MetaTensor(image_data, affine=input_img.affine)

    # Apply MONAI test transforms
    test_transforms = Compose(
        [
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("trilinear"),
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min_value, a_max=a_max_value, b_min=0.0, b_max=1.0, clip=True),
        ]
    )
    
    # Wrap the MetaTensor for the transform pipeline
    data = {"image": meta_tensor}
    transformed_data = test_transforms(data)

    # Convert to PyTorch tensor
    image_tensor = transformed_data["image"].clone().detach().unsqueeze(0).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        predictions = sliding_window_inference(
            image_tensor, spatial_size, sw_batch_size=4, predictor=model, overlap=0.8
        )

    processed_preds = torch.argmax(predictions, dim=1).detach().cpu().numpy().squeeze()
    pred_img = nib.Nifti1Image(processed_preds, affine=input_img.affine, header=input_img.header)
    # nii_save_path = os.path.join(output_dir, f"{base_filename}_pred.nii.gz")

    return pred_img