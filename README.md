# GRACE Brain Image Segmentation - Simpleware Bundle

## Overview

This project provides a Simpleware-compatible bundle for the GRACE brain image segmentation model. The model is implemented using MONAI's UNETR architecture and supports 3D medical image segmentation.

## Project Structure

```
GRACE_SimplewareBundle/
│── configs/
│   └── metadata.json        # Model metadata
│── scripts/
│   └── requirements.txt     # Dependencies
│   └── sip_inference.py     # Inference script
│── README.md                # Setup instructions
│── model.pth                # Trained model file
```

## Installation & Setup

1. Ensure you have a Python virtual environment with the necessary dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Place the `GRACE_SimplewareBundle` folder inside the **ExternalModels** directory of Simpleware.

3. Open **Simpleware** and navigate to **External Models** to select and use the GRACE model.

## Running Inference

To test the model, run the following:

```python
from scripts.sip_inference import run_inference

test_input = "path/to/image.nii.gz"  # Input path
output_mask = run_inference(test_input)
print(output_mask.shape)  # Should match input shape
```

## Model Details

- **Architecture:** MONAI UNETR
- **Input:** 3D MRI scans (single-channel)
- **Output:** Segmentation mask
- **Pre-trained Weights:** `model.pth`

## Metadata Configuration (configs/metadata.json)

Modify `metadata.json` to specify model parameters:

```json
{
  "name": "GRACE Brain Segmentation",
  "description": "A UNETR-based model for brain MRI segmentation.",
  "input_channels": 1,
  "output_channels": 12,
  "spatial_size": [64, 64, 64]
}
```

## License

This project is licensed under the Apache License 2.0.

## Contact

For any issues or improvements, please reach out to the development team.

Dev Patel

Chintan Acharya

