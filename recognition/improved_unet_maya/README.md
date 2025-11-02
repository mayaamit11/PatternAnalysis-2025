   # Prostate 3D Segmentation using Improved UNet3D

This project performs 3D medical image segmentation on the HipMRI Study on Prostate Cancer dataset using an **Improved UNet3D** architecture.  
The goal is to segment the prostate and other anatomical structures within MRI volumes, achieving **mean Dice ≥ 0.70** on the test set.  
Final performance: **mean Dice = 0.72 (the target was met)**.


    ## Files

    | File | Purpose |
    |------|----------|
    | `modules.py` | Contains the Improved UNet3D architecture (encoder-decoder with residual/dilated blocks). |
    | `dataset.py` | Loads and preprocesses 3D NIfTI volumes from HipMRI. |
    | `train.py` | Training + validation loop with Dice/Loss logging and checkpoint saving. |
    | `predict.py` | Loads best checkpoint, runs inference, reports Dice, saves overlay images. |
    | `utils.py` | Helper functions for metrics, Dice loss, etc. |
    | `runs/` | Contains logs, checkpoints, and `metrics.csv`. |
    | `metrics/` | Locally generated plots and tables for this report. |
    | `images/` | Png of example MRI slice and overlayed segmentation |
    
    ## Training Configuration
    | Setting | Value |
    |----------|-------|
    | Optimizer | Adam |
    | Learning Rate | 1e-3 |
    | Batch Size | 1 |
    | Epochs | 6 |
    | Loss | Dice + Cross Entropy |
    | GPU | NVIDIA A100 (Rangpur) |
    | Framework | PyTorch 2.5.1 |

    ## Dataset

    **Dataset:** HipMRI Study on Prostate Cancer  
    **Location (Rangpur):** `/home/groups/comp3710/HipMRI_Study_open/`

    - **semantic_MRs/** → 3D MRI input volumes  
    - **semantic_labels_only/** → Corresponding segmentation masks  
    - Loaded with Nibabel, normalised to zero-mean/unit-variance  
    - Cropped/padded to consistent shape  
    - Split 70% train, 15% val, 15% test


    ### Quantitative Results
    | Metric               | Training Dice | Validation Dice | Test Dice | Target | Met Target? |
    |----------------------|---------------|------------------|-----------|--------|-------------|
    | Mean Dice Coefficient| —             | 0.714           | **0.72** | ≥ 0.70 | ✅ Yes |
    | Validation Loss      | —             | 0.206           | —         | —      | —           |
    | Epochs Trained       | 8            | —                | —         | —      | —           |


    ### Training Curves
    ![Training vs Validation Loss](metrics/training_loss_curve.png)
    ![Validation Mean Dice](metrics/val_dice_curve.png)
    *Both training and validation loss decrease steadily while the Dice coefficient improves to 0.72, indicating stable training and convergence.*
    
    ### Example Output
    | Input Slice | Ground Truth | Predicted Segmentation |
    |--------------|---------------|------------------------|
    | ![](metrics/example_input.png) | ![](metrics/example_gt.png) | ![](metrics/example_pred.png) |

    ## Data (Rangpur)
    **Dataset:** HipMRI Study on Prostate Cancer  
    cd /home/groups/comp3710/HipMRI_Study_open/semantic_MRs$ — 3D MRI input volumes (.nii.gz)
    cd /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only$ — segmentation marked masks (prostate, bladder, rectum, bone, etc.)

    **Preprocessing:**
    - Loaded using **Nibabel**
    - Normalised to zero mean and unit variance
    - Cropped/padded to fixed size for 3D UNet input
    - Train/Validation/Test split ≈ 70% / 15% / 15%