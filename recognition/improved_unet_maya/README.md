# Prostate 3D Segmentation using Improved UNet3D 
**student number: 47400548**
This project performs 3D medical image segmentation on the HipMRI Study on Prostate Cancer dataset using an Improved UNet3D architecture.  
The goal is to segment the prostate and other anatomical structures within MRI volumes, achieving mean Dice ≥ 0.70 on the test set.  
Final performance: **mean Dice = 0.72 (the target was met, specified during PREDICT)**.


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

    ## Test Driver
    To reproduce the final test evaluation on Rangpur:
    ```bash
    python test_driver.py
    This script loads the trained model

    ## Dataset

    **Dataset:** HipMRI Study on Prostate Cancer  
    **Location (Rangpur):** `/home/groups/comp3710/HipMRI_Study_open/`

    - **semantic_MRs/** → 3D MRI input volumes  
    - **semantic_labels_only/** → Corresponding segmentation masks  
    - Loaded with Nibabel, normalised to zero-mean/unit-variance  
    - Cropped/padded to consistent shape  
    - Split 70% train, 15% val, 15% test


    Validation metrics from training (train.py) and final test Dice from predict.py

    | Metric               | Training Dice | Validation Dice | Test Dice | Target | Met Target? |
    |----------------------|---------------|------------------|-----------|--------|-------------|
    | Mean Dice Coefficient| —             | 0.714           | **0.72** | ≥ 0.70 | yesss |
    | Validation Loss      | —             | 0.206           | —         | —      | —           |
    | Epochs Trained       | 8            | —                | —         | —      | —           |


    ### Training Curves
    ![Training vs Validation Loss](metrics/training_loss_curve.png) 
    ![Validation Mean Dice](metrics/val_dice_curve.png) 
    *Both training and validation loss decrease steadily while the Dice coefficient improves to 0.72, indicating stable training and convergence.*
    
    ### Example Predictions (Test Set)
    **Ground-truth overlay** vs **Model prediction** for case B006_Week0 (slice 64)

    | Ground Truth (MRI + GT overlay) | Model Prediction (MRI + Pred overlay) |
    | ![](images/example_gt_overlay.png) | ![](images/example_pred.png) |
    These images display an example MRI slice from the HipMRI dataset — the left shows the ground-truth anatomical segmentation overlay, while the right shows the model’s predicted segmentation produced by the Improved UNet3D.

    ## Data (Rangpur)
    **Dataset:** HipMRI Study on Prostate Cancer  
    cd /home/groups/comp3710/HipMRI_Study_open/semantic_MRs$ — 3D MRI input volumes (.nii.gz)
    cd /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only$ — segmentation marked masks (prostate, bladder, rectum, bone, etc.)

    **Preprocessing:**
    - Loaded using **Nibabel**
    - Normalised to zero mean and unit variance
    - Cropped/padded to fixed size for 3D UNet input
    - Train/Validation/Test split ≈ 70% / 15% / 15%

    ## Environment & Execution (Rangpur HPC)

    projects/
└── final_project/
    ├── PatternAnalysis-2025/
    │   └── recognition/
    │       ├── improved_unet_maya/
    │       │   ├── dataset.py
    │       │   ├── modules.py
    │       │   ├── predict.py
    │       │   ├── train.py
    │       │   ├── test_driver.py
    │       │   ├── utils.py
    │       │   ├── README.md
    │       │   ├── __pycache__/
    │       │   ├── metrics/
    │       │   │   ├── training_loss_curve.png
    │       │   │   ├── val_dice_curve.png
    │       │   │   └── ...
    │       │   ├── images/
    │       │   │   ├── B006_Week0_LFOV.nii.gz
    |       |   |   ├── example_pred.png
    │       │   │   ├── example_gt_overlay.png
    │       │   │   ├── B006_Week0_SEMANTIC.nii.gz
    │       │   │   └── view_slice.py
    │       │   └── runs/
    │       │       └── unet3d/
    │       │           ├── logs/
    │       │           │   ├── 3d_loss.png
    │       │           │   └── 3d_dice.png
    │       │           ├── checkpoints/
    │       │           │   └── best_unet3d.pt <-- training data 
    │       │           └── metrics.csv
    │       └── README.md
    ├── logs/
    │   ├── unet3d_gpu_predict_324801.err
    │   ├── unet3d_gpu_predict_324801.out   → final Dice = 0.7200782299
    │   └── ...
    ├── runners/
    │   ├── unet3d_gpu_train.sbatch <-- runner
    │   └── unet3d_gpu_predict.sbatch <-- runner
    └── models/
        └── improved_unet3d/
            └── unet3d/
                ├── checkpoints/
                │   └── best_unet3d.pt
                ├── logs/
                │   ├── 3d_loss.png
                │   └── 3d_dice.png
                ├── test_driver_out/
                │   ├── test_dice.json
                │   └── test_dice.csv
                └── metrics.csv

   
    
    SUMMARY 
    The Improved UNet3D successfully segmented prostate MRI volumes from the HipMRI dataset, achieving a mean Dice of 0.72 on unseen test data.
    Training and validation metrics demonstrated stable convergence, and the final model generalized well across cases.
    All results were produced reproducibly on UQ’s Rangpur A100 GPU cluster.