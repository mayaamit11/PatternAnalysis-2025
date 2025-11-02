   # Prostate 3D Segmentation using Improved UNet3D

This project performs 3D medical image segmentation on the HipMRI Study on Prostate Cancer dataset using an **Improved UNet3D** architecture.  
The goal is to segment the prostate and other anatomical structures within MRI volumes, achieving **mean Dice ≥ 0.70** on the test set.  
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

    projects/final_project/runners
    ├─ logs #
    │  ├─ unet3d_gpu_predict_324801.err 
    |  ├─ unet3d_gpu_predict_324801.out --> # PREDICT RESULT the final dice value of 0.7200782299041748 
    |  ├─ ...
    │  └─ ...
    ├─ unet3d_gpu_train.sbatch #this file was ran first to train the model, produced a predict value but uses
    ├─ unet3d_gpu_predict.sbatch #after train was ran predict was then ran using weights and bias... 

    within projects/final_project/PatternAnalysis-2025/recognition/improved_unet_maya$

    improved_unet_maya/
    └── runs/
        └── unet3d/
            ├── checkpoints/
            │   └── best_unet3d.pt  #this is where the weights and bias post unet3d_gpu_train are stored 
            ├── logs/
            │   └── 3d_loss.png # TRAINING RESULT 
            │   └── 3d_dice.png # TRAINING RESULT dice validation per epoch
            └── metrics.csv

    ## --- unet3d_gpu_train.sbatch --- ##

        #!/bin/bash -l
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task=2
        #SBATCH --gres=gpu:a100:1
        #SBATCH --job-name=unet3d_gpu_predict
        #SBATCH -o logs/%x_%j.out
        #SBATCH -e logs/%x_%j.err
        #SBATCH --partition=comp3710
        #SBATCH -A comp3710
        #SBATCH --time=03:00:00

        set -euo pipefail
        mkdir -p logs

        echo "HOME=$HOME"
        hostname
        date
        echo "JOB ID=${SLURM_JOB_ID:-unknown}"

        # --- Env ---
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate torch
        which python && python -V

        # --- Paths ---
        CODE_DIR="/home/Student/s4740054/projects/final_project/PatternAnalysis-2025/recognition/improved_unet_maya"
        MR_DIR="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
        LBL_DIR="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

        # --- Predict config ---
        CROP=160
        OUT_DIR="./runs/unet3d/predict_gpu"

        cd "$CODE_DIR"

        # find newest checkpoint from training
        CKPT=$(ls -t ./runs/unet3d/checkpoints/*.pt 2>/dev/null | head -n 1 || true)
        if [[ -z "$CKPT" ]]; then
        echo "ERROR: No checkpoint found in ./runs/unet3d/checkpoints/"
        exit 1
        fi
        echo "Using checkpoint: $CKPT"

        echo "==> PREDICT (GPU, 3D)  crop=$CROP"
        python -u predict.py \
        --task prostate3d \
        --mr_dir "$MR_DIR" \
        --lbl_dir "$LBL_DIR" \
        --crop "$CROP" \
        --device cuda \
        --ckpt3d "$CKPT" \
        --out_dir "$OUT_DIR" \
        --batch 1

        echo "==> Outputs (top-level):"
        find "$OUT_DIR" -maxdepth 2 -type f | head -n 30 || true

        date
        echo "Done."

    ## ---  unet3d_gpu_predict.sbatch --- ##
        #!/bin/bash -l
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task=2
        #SBATCH --gres=gpu:a100:1
        #SBATCH --job-name=unet3d_gpu_train
        #SBATCH --partition=comp3710
        #SBATCH -A comp3710
        #SBATCH --time=03:00:00                   # use 3h explicitly (partition default)
        #SBATCH -o /home/Student/s4740054/projects/final_project/logs/%x_%j.out
        #SBATCH -e /home/Student/s4740054/projects/final_project/logs/%x_%j.err

        set -euo pipefail
        mkdir -p /home/Student/s4740054/projects/final_project/logs

        echo "HOME=$HOME"
        hostname
        date
        echo "JOB ID=${SLURM_JOB_ID:-unknown}"

        # --- Env ---
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate torch
        which python && python -V

        # --- Paths (match your layout) ---
        CODE_DIR="/home/Student/s4740054/projects/final_project/PatternAnalysis-2025/recognition/improved_unet_maya"
        MR_DIR="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
        LBL_DIR="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

        # --- Train config ---
        CROP=160          # must be divisible by model stride
        EPOCHS=8          # increase later for full run
        LR=3e-4
        OUT_DIR="./runs/unet3d"

        cd "$CODE_DIR"
        echo "==> TRAIN (GPU, 3D) | crop=$CROP  epochs=$EPOCHS  lr=$LR"

        python -u train.py \
        --task prostate3d \
        --mr_dir "$MR_DIR" \
        --lbl_dir "$LBL_DIR" \
        --crop "$CROP" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --device cuda \
        --out_dir ./runs

        echo "==> Checkpoints:"
        ls -lh ./runs/unet3d/checkpoints || true

        date
        echo "✅ Done."
    
    SUMMARY 
    The Improved UNet3D successfully segmented prostate MRI volumes from the HipMRI dataset, achieving a mean Dice of 0.72 on unseen test data.
    Training and validation metrics demonstrated stable convergence, and the final model generalized well across cases.
    All results were produced reproducibly on UQ’s Rangpur A100 GPU cluster.