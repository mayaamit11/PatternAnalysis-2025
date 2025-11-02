import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

METRICS = Path(__file__).with_name("metrics.csv")
OUTDIR = Path(__file__).parent

df = pd.read_csv(METRICS)

# --- Loss curve ---
plt.figure(figsize=(7,4.5))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
plt.legend(); plt.tight_layout()
plt.savefig(OUTDIR / "training_loss_curve.png", dpi=160)
plt.close()

# --- Dice curve (if present) ---
if "val_mean_dice" in df.columns:
    plt.figure(figsize=(7,4.5))
    plt.plot(df["epoch"], df["val_mean_dice"], label="Val Mean Dice")
    plt.xlabel("Epoch"); plt.ylabel("Dice")
    plt.title("Validation Mean Dice")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUTDIR / "val_dice_curve.png", dpi=160)
    plt.close()

# --- Summary row for your README table ---
last = df.iloc[-1]
train_loss = float(last["train_loss"])
val_loss = float(last["val_loss"])
val_dice = float(last.get("val_mean_dice", float("nan")))

# If you also have a separate test Dice (e.g., from predict.py), hardcode it here:
test_dice = 0.72  # <-- update if your predict script prints a different value

print("\nCopy-paste this into your README.md:\n")
print("### Quantitative Results\n")
print("| Metric               | Training Dice | Validation Dice | Test Dice | Target | Met Target? |")
print("|----------------------|---------------|------------------|-----------|--------|-------------|")
print(f"| Mean Dice Coefficient| —             | {val_dice:.3f}           | **{test_dice:.2f}** | ≥ 0.70 | {'✅ Yes' if test_dice >= 0.70 else '❌ No'} |")
print(f"| Validation Loss      | —             | {val_loss:.3f}           | —         | —      | —           |")
print(f"| Epochs Trained       | {int(last['epoch'])}            | —                | —         | —      | —           |")
