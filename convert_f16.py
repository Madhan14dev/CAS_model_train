import torch
import os
from models.ctenn_model import CTENN_Murmur

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
fp32_path = "/home/hariramanan/Project/madhan/hs_model/hs_murmur_ctenn_new_gate_model.pt"
fp16_path = "/home/hariramanan/Project/madhan/hs_model/hs_murmur_fp16_ctenn_new.pt"

print("Loading FP32 model...")

model = CTENN_Murmur()
model.load_state_dict(
    torch.load(fp32_path, map_location="cpu", weights_only=False)
)

model.eval()

print("Converting model to mixed FP16 (safe version)...")

# Convert entire model to FP16
model.half()

# 🔥 VERY IMPORTANT: revert mel module back to FP32
# model.mel.float()

# ---------------------------------------------------
# Save
# ---------------------------------------------------
torch.save(model.state_dict(), fp16_path)

print(f"FP16 model saved at:\n{fp16_path}")

# ---------------------------------------------------
# Show Size Comparison
# ---------------------------------------------------
fp32_size = os.path.getsize(fp32_path) / (1024**2)
fp16_size = os.path.getsize(fp16_path) / (1024**2)

print("\nSize Comparison:")
print(f"FP32 size: {fp32_size:.2f} MB")
print(f"FP16 size: {fp16_size:.2f} MB")
print(f"Reduction : {((fp32_size - fp16_size)/fp32_size)*100:.2f}%")