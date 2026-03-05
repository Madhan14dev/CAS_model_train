import torch
import torch.quantization as tq
from models.murmur_model import MurmurModel

best_model_path = "/home/hariramanan/Desktop/madhan/hs_model/hs_murmur_bilstm_tp_gate_model_2khz_400_nfft.pt"

print("Loading best FP32 model...")
model = MurmurModel()
model.load_state_dict(
    torch.load(best_model_path, map_location="cpu", weights_only=False)
)

model.eval()

print("Applying selective dynamic quantization...")

# ✅ Quantize ONLY BiLSTM
model.bilstm = tq.quantize_dynamic(
    model.bilstm,
    {torch.nn.LSTM},
    dtype=torch.qint8
)

# ✅ Quantize ONLY classifier (Linear layers)
model.classifier = tq.quantize_dynamic(
    model.classifier,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# ❌ DO NOT quantize model.temporal
# ❌ DO NOT quantize model.encoder

torch.save(
    model.state_dict(),
    "/home/hariramanan/Desktop/madhan/hs_model/hs_murmur_quantized_int8.pt"
)

print("Selective quantized model saved successfully!")