import torch
from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = r"/home/hariramanan/Project/madhan/HS_Dataset_v2/HS_wav_dataset_v2"
META_JSON = r"/home/hariramanan/Project/madhan/HS_Dataset_v2/meta_map_v2.json"

SAMPLE_RATE = 2000
MAX_SEC = 30
BATCH_SIZE = 16
window_sec=4
stride_sec=2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")