import torch
from huggingface_hub import hf_hub_download
import yaml


# Download model and config
config_path = hf_hub_download(repo_id="JusperLee/Dolphin", filename="conf.yml")
model_path = hf_hub_download(repo_id="JusperLee/Dolphin", filename="best_model.pth")


# Load model (you need to import Dolphin class from the repo)
with open(config_path) as f:
    config = yaml.safe_load(f)


model = Dolphin(**config['model'])
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()


# Prepare inputs
# audio: [batch, samples] - 16kHz audio
# video: [batch, frames, 1, height, width] - grayscale lip frames
audio_mixture = torch.randn(1, 64000)  # 4 seconds at 16kHz
video_frames = torch.randn(1, 100, 1, 88, 88)  # 4s at 25fps, 88x88 resolution


# Separate speech
with torch.no_grad():
    separated_audio = model(audio_mixture, video_frames)
