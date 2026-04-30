import torch
import torch.nn.functional as F
from model.model import build_model
import gdown
import json
import os

# Google Drive download
def download_model():
    if not os.path.exists("model/neuralnet.pth"):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(
            id="18bju8k1A8KiCsHGLaySPt0KH-p_uaWLy",  # Google Drive file ID
            output="model/neuralnet.pth",
            quiet=False
        )
        print("Model downloaded successfully.")

#Load class mapping
# with open("model/class_to_idx.json", "r") as f:
#     class_to_idx = json.load(f)

def load_model():
    download_model()
    model = build_model()
    model.load_state_dict(
        torch.load("model/neuralnet.pth", map_location="cpu")
    )
    model.eval()
    return model

def predict(model, tensor, classes):
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)

        top_prob, top_class = torch.max(probs, 1)

    return classes[top_class.item()], top_prob.item()