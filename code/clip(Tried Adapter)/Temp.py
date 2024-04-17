import torch
import numpy as np
import torch.nn as nn
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
import wandb
import pytorch_warmup as warmup
from tqdm import tqdm

# %%
import numpy as np
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(clip.available_models())
# %%
model, preprocess = clip.load("RN50", device=device)
image = preprocess(Image.open("alarm.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a picture of a alarm", 'a alarm', ]).to(device)
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print("Label probs:", probs)
nn.GELU
