from model import Unet
import torch 
import numpy as np
import cv2
from sacred import Experiment
import torch.nn.functional as F
import os

ex = Experiment('image segmentation')

@ex.automain
def main(img_path):
    img = cv2.imread(img_path).astype(np.float32) /255.0
    model = Unet()
    ckpt = "ckpt_best_loss.pth"
    states = torch.load(ckpt)
    model.load_state_dict(states)
    tensored = torch.from_numpy(img)[None, :, : ,:]
    model.eval()
    with torch.inference_mode():
        segmented = F.sigmoid(model(tensored)).squeeze().numpy()
        cv2.imwrite(f'{img_path}_segmented.png', segmented*255)
        print(f'image generated at f{os.path.abspath(img_path)}_segmented.png')