import sys
import cv2
import torch
import numpy as np
import time
from torch.nn import functional as F
# Import RIFE models dynamically or assume they are in path after setup
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_inference(model, I0, I1, n):
    if n == 1:
        return []
    
    if model.version >= 3.9:
        mid = model.inference(I0, I1, 0.5, 1.0)
    else:
        mid = model.inference(I0, I1)

    return make_inference(model, I0, mid, n//2) + [mid] + make_inference(model, mid, I1, n//2)

def process_video(video_path, output_path, multiplier, scale=1.0):
    # Dynamic import to handle the RIFE architecture loading
    sys.path.append('models') 
    from RIFE_HDv3 import Model
    
    model = Model()
    model.load_model('models', -1)
    model.eval()
    model.device()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    if fps == 0: fps = 24.0
    
    # Padding calculation
    s_h, s_w = int(h * scale), int(w * scale)
    ph = ((s_h - 1) // 64 + 1) * 64
    pw = ((s_w - 1) // 64 + 1) * 64
    padding = (0, pw - s_w, 0, ph - s_h)
    
    target_fps = fps * multiplier
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (s_w, s_h))
    
    last = None
    
    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (s_w, s_h))
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = F.pad(I1, padding)
            
            if last is not None:
                mid_frames = make_inference(model, last, I1, multiplier)
                for mid in mid_frames:
                    mid = mid[0].cpu().numpy()
                    mid = np.transpose(mid, (1, 2, 0))
                    mid = (mid * 255).astype(np.uint8)
                    mid = mid[:s_h, :s_w]
                    writer.write(mid)
            
            writer.write(frame)
            last = I1
            
    cap.release()
    writer.release()