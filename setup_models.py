import os
import requests

MODEL_DIR = "models"
SRC_DIR = "src"

FILES = [
    "IFNet_HDv3.py", "RIFE_HDv3.py", "refine.py", "flownet.pkl"
]
BASE_URL = "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/"

def download_file(filename):
    url = BASE_URL + filename
    path = os.path.join(MODEL_DIR, filename)
    print(f"Downloading {filename}...")
    r = requests.get(url, allow_redirects=True)
    with open(path, 'wb') as f:
        f.write(r.content)

def patch_file(filename, old, new):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, 'r') as f: content = f.read()
    if old in content:
        with open(path, 'w') as f: f.write(content.replace(old, new))

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for f in FILES:
        download_file(f)
    
    # Apply patches to make downloaded models work with local src files
    print("Patching files...")
    patch_file("RIFE_HDv3.py", "from train_log.IFNet_HDv3", "from IFNet_HDv3")
    patch_file("RIFE_HDv3.py", "from model.loss", "from src.loss")
    patch_file("RIFE_HDv3.py", "from model.warplayer", "from src.warplayer")
    patch_file("IFNet_HDv3.py", "from model.warplayer", "from src.warplayer")
    patch_file("IFNet_HDv3.py", "from model.loss", "from src.loss")
    
    print("Setup complete.")

if __name__ == "__main__":
    main()