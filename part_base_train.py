import os
import subprocess


DATA_DIR = "/mnt/data/jet_data"
SAVE_DIR = "/mnt/data/output"
os.makedirs(SAVE_DIR, exist_ok=True)

filelist_path = os.path.join(DATA_DIR, "filelist.txt")
os.makedirs(DATA_DIR, exist_ok=True)

# Download filelist to PVC
if not os.path.exists(filelist_path):
    subprocess.run(["wget", "https://huggingface.co/datasets/jet-universe/jetclass2/resolve/main/filelist.txt", "-O", filelist_path], check=True)

# Download parquet files into PVC
if len(os.listdir(DATA_DIR)) <= 1:  # only filelist.txt exists
    print("Downloading JetClass-II parquet files...")
    subprocess.run(["wget", "-c", "-i", filelist_path, "-P", DATA_DIR], check=True)


#sort data into tau vs QCD background events
