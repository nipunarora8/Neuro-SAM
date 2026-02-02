import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Constants for Model URLs
WEIGHTS_URLS = {
    "DeepD3_Benchmark.tif": "https://github.com/nipunarora8/Neuro-SAM/releases/download/weights/DeepD3_Benchmark.tif",
    "dendrite_model.torch": "https://github.com/nipunarora8/Neuro-SAM/releases/download/weights/dendrite_model.torch",
    "sam2.1_hiera_small.pt": "https://github.com/nipunarora8/Neuro-SAM/releases/download/weights/sam2.1_hiera_small.pt",
    "punet_best.pth": "https://github.com/nipunarora8/Neuro-SAM/releases/download/weights/punet_best.pth"
}

def get_weights_dir():
    """Get the directory where weights are stored (~/.neuro_sam/checkpoints)."""
    weights_dir = Path.home() / ".neuro_sam" / "checkpoints"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir

def download_file(url, dest_path):
    """Download a file from a URL to a destination path with a progress bar."""
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        block_size = 1024 # 1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                t.update(len(data))
                file.write(data)
        t.close()
        
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong with the download")
            return False
        
        print(f"Download complete: {dest_path}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def get_weights_path(filename, url=None):
    """
    Get the path to a weights file. 
    Checks local 'checkpoints' folder first, then ~/.neuro_sam/checkpoints.
    If not found, downloads it.
    """
    # 1. Check local checkpoints folder (development mode)
    local_path = Path("checkpoints") / filename
    if local_path.exists():
        return str(local_path.absolute())
        
    # 2. Check global cache directory
    weights_dir = get_weights_dir()
    cache_path = weights_dir / filename
    
    if cache_path.exists():
        return str(cache_path.absolute())
        
    # 3. Download if not found
    if url is None:
        url = WEIGHTS_URLS.get(filename)
        
    if url:
        print(f"Weights file {filename} not found locally. Downloading...")
        success = download_file(url, cache_path)
        if success:
            return str(cache_path.absolute())
        else:
            raise RuntimeError(f"Failed to download {filename}")
    else:
        raise FileNotFoundError(f"Weights file {filename} not found and no URL provided.")

def download_all_models():
    """Download all known models to the cache directory."""
    print("Downloading all Neuro-SAM models...")
    for filename, url in WEIGHTS_URLS.items():
        try:
            get_weights_path(filename, url)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
    print("All downloads processed.")
