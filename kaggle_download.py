"""
Download outputs from completed Kaggle kernel
Usage: python kaggle_download.py --type siglip2
"""
import subprocess
import sys
import argparse
from pathlib import Path
import shutil
from datetime import datetime

def run_cmd(cmd, timeout=60):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timeout"

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle kernel outputs")
    parser.add_argument("--type", required=True, choices=["siglip2", "dinov3"])
    args = parser.parse_args()
    
    # Get username
    rc, out, _ = run_cmd("kaggle config view")
    username = None
    for line in out.split('\n'):
        if 'username:' in line.lower():
            username = line.split(':')[1].strip()
            break
    
    kernel_slug = f"{username}/{args.type}-embed"
    
    print(f"Downloading outputs from {kernel_slug}...")
    
    # Download to temp directory
    temp_dir = Path(f"./temp_output_{args.type}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    rc, out, err = run_cmd(f'kaggle kernels output {kernel_slug} -p "{temp_dir}"', timeout=180)
    
    if rc != 0:
        print(f"Error downloading: {err}")
        print(f"\nCheck if kernel finished: https://www.kaggle.com/code/{kernel_slug}")
        sys.exit(1)
    
    # Move to result/ with proper naming
    result_dir = Path("./result")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = list(temp_dir.glob("*"))
    
    if not files:
        print("No output files found")
        sys.exit(1)
    
    print(f"\nDownloaded {len(files)} files:")
    for f in files:
        if f.is_file():
            # Rename with type and timestamp
            if "embedding" in f.name.lower() and f.suffix == ".npz":
                new_name = f"{args.type}_embeddings_{timestamp}.npz"
            elif "faiss" in f.name.lower() and f.suffix == ".bin":
                new_name = f"{args.type}_faiss_index_{timestamp}.bin"
            else:
                new_name = f"{args.type}_{f.name}"
            
            dest = result_dir / new_name
            shutil.move(str(f), str(dest))
            size_mb = dest.stat().st_size / 1024**2
            print(f"  ✓ {new_name} ({size_mb:.2f} MB)")
    
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n✓ All files saved to result/")

if __name__ == "__main__":
    main()
