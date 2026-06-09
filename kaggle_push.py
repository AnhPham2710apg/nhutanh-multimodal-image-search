"""
Simplified Kaggle Pipeline - Push only mode
Usage: 
  1. Push: python kaggle_push.py --type siglip2
  2. Wait ~10-15 minutes, check: https://kaggle.com/code/username/kernel-name
  3. Download: python kaggle_download.py --type siglip2
"""
import subprocess
import sys
import argparse
from pathlib import Path
import shutil
import json

def run_cmd(cmd, timeout=60):
    """Run command with error handling"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timeout"

def main():
    parser = argparse.ArgumentParser(description="Push Kaggle notebook (no wait)")
    parser.add_argument("--type", required=True, choices=["siglip2", "dinov3"])
    parser.add_argument("--dataset", default="chetankv/dogs-cats-images")
    args = parser.parse_args()
    
    print(f"Pushing {args.type} notebook to Kaggle...")
    
    # Prepare kernel directory
    kernel_dir = Path(f"./temp_kernel_{args.type}")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy notebook
    notebook_src = Path(f"./notebook/{args.type}-embed.ipynb")
    if not notebook_src.exists():
        notebook_src = Path(f"./notebook/{'dino-v3' if args.type == 'dinov3' else 'siglip-2'}-embed.ipynb")
    
    shutil.copy(notebook_src, kernel_dir / f"{args.type}-embed.ipynb")
    
    # Get username
    rc, out, _ = run_cmd("kaggle config view")
    username = None
    for line in out.split('\n'):
        if 'username:' in line.lower():
            username = line.split(':')[1].strip()
            break
    
    kernel_slug = f"{username}/{args.type}-embed"
    
    # Create metadata
    metadata = {
        "id": kernel_slug,
        "title": f"{args.type.upper()} Embed",
        "code_file": f"{args.type}-embed.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [args.dataset],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    with open(kernel_dir / "kernel-metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Push
    print(f"Pushing to Kaggle...")
    rc, out, err = run_cmd(f'kaggle kernels push -p "{kernel_dir}"', timeout=60)
    
    if rc != 0:
        print(f"Error: {err}")
        sys.exit(1)
    
    shutil.rmtree(kernel_dir, ignore_errors=True)
    
    print(f"\n✓ Kernel pushed: {kernel_slug}")
    print(f"  Monitor: https://www.kaggle.com/code/{kernel_slug}")
    print(f"\nWait 10-15 minutes, then download:")
    print(f"  python kaggle_download.py --type {args.type}")

if __name__ == "__main__":
    main()
