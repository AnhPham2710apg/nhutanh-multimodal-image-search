"""
Improved monitor using Kaggle Python API for accurate status detection
Usage: python kaggle_monitor_v2.py --type siglip2 --interval 30
"""
import time
import argparse
from pathlib import Path
import shutil
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess

def download_outputs(notebook_type, username):
    """Download and move outputs to result/"""
    kernel_slug = f"{username}/{notebook_type}-embed"
    temp_dir = Path(f"./temp_output_{notebook_type}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[DOWNLOAD] Downloading outputs from {kernel_slug}...")
    
    try:
        result = subprocess.run(
            f'kaggle kernels output {kernel_slug} -p "{temp_dir}"',
            capture_output=True,
            text=True,
            shell=True,
            timeout=180
        )
        
        if result.returncode != 0:
            print(f"  ✗ Error: {result.stderr}")
            return False
        
        # Move embeddings/index to result/
        result_dir = Path("./result")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        moved = 0
        
        for f in temp_dir.glob("*"):
            if f.is_file() and f.suffix != ".log":
                if "embedding" in f.name.lower() and f.suffix == ".npz":
                    new_name = f"{notebook_type}_embeddings_{timestamp}.npz"
                elif "faiss" in f.name.lower() and f.suffix == ".bin":
                    new_name = f"{notebook_type}_faiss_index_{timestamp}.bin"
                else:
                    continue
                
                dest = result_dir / new_name
                shutil.move(str(f), str(dest))
                size_mb = dest.stat().st_size / 1024**2
                print(f"  ✓ {new_name} ({size_mb:.2f} MB)")
                moved += 1
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if moved == 0:
            print(f"  ⚠ No output files found (only logs)")
            return False
        
        print(f"  ✓ Downloaded {moved} files to result/")
        return True
        
    except Exception as e:
        print(f"  ✗ Download error: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Monitor Kaggle kernel (API-based)")
    parser.add_argument("--type", required=True, choices=["siglip2", "dinov3"])
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--max-wait", type=int, default=1200, help="Max wait time in seconds")
    
    args = parser.parse_args()
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    username = api.get_config_value('username')
    kernel_slug = f"{username}/{args.type}-embed"
    
    print(f"Monitoring kernel: {kernel_slug}")
    print(f"Check interval: {args.interval}s")
    print(f"URL: https://www.kaggle.com/code/{kernel_slug}")
    print(f"\nWaiting for execution to complete...")
    
    start_time = time.time()
    check_count = 0
    last_status = None
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > args.max_wait:
            print(f"\n⚠ Timeout after {args.max_wait}s")
            print(f"  Check: https://www.kaggle.com/code/{kernel_slug}")
            print(f"  To download later: python kaggle_download.py --type {args.type}")
            return
        
        # Use kernels_status to get current execution status
        try:
            status_response = api.kernels_status(kernel_slug)
            current_status = status_response.get('status', 'unknown')
            
            if current_status == 'complete':
                print(f"\n✓ Kernel completed!")
                print(f"  Wait time: {elapsed:.0f}s ({elapsed/60:.1f} minutes)")
                
                # Download outputs
                if download_outputs(args.type, username):
                    print(f"\n✓ Pipeline completed successfully!")
                else:
                    print(f"\n⚠ Kernel finished but no outputs found")
                    print(f"  Check logs: https://www.kaggle.com/code/{kernel_slug}")
                
                return
            
            elif current_status in ['error', 'cancelAcknowledged', 'cancelled']:
                print(f"\n✗ Kernel execution failed: {current_status}")
                print(f"  Check logs: https://www.kaggle.com/code/{kernel_slug}")
                return
            
            # Print status every 2 checks (~40s with 20s interval)
            if current_status != last_status or check_count % 2 == 0:
                print(f"[{int(elapsed)}s] Status: {current_status}")
                last_status = current_status
            
            check_count += 1
                
        except Exception as e:
            print(f"[{int(elapsed)}s] API error: {e}, retrying...")
        
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
