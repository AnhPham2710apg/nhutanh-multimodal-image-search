# Kaggle Embeddings Pipeline

Automated pipeline for generating image embeddings using SigLIP2 and DINOv3 models on Kaggle's free GPU infrastructure.

## Quick Start

```bash
# 1. Push notebook to Kaggle
python kaggle_push.py --type siglip2

# 2. Wait 10-15 minutes (check https://kaggle.com/code/YOUR_USERNAME/siglip2-embed)

# 3. Download results
python kaggle_download.py --type siglip2
```

Results saved to `result/` as `siglip2_embeddings_YYYYMMDD_HHMMSS.npz` (3.22 MB) + FAISS index.

## Prerequisites

1. **Kaggle API credentials**: Download `kaggle.json` from https://kaggle.com/settings/account
2. **HuggingFace token** (for DINOv3): Get from https://huggingface.co/settings/tokens
3. **Python 3.11+** with uv

## One-Time Setup

### 1. Install Dependencies
```bash
uv sync
```

### 2. Configure Kaggle API
```bash
# Windows
mkdir %USERPROFILE%\.kaggle
# Copy kaggle.json to %USERPROFILE%\.kaggle\kaggle.json

# Linux/Mac  
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Create Kernels on Kaggle (Manual - Required)

**For SigLIP2:**
1. Go to https://kaggle.com/code → **New Notebook**
2. Title: **`siglip2-embed`** (exact name)
3. **Add Data** → Search `dogs-cats-images` → Add `chetankv/dogs-cats-images`
4. Settings: **Internet ON**, Accelerator **GPU T4** or **None** (CPU)
5. **Save Version** (don't need to run)

**For DINOv3:**
1. Go to https://kaggle.com/code → **New Notebook**
2. Title: **`dinov3-embed`** (exact name)
3. **Add Data** → Add `chetankv/dogs-cats-images`
4. **Add-ons** → **Secrets** → Add `HF_TOKEN` (value from HuggingFace)
5. Settings: **Internet ON**, Accelerator **None** (CPU recommended)
6. **Save Version**

## Usage

### Simple Workflow (Recommended)

**SigLIP2 Pipeline:**
```bash
# Step 1: Push notebook (triggers execution)
python kaggle_push.py --type siglip2

# Step 2: Wait 10-15 minutes
# Monitor: https://kaggle.com/code/YOUR_USERNAME/siglip2-embed

# Step 3: Download outputs
python kaggle_download.py --type siglip2
```

**DINOv3 Pipeline:**
```bash
python kaggle_push.py --type dinov3
# Wait 10-15 minutes
python kaggle_download.py --type dinov3
```

### Automated Monitoring (Experimental)

```bash
.\.venv\Scripts\python.exe kaggle_monitor_v2.py --type siglip2 --interval 30
```

**Note:** Monitor may not detect completion reliably. Manual check after 10-15 minutes is recommended.

## Outputs

Files saved to `result/`:
- **SigLIP2**: `siglip2_embeddings_YYYYMMDD_HHMMSS.npz` (3.22 MB) + `siglip2_faiss_index_YYYYMMDD_HHMMSS.bin` (2.93 MB)
- **DINOv3**: `dinov3_embeddings_YYYYMMDD_HHMMSS.npz` (2.98 MB) + `dinov3_faiss_index_YYYYMMDD_HHMMSS.bin` (2.93 MB)

## Project Structure

```
├── notebook/
│   ├── siglip-2-embed.ipynb    # SigLIP2 notebook with auto path detection
│   └── dino-v3-embed.ipynb     # DINOv3 notebook with HF token support
├── result/                      # Pipeline outputs (timestamped)
│   └── .gitkeep
├── kaggle_push.py              # Push notebook & trigger execution
├── kaggle_download.py          # Download completed outputs
├── kaggle_monitor_v2.py        # Auto-monitor (optional)
├── app.py                      # Flask image search app
└── prepare_embeddings.py       # Local embedding generation
```

## Advanced Options

### Custom Dataset
```bash
python kaggle_push.py --type siglip2 --dataset your-username/your-dataset
```

### Local Embedding Generation (CPU)
```bash
python prepare_embeddings.py
```

### Flask Image Search App
```bash
python app.py
# Open http://localhost:5000
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Kernel not found"** | Create kernel manually on Kaggle web first (see Setup step 3) |
| **"401 Unauthorized" (DINOv3)** | Add `HF_TOKEN` to Kaggle Secrets in kernel settings |
| **"Found 0 images"** | Ensure dataset added as Input in kernel settings |
| **Monitor timeout** | Normal - check manually after 10-15 min or use `kaggle_download.py` |

## Technical Notes

- **Execution Time**: 10-15 minutes (CPU mode, 1000 images)
- **CPU vs GPU**: CPU forced to avoid P100 compatibility issues
- **Dataset Path**: Auto-detects `/kaggle/input/dogs-cats-images/...` or `/kaggle/input/chetankv/...`
- **Timestamped Outputs**: Prevents overwrites on multiple runs

## GitHub Actions (Optional)

Workflow available in `.github/workflows/kaggle-pipeline.yml`:
1. Add repository secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`
2. Trigger: **Actions** → **Kaggle Embedding Pipeline** → **Run workflow**

**SigLIP2 Kernel:**
1. Go to https://kaggle.com/code → New Notebook
2. Title: `siglip2-embed`
3. Add Data: Search `dogs-cats-images` → Add `chetankv/dogs-cats-images`
4. Settings: Internet ON, Accelerator GPU T4 (or None for CPU)
5. Save Version

**DINOv3 Kernel:**
1. Go to https://kaggle.com/code → New Notebook
2. Title: `dinov3-embed`
3. Add Data: `chetankv/dogs-cats-images`
4. Add-ons → Secrets: Add `HF_TOKEN` (from https://huggingface.co/settings/tokens)
5. Settings: Internet ON, Accelerator GPU T4 (or None)
6. Save Version

## Usage

### Simple Workflow (Recommended)

**Step 1: Push Notebook**
```bash
python kaggle_push.py --type siglip2
# or
python kaggle_push.py --type dinov3
```

**Step 2: Wait 10-15 minutes** (check https://kaggle.com/code/YOUR_USERNAME/KERNEL_NAME)

**Step 3: Download Results**
```bash
python kaggle_download.py --type siglip2
# or
python kaggle_download.py --type dinov3
```

### Automated Monitoring (Optional)

Monitor kernel execution and auto-download when complete:
```bash
.\.venv\Scripts\python.exe kaggle_monitor_v2.py --type siglip2 --interval 30 --max-wait 1200
```

**Note:** Monitor script may timeout but kernel continues running. Check manually after 10-15 minutes.

## Output Files

Results saved to `result/` directory:
- `siglip2_embeddings_YYYYMMDD_HHMMSS.npz` (3.22 MB)
- `siglip2_faiss_index_YYYYMMDD_HHMMSS.bin` (2.93 MB)
- `dinov3_embeddings_YYYYMMDD_HHMMSS.npz` (2.98 MB)
- `dinov3_faiss_index_YYYYMMDD_HHMMSS.bin` (2.93 MB)

## Project Structure

```
.
├── notebook/                    # Kaggle notebook sources
│   ├── siglip-2-embed.ipynb    # SigLIP2 embedding generator
│   └── dino-v3-embed.ipynb     # DINOv3 embedding generator
├── result/                      # Pipeline outputs (*.npz, *.bin)
├── kaggle_push.py              # Push notebook to Kaggle
├── kaggle_download.py          # Download outputs from Kaggle
├── kaggle_monitor_v2.py        # Monitor kernel execution (API-based)
├── app.py                      # Flask web app (image search)
├── prepare_embeddings.py       # Local embedding generation
└── pyproject.toml              # Dependencies
```

## Advanced Usage

### Custom Dataset
```bash
python kaggle_push.py --type siglip2 --dataset your-username/your-dataset
```

### Local Embedding Generation
```bash
python prepare_embeddings.py
```

### Run Flask App
```bash
python app.py
# Open http://localhost:5000
```

## Troubleshooting

**"Kernel not found" error**
- Create the kernel manually on Kaggle web first (see Setup step 3)

**"401 Unauthorized" (DINOv3)**
- Add HF_TOKEN to Kaggle Secrets in kernel settings

**"Found 0 images" in logs**
- Ensure dataset is added as Input in kernel settings

**Monitor script doesn't detect completion**
- Normal behavior, check manually after 10-15 minutes
- Or use: `python kaggle_download.py --type {siglip2|dinov3}`

## Performance Notes

- **Execution Time**: 10-15 minutes per kernel (CPU mode)
- **GPU Mode**: Not recommended due to P100 compatibility issues
- **Dataset Size**: Tested with 1000 images (cats dataset)

## GitHub Actions (Optional)

Automated workflow available in `.github/workflows/kaggle-pipeline.yml`:
1. Add secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`
2. Trigger: Actions → Kaggle Embedding Pipeline → Run workflow

## License

MIT
