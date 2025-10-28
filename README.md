# YOLO Image Detection (Ultralytics)

Simple command-line tool that runs object detection on an image or a folder of images using Ultralytics YOLOv8, saves annotated results, and can export detections to CSV. Includes a lightweight Tkinter GUI.

## Quickstart

```bash
# 1) Create and activate a virtual environment (macOS)
python3 -m venv .venv
source ./.venv/bin/activate

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Run detection on a single image
python detect.py --source path/to/image.jpg

# 4) Or run on a directory of images
python detect.py --source path/to/images_dir --model yolov8s.pt --conf 0.35

# 5) Run detection on a video file (annotated video output)
python detect.py --source path/to/video.mp4
```

- **Outputs**: Saved under `runs/detect/pred` by default (configurable via `--output`).
- **Models**: Use any Ultralytics YOLOv8 model like `yolov8n.pt`, `yolov8s.pt`, etc.
- **Device**: Auto-detects by default, or set `--device cpu` or `--device 0` for first GPU.

### CSV Export (CLI)

Add `--csv` to write detections to a CSV file:

```bash
python detect.py --source path/to/images_dir --csv outputs/detections.csv
```

CSV columns: `image,label,confidence,x_min,y_min,x_max,y_max,width,height`

## Arguments

```bash
python detect.py --help
```

Key flags:
- `--source`: Path to an image file or a directory of images (required)
- `--model`: Model weights, default `yolov8n.pt`
- `--conf`: Confidence threshold (default `0.25`)
- `--output`: Output directory (default `runs/detect`)
- `--device`: Compute device (e.g. `cpu`, `0`)

## Notes for Apple Silicon (M1/M2/M3)

PyTorch on Apple Silicon can use `mps` (Metal) backend. Ultralytics will auto-select it when available. If you hit install issues for `torch`, see the official instructions and consider:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then reinstall `ultralytics` if needed.

Alternatively, use the pre-curated file for macOS:

```bash
pip install -r requirements-macos.txt
```

## Example Output

Running on a folder prints per-image summaries and saves annotated images:

```
[1/3] dog.jpg
  → dog (0.93), person (0.88)
[2/3] street.png
  → car (0.90), traffic light (0.77)
[3/3] empty.jpg
  → No objects detected above threshold
Done in 2.34s. Outputs saved under: runs/detect/pred
```

## GUI Usage (Tkinter)

Run the GUI to select a file or folder and export detections to CSV:

```bash
python gui.py
```

- Select image file or folder
- Optionally enable "Save annotated images"
- Choose CSV output path
- Click "Run Detection"

## Docker

Build the image and run the CLI inside a container:

```bash
# Build
docker build -t yolo-detect .

# Show help (default CMD)
docker run --rm -it -v "$PWD:/app" yolo-detect --help

# Run detection on a mounted folder and write CSV to mounted path
docker run --rm -it -v "$PWD:/app" yolo-detect \
  --source /app/path/to/images \
  --csv /app/outputs/detections.csv
```

## Build a macOS App (PyInstaller)

Create a standalone app for macOS users:

```bash
chmod +x build-macos.sh
./build-macos.sh
# Output in ./dist (one-file binary by default)
```

Notes:
- If Gatekeeper blocks the app, you may need to allow it in System Settings → Privacy & Security.
- For distribution outside your machine, consider code-signing and notarization. See Apple docs on Developer ID signing.

## GitHub Setup

Initialize a local repo, then create a GitHub repo and push.

```bash
# One-time local commit
git init
git add .
# If git asks for identity, set local config:
# git config user.name "<your-name>"
# git config user.email "<your-email>"

git commit -m "Initial commit: YOLO detection CLI"

# Create a new empty repo on GitHub, then add the remote and push
# Replace <your-username> and <repo> accordingly

git branch -M main
git remote add origin git@github.com:<your-username>/<repo>.git
# or: https://github.com/<your-username>/<repo>.git

git push -u origin main
```

---

MIT License
