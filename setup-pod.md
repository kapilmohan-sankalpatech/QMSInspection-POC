# RunPod Setup and Inference

## 1. Configure SSH on the RunPod

Run the following commands inside the RunPod container:

```bash
# Enable password authentication
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Allow root login with password
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config

# Generate SSH host keys if they do not exist
ssh-keygen -A

# Verify SSH configuration
if ! sshd -t; then
    echo "ERROR: SSH configuration is invalid."
    exit 1
fi

echo "SSH configuration is valid."

# Set root password using SSH_PASSWORD if provided
if [ -n "${SSH_PASSWORD}" ]; then
    echo "root:${SSH_PASSWORD}" | chpasswd
    echo "Root password configured."
else
    echo "WARNING: SSH_PASSWORD is not set."
    echo "Set the root password manually using:"
    echo "passwd"
fi

# Restart SSH without systemctl
service ssh restart || /usr/sbin/sshd

# Verify SSH is listening
if ss -lntp | grep -q ':22 '; then
    echo "SSH is running and listening on port 22."
else
    echo "WARNING: SSH does not appear to be listening on port 22."
fi

echo "If there was no error above, set the password using 'passwd' if required."
```

If `SSH_PASSWORD` is not configured, set the root password manually:

```bash
passwd
```

Enter the password when prompted.

---

## 2. Login to the RunPod Using SSH

From your local Windows machine:

```powershell
ssh -p 22072 root@194.68.245.123
```

Enter the root password configured in the previous step.

---

## 3. Check NVIDIA GPU and PyTorch

After logging into the RunPod, check the GPU:

```bash
nvidia-smi
```

Then check PyTorch and CUDA:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

You should see information similar to:

```text
PyTorch: 2.x.x
CUDA available: True
PyTorch CUDA: 12.x
GPU: NVIDIA L4
```

The exact versions may vary depending on the RunPod image.

---

## 4. Go to the Project Directory

```bash
cd /workspace/QMSInspection-POC
```

Verify the project files:

```bash
ls -lah
```

---

## 5. Install Python Dependencies

Install all dependencies from `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

The `requirements.txt` should contain:

```text
# Core
ultralytics>=8.2.0
opencv-python>=4.9.0
numpy==1.26.4
pyyaml>=6.0
pillow>=10.0.0
matplotlib>=3.8.0

# ONNX / export tooling
onnx>=1.16.0
onnxsim>=0.4.36
onnxruntime-gpu>=1.18.0

# TensorRT
tensorrt>=10.0.0
pycuda>=2024.1

# Note: torch/torchvision are installed separately with the
# correct CUDA-specific version.
```

### Why NumPy 1.26.4?

The RunPod PyTorch environment may use a PyTorch build compiled against NumPy 1.x. Using NumPy 2.x can result in:

```text
RuntimeError: Numpy is not available
```

Therefore, keep:

```text
numpy==1.26.4
```

Verify NumPy:

```bash
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
```

Expected:

```text
NumPy: 1.26.4
```

Verify all installed packages:

```bash
python3 -m pip list
```

---

## 6. Verify Ultralytics

Run:

```bash
python3 -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
```

Expected output:

```text
Ultralytics: 8.x.x
```

---

## 7. Run Inference

Run the crack segmentation model against the input folder:

```bash
python3 setup_pretrained_yolo_model.py --source /workspace/QMSInspection-POC/input/defect-free/
```

You can also specify a confidence threshold:

```bash
python3 setup_pretrained_yolo_model.py --source /workspace/QMSInspection-POC/input/defect-free/ --conf 0.5
```

The script will automatically download the pretrained model if it does not already exist:

```text
crack_seg_yolov8n.pt
```

The default output directory is:

```text
/workspace/QMSInspection-POC/outputs/
```

Ultralytics creates a prediction directory such as:

```text
/workspace/QMSInspection-POC/outputs/predict/
```

or:

```text
/workspace/QMSInspection-POC/outputs/predict-2/
```

depending on how many times inference has been run.

---

## 8. Verify the Inference Output

List the output directories:

```bash
ls -lah /workspace/QMSInspection-POC/outputs/
```

For example:

```text
predict/
predict-2/
```

Check the latest prediction folder:

```bash
ls -lah /workspace/QMSInspection-POC/outputs/predict-2/
```

The generated images will be inside this directory.

---

## 9. Download Inference Results to Local Machine

From your **local Windows PowerShell**, download the prediction results:

```powershell
scp -P 22072 -r root@194.68.245.123:/workspace/QMSInspection-POC/outputs/predict-2 C:\workspace\QMSInspection-POC\outputs\
```

The results will be available locally at:

```text
C:\workspace\QMSInspection-POC\outputs\predict-2\
```

If the prediction directory is `predict` instead:

```powershell
scp -P 22072 -r root@194.68.245.123:/workspace/QMSInspection-POC/outputs/predict C:\workspace\QMSInspection-POC\outputs\
```

---

## Complete Workflow

For a fresh RunPod, the overall process is:

```text
1. Configure SSH
       ↓
2. Generate SSH host keys
       ↓
3. Validate SSH configuration
       ↓
4. Set root password
       ↓
5. Start SSH
       ↓
6. SSH into RunPod
       ↓
7. Check nvidia-smi
       ↓
8. Check PyTorch + CUDA
       ↓
9. cd /workspace/QMSInspection-POC
       ↓
10. Install requirements.txt
       ↓
11. Verify NumPy / PyTorch / Ultralytics
       ↓
12. Run inference
       ↓
13. Verify outputs
       ↓
14. Download outputs to local machine
```

## Quick Commands

### SSH

```powershell
ssh -p 22072 root@194.68.245.123
```

### GPU

```bash
nvidia-smi
```

### PyTorch + CUDA

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Project

```bash
cd /workspace/QMSInspection-POC
```

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Verify packages

```bash
python3 -m pip list
```

### Run inference

```bash
python3 setup_pretrained_yolo_model.py --source /workspace/QMSInspection-POC/input/defect-free/
```

### Check output

```bash
ls -lah /workspace/QMSInspection-POC/outputs/
```

### Download output to Windows

Run from **local PowerShell**:

```powershell
scp -P 22072 -r root@194.68.245.123:/workspace/QMSInspection-POC/outputs/predict-2 C:\workspace\QMSInspection-POC\outputs\
```

## 10. Create a Reusable RunPod Image

Once the following are successfully verified:

```text
✓ SSH working
✓ NVIDIA GPU detected
✓ PyTorch working
✓ CUDA available
✓ NumPy 1.26.4
✓ Ultralytics installed
✓ requirements.txt installed
✓ Model downloaded
✓ Inference completed successfully
```

Create a **custom RunPod image** from this configured Pod.

The image can then be reused to create new Pods without reinstalling the Python dependencies every time.

Keep frequently changing data such as inspection images outside the image where possible:

```text
RunPod Image
├── Python
├── CUDA
├── PyTorch
├── Ultralytics
└── Required packages

QMSInspection-POC
├── setup_pretrained_yolo_model.py
├── requirements.txt
├── crack_seg_yolov8n.pt
├── input/
└── outputs/
```
