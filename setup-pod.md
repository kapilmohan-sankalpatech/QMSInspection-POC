# RunPod Setup and Inference

## 1. Configure SSH on RunPod

Run the following inside the RunPod Pod:

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

# Restart SSH
service ssh restart || /usr/sbin/sshd

# Verify SSH is listening
if ss -lntp | grep -q ':22 '; then
    echo "SSH is running and listening on port 22."
else
    echo "WARNING: SSH does not appear to be listening on port 22."
fi

echo "SSH setup completed."
```

If `SSH_PASSWORD` was not provided, set the password manually:

```bash
passwd
```

---

## 2. Connect to the RunPod Pod

From your Windows machine:

```bash
ssh -p 22170 root@194.68.245.123
```

---

## 3. Check NVIDIA GPU

Inside the RunPod Pod:

```bash
nvidia-smi
```

You should see the NVIDIA GPU information.

---

## 4. Check PyTorch and CUDA

Run:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output should look similar to:

```text
PyTorch: 2.x.x
CUDA available: True
PyTorch CUDA: 12.x
GPU: NVIDIA L4
```

The exact CUDA/PyTorch versions depend on the RunPod image.

---

## 5. Go to the Project

```bash
cd /workspace/QMSInspection-POC
```

Check the files:

```bash
ls -la
```

---

## 6. Install Python Dependencies

Install everything from `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

### Important: NumPy version

The project requires:

```text
numpy==1.26.4
```

This avoids compatibility issues with packages compiled against NumPy 1.x.

Verify:

```bash
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
```

Expected:

```text
NumPy: 1.26.4
```

---

## 7. Verify Python Environment

Run:

```bash
python3 -c "import numpy; import torch; import ultralytics; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__)"
```

---

## 8. Run YOLO Inference

Input folder:

```text
/workspace/QMSInspection-POC/input/defect-free/
```

Run:

```bash
python3 setup_pretrained_yolo_model.py --source /workspace/QMSInspection-POC/input/defect-free/
```

With a custom confidence threshold:

```bash
python3 setup_pretrained_yolo_model.py --source /workspace/QMSInspection-POC/input/defect-free/ --conf 0.5
```

The model:

```text
crack_seg_yolov8n.pt
```

will be downloaded automatically if it does not already exist.

---

## 9. Check Inference Output

The configured output directory is:

```text
/workspace/QMSInspection-POC/outputs
```

Ultralytics creates a prediction directory such as:

```text
/workspace/QMSInspection-POC/outputs/predict-2
```

Check:

```bash
ls -lah /workspace/QMSInspection-POC/outputs/
```

Then:

```bash
ls -lah /workspace/QMSInspection-POC/outputs/predict-2/
```

---

## 10. Download Results to Windows

From your **Windows machine**, run:

```powershell
scp -P 22170 -r root@194.68.245.123:/workspace/QMSInspection-POC/outputs/predict-2 C:\workspace\QMSInspection-POC\outputs\
```

This copies the complete `predict-2` directory from RunPod to your local machine.

If the output directory is different, replace `predict-2` with the actual directory name.

---

# Complete Workflow

### On RunPod

```bash
cd /workspace/QMSInspection-POC
```

```bash
nvidia-smi
```

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 setup_pretrained_yolo_model.py --source /workspace/QMSInspection-POC/input/defect-free/
```

Check:

```bash
ls -lah /workspace/QMSInspection-POC/outputs/
```

### On Windows

```powershell
scp -P 22170 -r root@194.68.245.123:/workspace/QMSInspection-POC/outputs/predict-2 C:\workspace\QMSInspection-POC\outputs\
```

---

# Quick Commands

### SSH

```bash
ssh -p 22170 root@194.68.245.123
```

### GPU

```bash
nvidia-smi
```

### Project

```bash
cd /workspace/QMSInspection-POC
```

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Check NumPy

```bash
python3 -c "import numpy; print(numpy.__version__)"
```

### Check PyTorch/CUDA

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Run inference

```bash
python3 setup_pretrained_yolo_model.py --source /workspace/QMSInspection-POC/input/defect-free/
```

### Download results

```powershell
scp -P 22170 -r root@194.68.245.123:/workspace/QMSInspection-POC/outputs/predict-2 C:\workspace\QMSInspection-POC\outputs\
```

---

# Optional: Create a Reusable RunPod Image

Once the Pod is fully configured and inference is working:

1. Stop making changes to the environment.
2. Create a custom image from the configured Pod using RunPod's image/template workflow.
3. The image should contain:

   * CUDA environment
   * Python
   * PyTorch
   * Ultralytics
   * ONNX dependencies
   * TensorRT dependencies
   * Project dependencies
4. Keep frequently changing inspection images outside the image.
5. New Pods created from the image can then start with the required environment already installed.

This avoids reinstalling all Python dependencies every time a new Pod is created.
