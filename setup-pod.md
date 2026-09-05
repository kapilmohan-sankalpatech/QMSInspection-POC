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

## 2. Login to the RunPod using SSH

From your local machine:

```bash
ssh -p 22072 root@194.68.245.123
```

Enter the root password configured in the previous step.

---

## 3. Check NVIDIA GPU and PyTorch

After logging into the RunPod, run:

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

Verify the installed packages:

```bash
python3 -m pip list
```

---

## 6. Run Inference

Run the crack segmentation model against the input folder:

```bash
python3 setup_pretrained_yolo_model.py --source /workspace/QMSInspection-POC/input/defect-free/
```

The script will download the pretrained model automatically if:

```text
crack_seg_yolov8n.pt
```

does not already exist.

The default output directory is:

```text
/workspace/QMSInspection-POC/outputs/
```

---

## 7. Verify the Output

Check the generated output:

```bash
ls -lah /workspace/QMSInspection-POC/outputs/
```

---

## Complete Workflow

For a fresh RunPod, the overall process is:

```text
1. Configure SSH
       ↓
2. Set root password
       ↓
3. SSH into RunPod
       ↓
4. Check nvidia-smi
       ↓
5. Check PyTorch + CUDA
       ↓
6. cd /workspace/QMSInspection-POC
       ↓
7. pip install -r requirements.txt
       ↓
8. pip list
       ↓
9. Run inference
       ↓
10. Verify outputs
```

### Quick Commands

```bash
ssh -p 22072 root@194.68.245.123
```

```bash
nvidia-smi
```

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

```bash
cd /workspace/QMSInspection-POC
```

```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 -m pip list
```

```bash
python3 setup_pretrained_yolo_model.py --source /workspace/QMSInspection-POC/input/defect-free/
```

```bash
ls -lah /workspace/QMSInspection-POC/outputs/
```
