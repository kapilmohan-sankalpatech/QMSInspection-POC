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

# Set root password
if [ -n "${SSH_PASSWORD}" ]; then
    echo "root:${SSH_PASSWORD}" | chpasswd
    echo "Root password configured."
else
    echo "WARNING: SSH_PASSWORD is not set."
    echo "Use 'passwd' manually to set the root password."
fi

# Restart SSH without systemctl
service ssh restart || /usr/sbin/sshd

# Verify SSH is listening
if ss -lntp | grep -q ':22 '; then
    echo "SSH is running and listening on port 22."
else
    echo "WARNING: SSH does not appear to be listening on port 22."
fi







then login to using ssh

check

ssh -p 22072 root@194.68.245.123 "nvidia-smi && python3 -c \"import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\""
