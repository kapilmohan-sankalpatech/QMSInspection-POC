# Enable password authentication
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Allow root login with password
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config

# Verify SSH configuration
sshd -t

# Set root password
echo "root:${SSH_PASSWORD}" | chpasswd

# Restart SSH without systemctl
service ssh restart



then login to using ssh

check

ssh -p 22072 root@194.68.245.123 "nvidia-smi && python3 -c \"import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\""
