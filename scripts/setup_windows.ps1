<#
Windows setup script (PowerShell)
Prerequisites (recommended):
- NVIDIA driver supporting CUDA for your GPU (GTX 1650).
- (Recommended) WSL2 with Ubuntu + NVIDIA WSL support for best compatibility.

This script creates a Python virtual environment, activates it, and installs project dependencies.
If you prefer, use conda for GPU-compatible PyTorch; instructions are printed below.
#>

Write-Host "---- Face-Emotion_Recognition Windows setup ----"

# Check for nvidia-smi
try {
    $n = & nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "NVIDIA GPU detected (nvidia-smi works)."
    } else {
        Write-Host "nvidia-smi not found or driver missing. If you have a GPU, install NVIDIA drivers or use WSL2 with NVIDIA support."
    }
} catch {
    Write-Host "nvidia-smi not available. If you plan to use GPU, ensure drivers are installed or use WSL2." -ForegroundColor Yellow
}

# Create venv
if (-Not (Test-Path -Path .venv)) {
    Write-Host "Creating virtualenv .venv..."
    python -m venv .venv
} else {
    Write-Host ".venv already exists."
}

Write-Host "To activate the virtual environment run:" -ForegroundColor Cyan
Write-Host ".\.venv\Scripts\Activate.ps1"

Write-Host "After activation, run the following (PowerShell):" -ForegroundColor Cyan
Write-Host "pip install --upgrade pip setuptools wheel"
Write-Host "# Install PyTorch compatible with your CUDA. Example for CUDA 11.8 (adjust if you have different CUDA):"
Write-Host "pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio --extra-index-url https://pypi.org/simple"
Write-Host "# Then install the remaining requirements"
Write-Host "pip install -r requirements.txt"

Write-Host "If you prefer conda (recommended for Windows drivers), use these steps:" -ForegroundColor Cyan
Write-Host "1) Install Anaconda/Miniconda"
Write-Host "2) Create env: conda create -n fer python=3.10 -y"
Write-Host "3) Activate: conda activate fer"
Write-Host "4) Install PyTorch (CUDA 11.8): conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia"
Write-Host "5) pip install -r requirements.txt"

Write-Host "Setup script finished. Activate venv then install PyTorch as appropriate for your system." -ForegroundColor Green
