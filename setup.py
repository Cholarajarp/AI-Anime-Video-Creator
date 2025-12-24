#!/usr/bin/env python3
"""
AI Video Creator - Setup Script
================================
Helps set up the development environment and download required models.
"""

import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent


def main():
    """Main setup function."""
    print_banner()

    print("This script will help you set up the AI Video Creator.\n")

    # Step 1: Check Python version
    check_python()

    # Step 2: Create virtual environment
    if prompt_yes_no("Create a virtual environment?"):
        create_venv()

    # Step 3: Install dependencies
    if prompt_yes_no("Install Python dependencies?"):
        install_dependencies()

    # Step 4: Create directories
    create_directories()

    # Step 5: Create .env file
    if prompt_yes_no("Create .env configuration file?"):
        create_env_file()

    # Step 6: Download models
    if prompt_yes_no("Download recommended AI models? (This may take a while)"):
        download_models()

    # Step 7: Check ComfyUI
    print("\n" + "="*60)
    print("ComfyUI Setup")
    print("="*60)
    print("""
ComfyUI is required for video generation. You need to:

1. Clone ComfyUI:
   git clone https://github.com/comfyanonymous/ComfyUI.git

2. Install ComfyUI dependencies:
   cd ComfyUI
   pip install -r requirements.txt

3. Install AnimateDiff-Evolved custom node:
   cd custom_nodes
   git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git

4. Start ComfyUI:
   python main.py --listen 0.0.0.0 --port 8188

For more details, see the ComfyUI documentation.
    """)

    # Done
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("""
To start the application:

1. Make sure ComfyUI is running on port 8188

2. Activate your virtual environment:
   Windows: .\\venv\\Scripts\\activate
   Linux/Mac: source venv/bin/activate

3. Run the application:
   python run.py

4. Open in browser:
   http://localhost:7860
    """)


def print_banner():
    """Print setup banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           AI Anime Video Creator - Setup Script               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def check_python():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version < (3, 10):
        print(f"❌ Python 3.10+ is required. You have Python {version.major}.{version.minor}")
        sys.exit(1)
    print(f"Python {version.major}.{version.minor}.{version.micro}\n")


def prompt_yes_no(question: str) -> bool:
    """Prompt for yes/no answer."""
    while True:
        response = input(f"{question} [y/n]: ").strip().lower()
        if response in ["y", "yes"]:
            return True
        if response in ["n", "no"]:
            return False
        print("Please enter 'y' or 'n'")


def create_venv():
    """Create virtual environment."""
    print("\nCreating virtual environment...")
    venv_path = PROJECT_ROOT / "venv"

    if venv_path.exists():
        print("Virtual environment already exists.")
        return

    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print(f"Virtual environment created at {venv_path}")
        print("\nTo activate:")
        if sys.platform == "win32":
            print(f"  .\\venv\\Scripts\\activate")
        else:
            print(f"  source venv/bin/activate")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create virtual environment: {e}")


def install_dependencies():
    """Install Python dependencies."""
    print("\nInstalling dependencies...")

    requirements_file = PROJECT_ROOT / "requirements.txt"
    if not requirements_file.exists():
        print("requirements.txt not found")
        return

    # Determine pip path
    if (PROJECT_ROOT / "venv").exists():
        if sys.platform == "win32":
            pip_path = PROJECT_ROOT / "venv" / "Scripts" / "pip"
        else:
            pip_path = PROJECT_ROOT / "venv" / "bin" / "pip"
    else:
        pip_path = "pip"

    try:
        # Install PyTorch with CUDA
        print("Installing PyTorch with CUDA support...")
        subprocess.run([
            str(pip_path), "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)

        # Install other requirements
        print("Installing other dependencies...")
        subprocess.run([
            str(pip_path), "install", "-r", str(requirements_file)
        ], check=True)

        print("Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")


def create_directories():
    """Create required directories."""
    print("\nCreating directories...")

    directories = [
        "data",
        "outputs",
        "temp",
        "logs",
        "models/checkpoints",
        "models/motion_modules",
        "models/loras",
        "models/vae",
        "workflows"
    ]

    for dir_name in directories:
        dir_path = PROJECT_ROOT / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  {dir_name}/")

    print("Directories created\n")


def create_env_file():
    """Create .env configuration file."""
    print("\nCreating .env file...")

    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / "config" / ".env.example"

    if env_path.exists():
        print(".env already exists. Skipping.")
        return

    if example_path.exists():
        shutil.copy(example_path, env_path)
        print(f"Created .env from template")
        print("   Edit .env to customize your configuration")
    else:
        # Create minimal .env
        env_content = """# AI Video Creator Configuration
HOST=0.0.0.0
PORT=7860
COMFYUI_HOST=127.0.0.1
COMFYUI_PORT=8188
DEBUG=false
"""
        env_path.write_text(env_content)
        print(f"Created minimal .env file")


def download_models():
    """Download recommended AI models."""
    print("\nDownloading AI models...")
    print("This may take a while depending on your internet connection.\n")

    models_to_download = [
        {
            "name": "AnimateDiff Motion Module v2",
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt",
            "path": PROJECT_ROOT / "models" / "motion_modules" / "mm_sd_v15_v2.ckpt",
            "size": "1.8 GB"
        }
    ]

    for model in models_to_download:
        if model["path"].exists():
            print(f"  ⏭️ {model['name']} already exists")
            continue

        print(f"  ⬇️ Downloading {model['name']} ({model['size']})...")
        try:
            download_file(model["url"], model["path"])
            print(f"  ✅ {model['name']} downloaded")
        except Exception as e:
            print(f"  ❌ Failed to download {model['name']}: {e}")

    print("""
Note: You also need to download an anime checkpoint model.
Recommended options:
  - DreamShaper 8: https://huggingface.co/Lykon/dreamshaper-8
  - MeinaMix V11: https://civitai.com/models/7240
  - Counterfeit V3: https://huggingface.co/gsdf/Counterfeit-V3.0

Place the .safetensors file in: models/checkpoints/
    """)


def download_file(url: str, destination: Path, chunk_size: int = 8192):
    """Download a file with progress indicator."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(destination, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r    Progress: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)

        print()  # New line after progress


if __name__ == "__main__":
    main()

