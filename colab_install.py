#!/usr/bin/env python3
"""
SHNN Google Colab Installation Script

This script installs the SHNN (Spiking Hypergraph Neural Network) library
in Google Colab by building from Rust source with maturin.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Run a shell command with error handling."""
    print(f"🔄 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            print(f"✅ {result.stdout}")
        if result.stderr and result.returncode != 0:
            print(f"⚠️ {result.stderr}")
            
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        if check:
            raise
        return e

def install_rust():
    """Install Rust toolchain if not present."""
    print("📦 Installing Rust toolchain...")
    
    # Check if Rust is already installed
    try:
        result = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Rust already installed: {result.stdout.strip()}")
            return
    except FileNotFoundError:
        pass
    
    # Install Rust
    install_cmd = """
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    """
    
    run_command(install_cmd, "Installing Rust toolchain")
    
    # Update PATH for current session
    cargo_path = Path.home() / ".cargo" / "bin"
    if cargo_path.exists():
        os.environ["PATH"] = str(cargo_path) + ":" + os.environ.get("PATH", "")

def install_python_dependencies():
    """Install Python dependencies."""
    print("🐍 Installing Python dependencies...")
    
    dependencies = [
        "maturin>=1.0.0",
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "jupyter>=1.0.0",
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}", f"Installing {dep}")

def clone_shnn_repository():
    """Clone the SHNN repository."""
    print("📁 Cloning SHNN repository...")
    
    # Remove existing directory if present
    if os.path.exists("SHNN"):
        run_command("rm -rf SHNN", "Removing existing SHNN directory")
    
    # Clone repository
    run_command(
        "git clone https://github.com/your-username/SHNN.git",
        "Cloning SHNN repository"
    )
    
    if not os.path.exists("SHNN"):
        print("❌ Failed to clone repository. Please check the URL and try again.")
        return False
    
    return True

def build_shnn_python():
    """Build the SHNN Python extension using maturin."""
    print("🔨 Building SHNN Python extension...")
    
    # Change to the Python crate directory
    python_crate_dir = "SHNN/crates/shnn-python"
    
    if not os.path.exists(python_crate_dir):
        print(f"❌ Python crate directory not found: {python_crate_dir}")
        return False
    
    original_dir = os.getcwd()
    
    try:
        os.chdir(python_crate_dir)
        
        # Build and install the Python extension
        run_command(
            "maturin develop --release",
            "Building and installing SHNN Python extension"
        )
        
        return True
        
    finally:
        os.chdir(original_dir)

def verify_installation():
    """Verify that SHNN was installed correctly."""
    print("🔍 Verifying installation...")
    
    try:
        # Test import
        result = subprocess.run([
            sys.executable, "-c", 
            "import shnn; print(f'SHNN version: {shnn.__version__}'); print('✅ Installation successful!')"
        ], capture_output=True, text=True, check=True)
        
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Installation verification failed!")
        print(f"Error: {e.stderr}")
        return False

def setup_colab_environment():
    """Set up additional Colab-specific configurations."""
    print("⚙️ Setting up Colab environment...")
    
    # Create a simple test script
    test_script = '''
import shnn
import numpy as np

# Test basic functionality
print("Testing SHNN basic functionality...")

# Create a simple network
network = shnn.Network(num_neurons=100, connectivity=0.1)
print(f"Network created: {network}")

# Create some test spikes
spikes = shnn.create_poisson_spike_train(rate=50.0, duration=0.1, neuron_id=0)
print(f"Generated {len(spikes)} spikes")

# Test spike encoding
encoder = shnn.PoissonEncoder(max_rate=100.0)
encoded_spikes = encoder.encode(0.5, 0.1, 1)
print(f"Encoded {len(encoded_spikes)} spikes")

print("✅ All basic tests passed!")
'''
    
    with open("test_shnn.py", "w") as f:
        f.write(test_script)
    
    print("📄 Created test_shnn.py for validation")

def main():
    """Main installation function."""
    print("🚀 Starting SHNN installation for Google Colab")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Install Rust
        install_rust()
        
        # Step 2: Install Python dependencies
        install_python_dependencies()
        
        # Step 3: Clone repository
        if not clone_shnn_repository():
            return False
        
        # Step 4: Build Python extension
        if not build_shnn_python():
            return False
        
        # Step 5: Verify installation
        if not verify_installation():
            return False
        
        # Step 6: Setup Colab environment
        setup_colab_environment()
        
        elapsed_time = time.time() - start_time
        print("=" * 60)
        print(f"🎉 SHNN installation completed successfully!")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        print("\n📖 You can now use SHNN in your notebooks:")
        print("   import shnn")
        print("   network = shnn.Network(num_neurons=1000)")
        print("\n🧪 Run 'python test_shnn.py' to validate the installation")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)