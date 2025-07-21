"""
SHNN - Spiking Hypergraph Neural Network Library

High-performance Python bindings for neuromorphic computing and spiking neural networks.
Built with Rust for maximum performance and Python for ease of use.
"""

# Import the compiled Rust extension
try:
    from ._shnn_python import *
    __version__ = "0.1.0"
except ImportError as e:
    raise ImportError(
        "Failed to import SHNN Rust extension. "
        "Please ensure the package is properly installed. "
        f"Error: {e}"
    ) from e

# Import additional Python utilities if they exist
try:
    from .utils import *
except ImportError:
    pass

try:
    from .visualization import *
except ImportError:
    pass

# Package metadata
__author__ = "SHNN Team"
__email__ = "team@shnn-project.org"
__license__ = "MIT OR Apache-2.0"
__description__ = "High-performance spiking neural networks with Rust backend"

# Version info
VERSION_INFO = {
    "version": __version__,
    "rust_backend": True,
    "features": {
        "cuda": False,  # Will be determined at runtime
        "opencl": False,
        "neuromorphic": False,
    }
}

def info():
    """Print SHNN package information."""
    print(f"SHNN v{__version__}")
    print(f"High-performance spiking neural networks")
    print(f"Rust backend: {VERSION_INFO['rust_backend']}")
    print(f"Features: {VERSION_INFO['features']}")

def test_installation():
    """Test basic SHNN functionality."""
    try:
        # Test basic imports
        network = Network(num_neurons=10, connectivity=0.1)
        neuron = LIFNeuron()
        encoder = PoissonEncoder(max_rate=100.0)
        
        print("✅ SHNN installation test passed!")
        print(f"   Network: {network}")
        print(f"   Neuron: {neuron}")
        print(f"   Encoder: {encoder}")
        return True
        
    except Exception as e:
        print(f"❌ SHNN installation test failed: {e}")
        return False

# Convenience functions
def create_lif_network(input_size, hidden_size, output_size):
    """Create a simple LIF network."""
    layer_sizes = [input_size, hidden_size, output_size]
    return Network.feedforward(layer_sizes)

def create_poisson_spikes(rate, duration, neuron_id=0, seed=None):
    """Create Poisson spike train."""
    encoder = PoissonEncoder(max_rate=rate, seed=seed)
    return encoder.encode(1.0, duration, neuron_id)