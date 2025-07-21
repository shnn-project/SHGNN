# SHNN Documentation

Welcome to the comprehensive documentation for the Spiking Hypergraph Neural Network (SHNN) library - a cutting-edge neuromorphic computing framework designed for high-performance, biologically-inspired artificial intelligence.

## Table of Contents

- [Architecture Overview](architecture/README.md)
- [Getting Started](getting-started/README.md)
- [User Guide](user-guide/README.md)
- [API Reference](api/README.md)
- [Technical Specifications](technical/README.md)
- [Examples](examples/README.md)
- [Tutorials](tutorials/README.md)
- [Hardware Support](hardware/README.md)
- [Performance Optimization](performance/README.md)
- [Contributing](../CONTRIBUTING.md)

## What is SHNN?

SHNN (Spiking Hypergraph Neural Network) is a comprehensive neuromorphic computing library that combines the biological realism of spiking neural networks with the mathematical power of hypergraph structures. It provides:

### 🧠 **Neuromorphic Computing**
- Biologically-inspired spiking neural networks
- Event-driven computation with temporal dynamics
- Multiple neuron models (LIF, AdEx, Izhikevich)
- Spike-timing dependent plasticity (STDP)

### 🔗 **Hypergraph Architecture**
- Multi-synaptic connections beyond pairwise edges
- Complex network topologies and dynamics
- Efficient hypergraph data structures
- Advanced connectivity patterns

### 🚀 **High Performance**
- Zero-cost abstractions in Rust
- Hardware acceleration (CUDA, OpenCL, FPGA)
- Neuromorphic chip support (Intel Loihi, SpiNNaker)
- RRAM and emerging memory technologies

### 🌐 **Multi-Platform Support**
- Standard systems (Linux, macOS, Windows)
- WebAssembly for browser deployment
- Embedded systems (ARM Cortex-M, RISC-V)
- Real-time and deterministic processing

### 🐍 **Python Integration**
- Comprehensive Python bindings
- NumPy array integration
- Matplotlib visualization support
- Jupyter notebook compatibility

## Key Features

### Neuromorphic Primitives
- **Temporal Dynamics**: Event-driven computation with precise spike timing
- **Biological Realism**: Membrane potential dynamics and refractory periods
- **Plasticity**: Spike-timing dependent learning and homeostatic mechanisms
- **Encoding**: Multiple spike encoding schemes (rate, temporal, population)

### Advanced Architecture
- **Hypergraph Networks**: Multi-way connections and complex topologies
- **Modular Design**: Composable components for custom architectures
- **Type Safety**: Compile-time guarantees for network correctness
- **Memory Efficiency**: Optimized data structures for large-scale networks

### Hardware Acceleration
- **GPU Computing**: CUDA and OpenCL implementations
- **FPGA Deployment**: High-throughput custom logic
- **Neuromorphic Chips**: Native support for specialized hardware
- **Edge Computing**: Efficient embedded implementations

### Developer Experience
- **Rich API**: Intuitive interfaces in Rust and Python
- **Comprehensive Testing**: Extensive test suites and benchmarks
- **Documentation**: Detailed guides and examples
- **Tooling**: Profiling, visualization, and debugging tools

## Quick Start

### Rust
```rust
use shnn_core::prelude::*;

// Create a simple spiking neural network
let mut network = Network::builder()
    .with_neurons(1000)
    .with_connectivity(0.1)
    .build()?;

// Add plasticity
network.add_plasticity_rule(STDPRule::default())?;

// Process spikes
let input_spikes = generate_poisson_spikes(100.0, 0.1)?;
let output_spikes = network.process(input_spikes)?;
```

### Python
```python
import shnn

# Create network with hardware acceleration
network = shnn.Network(num_neurons=1000, connectivity=0.1)
accelerator = shnn.AcceleratorRegistry.get_best_accelerator()
network.deploy_to_hardware(accelerator.id)

# Generate and process spikes
spikes = shnn.generate_poisson_spikes(rate=100, duration=0.1)
output = network.process_spikes(spikes)

# Visualize results
shnn.plot_raster(output, title="Network Activity")
```

### WebAssembly
```javascript
import init, { Network, generate_poisson_spikes } from 'shnn-wasm';

async function run() {
    await init();
    
    const network = new Network(1000, 0.1);
    const spikes = generate_poisson_spikes(100, 0.1);
    const output = network.process_spikes(spikes);
    
    console.log(`Processed ${output.length} output spikes`);
}
```

## Architecture Highlights

### Modular Crate Structure
```
shnn/
├── shnn-core/          # Core neuromorphic primitives
├── shnn-async/         # Asynchronous processing
├── shnn-wasm/          # WebAssembly bindings  
├── shnn-embedded/      # Embedded/no-std support
├── shnn-ffi/           # Hardware acceleration
├── shnn-python/        # Python bindings
└── shnn-bench/         # Benchmarking suite
```

### Hardware Support Matrix

| Hardware | Status | Performance | Features |
|----------|--------|-------------|----------|
| CPU | ✅ Full | Baseline | All features |
| CUDA GPU | ✅ Full | 10-100x | Parallel processing |
| OpenCL | ✅ Full | 5-50x | Cross-platform |
| Intel Loihi | ✅ Beta | 1000x | Native neuromorphic |
| SpiNNaker | ✅ Beta | 100x | Massively parallel |
| FPGA | 🚧 Alpha | 50-500x | Custom logic |
| RRAM | 🚧 Experimental | TBD | In-memory compute |

### Performance Characteristics

- **Latency**: Sub-microsecond spike processing
- **Throughput**: Millions of spikes per second
- **Scalability**: Networks with millions of neurons
- **Memory**: Efficient sparse representations
- **Power**: Optimized for neuromorphic hardware

## Documentation Structure

### For Beginners
1. [Getting Started Guide](getting-started/README.md)
2. [Basic Concepts](getting-started/concepts.md)
3. [First Network](getting-started/first-network.md)
4. [Simple Examples](examples/basic/README.md)

### For Developers
1. [Architecture Overview](architecture/README.md)
2. [API Reference](api/README.md)
3. [Advanced Features](user-guide/advanced/README.md)
4. [Performance Optimization](performance/README.md)

### For Researchers
1. [Technical Specifications](technical/README.md)
2. [Neuromorphic Principles](technical/neuromorphic.md)
3. [Hypergraph Theory](technical/hypergraphs.md)
4. [Benchmarking](technical/benchmarks.md)

### For Hardware Developers
1. [Hardware Acceleration](hardware/README.md)
2. [FFI Interface](hardware/ffi.md)
3. [Custom Accelerators](hardware/custom.md)
4. [Embedded Deployment](hardware/embedded.md)

## Community and Support

- **GitHub**: [github.com/shnn-project/shnn](https://github.com/shnn-project/shnn)
- **Documentation**: [docs.shnn-project.org](https://docs.shnn-project.org)
- **Discussions**: [github.com/shnn-project/shnn/discussions](https://github.com/shnn-project/shnn/discussions)
- **Issues**: [github.com/shnn-project/shnn/issues](https://github.com/shnn-project/shnn/issues)

## License

SHNN is dual-licensed under MIT and Apache 2.0 licenses. See [LICENSE-MIT](../LICENSE-MIT) and [LICENSE-APACHE](../LICENSE-APACHE) for details.

## Citation

If you use SHNN in your research, please cite:

```bibtex
@software{shnn2024,
  title = {SHNN: Spiking Hypergraph Neural Networks},
  author = {SHNN Team},
  year = {2024},
  url = {https://github.com/shnn-project/shnn},
  version = {0.1.0}
}
```

---

*This documentation is continuously updated. For the latest information, please visit our [GitHub repository](https://github.com/shnn-project/shnn).*