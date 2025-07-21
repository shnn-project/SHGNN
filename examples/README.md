# SHNN Examples

This directory contains comprehensive examples demonstrating various aspects of the SHNN (Spiking Hypergraph Neural Network) library. The examples are organized by complexity and use case, providing a learning path from basic concepts to advanced applications.

## Directory Structure

```
examples/
├── basic/                  # Fundamental concepts and simple networks
├── advanced/              # Complex architectures and applications
├── platforms/             # Platform-specific examples
├── hardware/              # Hardware acceleration examples
├── research/              # Research and experimental applications
├── benchmarks/            # Performance benchmarking examples
└── tutorials/             # Step-by-step learning tutorials
```

## Quick Start Examples

### Basic Network (Rust)
```bash
cd examples/basic/simple-network
cargo run
```

### Python Integration
```bash
cd examples/platforms/python
python simple_snn.py
```

### WebAssembly Demo
```bash
cd examples/platforms/wasm
npm install && npm run dev
```

## Example Categories

### 🟢 Basic Examples
Perfect for newcomers to get familiar with SHNN concepts:

- **[simple-network](basic/simple-network/)**: Create your first spiking neural network
- **[neuron-models](basic/neuron-models/)**: Compare different neuron types
- **[spike-patterns](basic/spike-patterns/)**: Generate and analyze spike patterns
- **[plasticity-demo](basic/plasticity-demo/)**: Basic learning with STDP
- **[visualization](basic/visualization/)**: Plot network activity

### 🟡 Intermediate Examples
Building more complex networks and features:

- **[multilayer-network](intermediate/multilayer-network/)**: Multi-layer architectures
- **[real-time-processing](intermediate/real-time-processing/)**: Streaming spike processing
- **[custom-neurons](intermediate/custom-neurons/)**: Implement custom neuron models
- **[network-topology](intermediate/network-topology/)**: Different connectivity patterns
- **[encoding-decoding](intermediate/encoding-decoding/)**: Data encoding strategies

### 🔴 Advanced Examples
Sophisticated applications and research scenarios:

- **[reservoir-computing](advanced/reservoir-computing/)**: Liquid state machines
- **[temporal-pattern-recognition](advanced/temporal-pattern-recognition/)**: Time-series classification
- **[attention-mechanism](advanced/attention-mechanism/)**: Neuromorphic attention
- **[continual-learning](advanced/continual-learning/)**: Lifelong learning systems
- **[multi-modal-fusion](advanced/multi-modal-fusion/)**: Sensory integration

### 🖥️ Platform Examples
Demonstrate cross-platform capabilities:

- **[python-integration](platforms/python/)**: Complete Python workflows
- **[wasm-browser](platforms/wasm/)**: Browser-based neuromorphic computing
- **[embedded-systems](platforms/embedded/)**: Microcontroller deployment
- **[mobile-apps](platforms/mobile/)**: Smartphone integration
- **[cloud-deployment](platforms/cloud/)**: Scalable cloud processing

### ⚡ Hardware Examples
Showcase hardware acceleration:

- **[cuda-acceleration](hardware/cuda/)**: GPU-accelerated networks
- **[opencl-computing](hardware/opencl/)**: Cross-platform parallel processing
- **[fpga-deployment](hardware/fpga/)**: FPGA implementation
- **[neuromorphic-chips](hardware/neuromorphic/)**: Loihi and SpiNNaker integration
- **[edge-computing](hardware/edge/)**: Optimized edge deployment

### 🔬 Research Examples
Cutting-edge applications and experiments:

- **[adaptive-topology](research/adaptive-topology/)**: Dynamic network reconfiguration
- **[meta-learning](research/meta-learning/)**: Learning to learn
- **[neuromorphic-vision](research/neuromorphic-vision/)**: Event-based vision processing
- **[bio-inspired-control](research/bio-inspired-control/)**: Robotic control systems
- **[quantum-neuromorphic](research/quantum-neuromorphic/)**: Quantum-inspired networks

## Running Examples

### Prerequisites

Make sure you have the required dependencies installed:

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python (for Python examples)
pip install numpy matplotlib jupyter

# Node.js (for WebAssembly examples)
npm install -g wasm-pack

# CUDA (for GPU examples, optional)
# Follow NVIDIA CUDA installation guide
```

### Individual Examples

Each example directory contains:
- `README.md`: Detailed explanation and instructions
- `Cargo.toml` or `pyproject.toml`: Dependencies and configuration
- Source code with extensive comments
- Sample data (where applicable)
- Expected output or visualization

### Batch Execution

Run all basic examples:
```bash
./scripts/run_basic_examples.sh
```

Run platform-specific examples:
```bash
./scripts/run_platform_examples.sh --platform python
./scripts/run_platform_examples.sh --platform wasm
```

Benchmark examples:
```bash
cargo run --bin benchmark-examples --release
```

## Learning Path

### For Beginners
1. Start with [simple-network](basic/simple-network/)
2. Explore [neuron-models](basic/neuron-models/)
3. Try [spike-patterns](basic/spike-patterns/)
4. Learn about [plasticity-demo](basic/plasticity-demo/)

### For Developers
1. Study [multilayer-network](intermediate/multilayer-network/)
2. Implement [custom-neurons](intermediate/custom-neurons/)
3. Explore [real-time-processing](intermediate/real-time-processing/)
4. Try [hardware acceleration](hardware/)

### For Researchers
1. Investigate [reservoir-computing](advanced/reservoir-computing/)
2. Experiment with [adaptive-topology](research/adaptive-topology/)
3. Explore [neuromorphic-vision](research/neuromorphic-vision/)
4. Study [meta-learning](research/meta-learning/)

## Performance Guidelines

### Optimization Tips
- Use hardware acceleration when available
- Choose appropriate batch sizes
- Enable compiler optimizations (`--release`)
- Profile memory usage for large networks
- Consider sparse representations

### Benchmarking
Each example includes performance metrics and expected runtimes on reference hardware. Use the benchmarking examples to compare performance across different configurations.

## Contributing Examples

We welcome contributions of new examples! Please follow these guidelines:

1. **Clear Documentation**: Include comprehensive README with theory and usage
2. **Self-Contained**: Examples should run independently with minimal setup
3. **Educational Value**: Focus on teaching specific concepts or techniques
4. **Performance Awareness**: Include timing and memory usage information
5. **Cross-Platform**: Test on multiple platforms where applicable

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

## Troubleshooting

### Common Issues

**Compilation Errors**
- Ensure Rust toolchain is up to date: `rustup update`
- Check feature flags match your hardware capabilities
- Verify all dependencies are installed

**Runtime Errors**
- Check available system memory for large networks
- Verify hardware drivers (CUDA, OpenCL) are installed
- Ensure input data formats match expected schemas

**Performance Issues**
- Enable hardware acceleration if available
- Use release builds for benchmarking: `cargo run --release`
- Adjust batch sizes based on available memory
- Profile code to identify bottlenecks

### Getting Help

- Check the main [documentation](../docs/)
- Search [GitHub issues](https://github.com/shnn-project/shnn/issues)
- Join our [community discussions](https://github.com/shnn-project/shnn/discussions)
- Review the [FAQ](../docs/faq.md)

## Citation

If you use these examples in your research or projects, please cite:

```bibtex
@misc{shnn_examples2024,
  title = {SHNN Examples: Comprehensive Demonstrations of Spiking Hypergraph Neural Networks},
  author = {SHNN Team},
  year = {2024},
  url = {https://github.com/shnn-project/shnn/tree/main/examples},
  note = {Software examples and tutorials}
}
```

---

**Happy Learning!** These examples demonstrate the power and versatility of SHNN for neuromorphic computing applications. Start with the basics and gradually explore more advanced concepts as you become comfortable with the framework.