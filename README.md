  
# Spiking Hypergraph Neural Networks in Rust

> ⚠️ **PROJECT UNDER CONSTRUCTION** ⚠️
>
> **Hypergraph Modularization in Progress**: We are currently refactoring this project to separate hypergraph-specific functionality from generic spiking neural network primitives. This will enable users to employ any data structure (traditional graphs, matrices, sparse matrices, etc.) for neural connectivity while preserving all neuromorphic capabilities.
>
> **Current Status**:
> - ✅ Core neuromorphic primitives (neurons, spikes, plasticity, encoding) are already data-structure agnostic
> - 🚧 Creating connectivity abstraction layer to support multiple data structures
> - 🚧 Implementing traditional graph, matrix, and sparse matrix connectivity options
> - 📋 See [`HYPERGRAPH_MODULARIZATION_PLAN.md`](HYPERGRAPH_MODULARIZATION_PLAN.md) for detailed implementation plan
>
> **Impact**: Existing hypergraph functionality will be preserved with backward compatibility. New users will benefit from choosing optimal data structures for their specific use cases.

## Is This Actually Useful?

Theoretically Promising: Hypergraphs naturally encode multi-neuron interactions, which SNNs badly need.

Training Flexibility: Allows learning to occur at the level of groups, not individuals—better for gradient-free or biologically plausible methods.

 More Robust Representations: Hyperpaths can encode complex motifs, reduce sensitivity to individual spike timings.

⚠️ Challenges:
	•	Building efficient hardware for this is very hard (hypergraph inference is more memory-intensive).
	•	Designing meaningful hypergraph topologies from real data is non-trivial—need a way to learn or regularize structure.
	•	Theoretical frameworks are nascent—not many tools or convergence guarantees.


## Overview

SHGNN is a spiking neural networks with the substrate bein of hypergraph structures. Built in Rust to achieve morelike neuromorphic efficiency through event-driven computation, zero-cost abstractions, and memory-safe spike routing.


###  Example

```rust
use shgnn::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple spiking network
    let mut network = HypergraphNetwork::builder()
        .neurons(100)
        .hyperedges(200)
        .plasticity(PlasticityType::STDP)
        .build()?;
    
    // Add neurons with different types
    for i in 0..50 {
        network.add_neuron(NeuronType::LIF, LIFConfig::default())?;
    }
    for i in 50..100 {
        network.add_neuron(NeuronType::Izhikevich, IzhikevichConfig::default())?;
    }
    
    // Create hyperedge connections
    let hyperedge = HyperedgeBuilder::new()
        .inputs(&[NeuronId(0), NeuronId(1), NeuronId(2)])
        .outputs(&[NeuronId(50), NeuronId(51)])
        .pattern(ConnectionPattern::ManyToMany)
        .threshold(1.5)
        .build(HyperedgeId(0))?;
    
    network.add_hyperedge(hyperedge);
    
    // Simulate spike processing
    let input_spikes = vec![
        Spike { source: NeuronId(0), timestamp: 1000, amplitude: 1.0 },
        Spike { source: NeuronId(1), timestamp: 1100, amplitude: 0.8 },
    ];
    
    let output_spikes = network.process_spikes(input_spikes).await?;
    println!("Generated {} output spikes", output_spikes.len());
    
    Ok(())
}
```

## 📦 Crate Structure

- **`shgnn-core`** - Core neuromorphic primitives and data structures
- **`shgnn-async`** - Asynchronous spike processing and event-driven computation
- **`shgnn-wasm`** - WebAssembly bindings for browser deployment
- **`shgnn-embedded`** - No-std support for microcontroller deployment
- **`shgnn-ffi`** - Foreign function interface for hardware acceleration
- **`shgnn-python`** - Python bindings for integration with existing workflows
- **`shgnn-bench`** - Comprehensive benchmarking suite

## 🎯 Use Cases

### Real-Time Robotics
```rust
use shgnn::robotics::*;

let controller = NeuromorphicController::new(ControllerConfig {
    sensory_neurons: 200,
    motor_neurons: 60,
    control_frequency: 1000.0, // 1kHz
})?;

// Process sensory input and generate motor commands
let motor_commands = controller.control_step(sensory_data).await?;
```

### Edge AI Inference
```rust
use shgnn::embedded::*;

let classifier = EmbeddedSHGNN::new(EmbeddedConfig {
    max_neurons: 1000,
    quantization: QuantizationType::Int8,
    power_budget: PowerBudget::UltraLowPower,
})?;

let prediction = classifier.classify(sensor_data)?;
```

### Brain-Computer Interfaces
```rust
use shgnn::bci::*;

let bci = NeuromorphicBCI::new(BCIConfig {
    channels: 64,
    sampling_rate: 1000.0,
    spike_detection_threshold: -45.0,
})?;

let intention = bci.decode_intention(neural_signals).await?;
```

## 🏗️ Architecture


```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Sensory Input   │───▶│ Spike Encoding   │───▶│ Hypergraph      │
│ Processing      │    │ Pipeline         │    │ Network         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Motor Output    │◀───│ Spike Decoding   │◀───│ STDP Learning   │
│ Control         │    │ Pipeline         │    │ Mechanisms      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

1. **Neuron Models**: LIF, AdEx, Izhikevich, Hodgkin-Huxley
2. **Hyperedge Patterns**: Convergent, Divergent, Associative, Competitive
3. **Plasticity Rules**: STDP, Homeostatic, BCM, Oja
4. **Encoding Schemes**: Rate-based, Temporal, Population, Phase


- **[Basic Network](examples/basic-network/)** - Simple SHGNN setup
- **[Robotics Control](examples/robotics-control/)** - Real-time robot control
- **[Edge AI](examples/edge-ai/)** - Embedded pattern recognition
- **[BCI Demo](examples/bci-demo/)** - Brain-computer interface



### Development Setup

```bash
# Clone the repository
git clone https://github.com/shgnn/shgnn.git
cd shgnn

# Install Rust toolchain
rustup install stable
rustup component add rustfmt clippy

# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace
```

