# Getting Started with SHNN

Welcome to SHNN! This guide will help you get up and running with the Spiking Hypergraph Neural Network library quickly and efficiently.

## Table of Contents

- [Installation](#installation)
- [Basic Concepts](#basic-concepts)
- [Your First Network](#your-first-network)
- [Key Features](#key-features)
- [Next Steps](#next-steps)

## Installation

### Rust

Add SHNN to your `Cargo.toml`:

```toml
[dependencies]
shnn-core = "0.1.0"

# Optional: for async processing
shnn-async = { version = "0.1.0", optional = true }

# Optional: for hardware acceleration
shnn-ffi = { version = "0.1.0", optional = true }

# Optional: for WebAssembly
shnn-wasm = { version = "0.1.0", optional = true }

# Optional: for embedded systems
shnn-embedded = { version = "0.1.0", optional = true }
```

### Python

Install via pip:

```bash
pip install shnn-python
```

Or with conda:

```bash
conda install -c conda-forge shnn-python
```

For development installation:

```bash
git clone https://github.com/shnn-project/shnn
cd shnn
pip install -e ./crates/shnn-python
```

### WebAssembly

Install via npm:

```bash
npm install shnn-wasm
```

Or include in your HTML:

```html
<script type="module">
  import init, { Network } from 'https://unpkg.com/shnn-wasm/shnn_wasm.js';
  // Your code here
</script>
```

### System Requirements

**Minimum Requirements:**
- Rust 1.70+ (for Rust API)
- Python 3.8+ (for Python API)
- Node.js 16+ (for WebAssembly)

**Recommended:**
- Rust 1.75+
- Python 3.10+
- Node.js 18+
- CUDA 11.0+ (for GPU acceleration)
- OpenCL 2.0+ (for OpenCL acceleration)

## Basic Concepts

### Spiking Neural Networks

Unlike traditional artificial neural networks that use continuous values, spiking neural networks (SNNs) communicate through discrete spike events, similar to biological neurons.

**Key Properties:**
- **Temporal Dynamics**: Time is explicitly modeled
- **Event-Driven**: Computation only occurs when spikes happen
- **Biological Realism**: Closer to how real brains work
- **Energy Efficiency**: Sparse activation patterns

### Hypergraph Networks

Traditional neural networks use pairwise connections (edges), while SHNN uses hypergraphs that allow multi-way connections (hyperedges).

**Advantages:**
- **Higher-Order Interactions**: Capture complex relationships
- **Flexible Topology**: More expressive network structures
- **Mathematical Foundation**: Well-defined algebraic operations
- **Biological Accuracy**: Better model of neural circuits

### Core Components

1. **Neurons**: Processing units with membrane dynamics
2. **Spikes**: Discrete events carrying information
3. **Hyperedges**: Multi-way connections between neurons
4. **Plasticity**: Learning rules that modify connections
5. **Encoders**: Convert data to spike patterns

## Your First Network

### Rust Example

```rust
use shnn_core::prelude::*;
use anyhow::Result;

fn main() -> Result<()> {
    // Create a simple network with 100 neurons
    let mut network = Network::builder()
        .with_neurons(100)
        .with_connectivity(0.1)  // 10% connection probability
        .with_topology(Topology::Random)
        .build()?;

    // Add different neuron types
    network.add_neuron_population(
        PopulationConfig::new()
            .with_size(80)
            .with_neuron_type(NeuronType::LIF)
            .with_parameters(LIFParameters::default())
    )?;

    network.add_neuron_population(
        PopulationConfig::new()
            .with_size(20)
            .with_neuron_type(NeuronType::Izhikevich)
            .with_parameters(IzhikevichParameters::regular_spiking())
    )?;

    // Add plasticity
    network.add_plasticity_rule(
        STDPRule::new()
            .with_learning_rate(0.01)
            .with_time_constants(20.0, 20.0)
    )?;

    // Generate input spikes
    let input_spikes = generate_poisson_spikes(
        10,      // 10 input neurons
        100.0,   // 100 Hz average rate
        0.1,     // 100ms duration
        Some(42) // Random seed
    )?;

    // Process the input
    let output_spikes = network.process(input_spikes)?;

    println!("Generated {} output spikes", output_spikes.len());
    
    // Analyze results
    let firing_rates = calculate_firing_rates(&output_spikes, 0.1)?;
    println!("Average firing rate: {:.2} Hz", firing_rates.mean());

    Ok(())
}
```

### Python Example

```python
import shnn
import numpy as np
import matplotlib.pyplot as plt

# Create network
network = shnn.Network(
    num_neurons=1000,
    connectivity=0.1,
    dt=0.001
)

# Configure neuron populations
network.add_population(
    size=800,
    neuron_type="LIF",
    parameters=shnn.NeuronParameters.lif(tau_m=20.0, v_threshold=-50.0)
)

network.add_population(
    size=200,
    neuron_type="Izhikevich",
    parameters=shnn.NeuronParameters.regular_spiking()
)

# Add plasticity
stdp_rule = shnn.STDPRule(
    a_plus=0.01,
    a_minus=0.01,
    tau_plus=20.0,
    tau_minus=20.0
)
network.add_plasticity_rule(stdp_rule)

# Generate input spikes
input_spikes = shnn.generate_poisson_spikes(
    num_neurons=100,
    rate=50.0,
    duration=0.5,
    seed=42
)

# Process spikes
output_spikes = network.process_spikes(input_spikes)

# Visualize results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Raster plot
shnn.plot_spike_raster(
    output_spikes,
    ax=ax1,
    title="Network Spike Activity"
)

# Firing rate over time
times, rates = shnn.calculate_population_rate(
    output_spikes,
    bin_size=0.01,
    total_neurons=1000
)

ax2.plot(times, rates)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Population Rate (Hz)')
ax2.set_title('Population Firing Rate')

plt.tight_layout()
plt.show()

# Print statistics
print(f"Total output spikes: {len(output_spikes)}")
print(f"Average firing rate: {len(output_spikes) / (1000 * 0.5):.2f} Hz")
```

### WebAssembly Example

```javascript
import init, { 
  Network, 
  generate_poisson_spikes,
  plot_spike_raster 
} from 'shnn-wasm';

async function runSimulation() {
    // Initialize WASM module
    await init();
    
    // Create network
    const network = new Network(500, 0.1);
    
    // Generate input spikes
    const inputSpikes = generate_poisson_spikes(50, 100.0, 0.1);
    
    // Process spikes
    const outputSpikes = network.process_spikes(inputSpikes);
    
    // Visualize in browser
    const canvas = document.getElementById('spike-plot');
    plot_spike_raster(outputSpikes, canvas);
    
    console.log(`Processed ${outputSpikes.length} spikes`);
}

// Run the simulation
runSimulation().catch(console.error);
```

## Key Features

### 1. Multiple Neuron Models

```rust
// Leaky Integrate-and-Fire
let lif = LIFNeuron::new(LIFParameters {
    tau_m: 20.0,        // Membrane time constant
    v_threshold: -50.0, // Spike threshold
    v_reset: -70.0,     // Reset potential
    v_rest: -70.0,      // Resting potential
    ..Default::default()
});

// Adaptive Exponential
let adex = AdExNeuron::new(AdExParameters {
    tau_m: 20.0,
    delta_t: 2.0,       // Spike sharpness
    v_spike: 0.0,       // Spike detection threshold
    tau_w: 100.0,       // Adaptation time constant
    ..Default::default()
});

// Izhikevich
let izh = IzhikevichNeuron::new(
    IzhikevichParameters::regular_spiking()
);
```

### 2. Flexible Connectivity

```rust
// Random connectivity
let random_net = Network::builder()
    .with_topology(Topology::Random { probability: 0.1 })
    .build()?;

// Small-world network
let small_world = Network::builder()
    .with_topology(Topology::SmallWorld { 
        k: 6, 
        p: 0.1 
    })
    .build()?;

// Custom hypergraph
let mut hypergraph = Hypergraph::new();
hypergraph.add_hyperedge(vec![0, 1, 2], 1.0)?; // 3-way connection
```

### 3. Hardware Acceleration

```python
# Discover available hardware
accelerators = shnn.AcceleratorRegistry.discover_accelerators()
for acc in accelerators:
    print(f"Found {acc.name}: {acc.accelerator_type}")

# Use GPU acceleration
if accelerators:
    best_acc = shnn.AcceleratorRegistry.get_best_accelerator(
        required_neurons=10000,
        prefer_neuromorphic=True
    )
    if best_acc:
        network.deploy_to_hardware(best_acc.id)
        print(f"Deployed to {best_acc.name}")
```

### 4. Real-time Processing

```rust
use shnn_async::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mut runtime = AsyncRuntime::new()?;
    
    // Create streaming network
    let network = runtime.create_network(
        NetworkConfig::default()
    ).await?;
    
    // Process spike stream
    let mut spike_stream = SpikeStream::from_source(input_source);
    
    while let Some(spike_batch) = spike_stream.next().await {
        let output = network.process_batch(spike_batch).await?;
        
        // Handle output in real-time
        handle_output(output).await?;
    }
    
    Ok(())
}
```

### 5. Plasticity and Learning

```rust
// STDP learning
let stdp = STDPRule::new()
    .with_learning_rates(0.01, 0.01)
    .with_time_constants(20.0, 20.0)
    .with_weight_bounds(0.0, 1.0);

// Homeostatic scaling
let homeostatic = HomeostaticRule::new()
    .with_target_rate(10.0)    // Target 10 Hz
    .with_learning_rate(0.001)
    .with_time_constant(1000.0);

// Apply to network
network.add_plasticity_rule(stdp)?;
network.add_plasticity_rule(homeostatic)?;
```

## Next Steps

### Learning Path

1. **Explore Examples**: Check out the [examples directory](../examples/) for more complex use cases

2. **Read the User Guide**: Dive deeper into specific features in the [User Guide](../user-guide/)

3. **API Reference**: Consult the [API Reference](../api/) for detailed function documentation

4. **Hardware Setup**: Configure hardware acceleration following the [Hardware Guide](../hardware/)

5. **Performance Tuning**: Optimize your networks with the [Performance Guide](../performance/)

### Advanced Topics

- **Custom Neuron Models**: Implement your own neuron dynamics
- **Network Architectures**: Design complex multi-layer networks
- **Real-time Applications**: Build responsive neuromorphic systems
- **Distributed Processing**: Scale across multiple machines
- **Hardware Integration**: Interface with neuromorphic chips

### Community Resources

- **GitHub Discussions**: Ask questions and share ideas
- **Examples Repository**: Real-world application examples
- **Benchmarks**: Performance comparisons and optimizations
- **Research Papers**: Academic publications using SHNN

### Troubleshooting

**Common Issues:**

1. **Installation Problems**: Check system requirements and dependencies
2. **Performance Issues**: Enable hardware acceleration or adjust batch sizes
3. **Memory Usage**: Use sparse representations for large networks
4. **Compilation Errors**: Ensure compatible Rust/Python versions

**Getting Help:**

- Check the [FAQ](../faq.md)
- Search [GitHub Issues](https://github.com/shnn-project/shnn/issues)
- Join our [Discord Community](https://discord.gg/shnn)
- Read the [Troubleshooting Guide](../troubleshooting.md)

---

**Congratulations!** You've successfully created your first SHNN network. The combination of biological realism, mathematical rigor, and high performance makes SHNN ideal for cutting-edge neuromorphic computing applications.

Continue exploring to unlock the full potential of spiking hypergraph neural networks! 🧠⚡