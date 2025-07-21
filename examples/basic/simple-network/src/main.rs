//! Simple Network Example
//!
//! This example demonstrates the basic usage of SHNN by creating a simple
//! spiking neural network with different neuron types, processing spike
//! patterns, and analyzing the results.
//!
//! Usage:
//!   cargo run
//!   cargo run -- --neurons 500 --duration 0.2
//!   cargo run -- --help

use anyhow::Result;
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

// Import SHNN core components
use shnn_core::{
    neuron::{LIFNeuron, IzhikevichNeuron, NeuronParameters, NeuronState},
    spike::{Spike, SpikeTime, SpikeBuffer},
    hypergraph::{Hypergraph, HypergraphBuilder},
    plasticity::{STDPRule, PlasticityRule},
    encoding::{PoissonEncoder, SpikeEncoder},
    time::TimeStep,
    math::Statistics,
};

/// Command line arguments
#[derive(Parser, Debug)]
#[command(name = "simple-network")]
#[command(about = "A simple SHNN example demonstrating basic functionality")]
struct Args {
    /// Number of neurons in the network
    #[arg(short, long, default_value = "100")]
    neurons: usize,

    /// Simulation duration in seconds
    #[arg(short, long, default_value = "0.1")]
    duration: f64,

    /// Input spike rate in Hz
    #[arg(short, long, default_value = "50.0")]
    rate: f32,

    /// Random seed for reproducibility
    #[arg(short, long, default_value = "42")]
    seed: u64,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Save results to file
    #[arg(long)]
    output: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("🧠 SHNN Simple Network Example");
    println!("================================");
    println!("Neurons: {}", args.neurons);
    println!("Duration: {:.3}s", args.duration);
    println!("Input rate: {:.1} Hz", args.rate);
    println!("Seed: {}", args.seed);
    println!();

    // Initialize random number generator
    let mut rng = StdRng::seed_from_u64(args.seed);

    // Step 1: Create the network
    println!("📊 Creating network...");
    let network = create_network(args.neurons, &mut rng)?;
    println!("✓ Network created with {} neurons", args.neurons);

    // Step 2: Generate input spikes
    println!("⚡ Generating input spikes...");
    let input_spikes = generate_input_spikes(
        args.neurons / 10, // 10% input neurons
        args.duration,
        args.rate,
        args.seed,
    )?;
    println!("✓ Generated {} input spikes", input_spikes.len());

    // Step 3: Simulate the network
    println!("🔄 Running simulation...");
    let start_time = Instant::now();
    let output_spikes = simulate_network(&network, &input_spikes, args.duration)?;
    let simulation_time = start_time.elapsed();
    println!("✓ Simulation completed in {:.2?}", simulation_time);

    // Step 4: Analyze results
    println!("📈 Analyzing results...");
    analyze_results(&input_spikes, &output_spikes, args.duration, args.verbose)?;

    // Step 5: Save results if requested
    if let Some(output_path) = args.output {
        save_results(&output_spikes, &output_path)?;
        println!("💾 Results saved to {}", output_path);
    }

    println!("\n🎉 Example completed successfully!");
    Ok(())
}

/// Create a simple spiking neural network
fn create_network(num_neurons: usize, rng: &mut StdRng) -> Result<SimpleNetwork> {
    let mut network = SimpleNetwork::new(num_neurons);

    // Create different types of neurons
    let num_lif = (num_neurons as f64 * 0.8) as usize;  // 80% LIF neurons
    let num_izh = num_neurons - num_lif;                // 20% Izhikevich neurons

    // Add LIF neurons
    for i in 0..num_lif {
        let params = NeuronParameters::lif_default();
        let neuron = LIFNeuron::new(params);
        network.add_neuron(i, Box::new(neuron));
    }

    // Add Izhikevich neurons with different dynamics
    for i in num_lif..num_neurons {
        let params = if rng.gen::<f32>() < 0.5 {
            NeuronParameters::izhikevich_regular_spiking()
        } else {
            NeuronParameters::izhikevich_fast_spiking()
        };
        let neuron = IzhikevichNeuron::new(params);
        network.add_neuron(i, Box::new(neuron));
    }

    // Create random connectivity
    create_connectivity(&mut network, 0.1, rng)?; // 10% connectivity

    // Add STDP plasticity
    let stdp_rule = STDPRule::new()
        .with_learning_rates(0.01, 0.01)
        .with_time_constants(
            TimeStep::from_millis(20.0),
            TimeStep::from_millis(20.0)
        )
        .with_weight_bounds(0.0, 1.0);
    
    network.add_plasticity_rule(Box::new(stdp_rule));

    Ok(network)
}

/// Generate Poisson input spikes
fn generate_input_spikes(
    num_input_neurons: usize,
    duration: f64,
    rate: f32,
    seed: u64,
) -> Result<Vec<Spike>> {
    let mut encoder = PoissonEncoder::new(rate, Some(seed));
    let mut spikes = Vec::new();

    for neuron_id in 0..num_input_neurons {
        let neuron_spikes = encoder.encode(
            1.0, // Full rate
            TimeStep::from_secs_f64(duration),
            neuron_id as u32,
        )?;
        spikes.extend(neuron_spikes);
    }

    // Sort spikes by time
    spikes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    Ok(spikes)
}

/// Simulate the network with input spikes
fn simulate_network(
    network: &SimpleNetwork,
    input_spikes: &[Spike],
    duration: f64,
) -> Result<Vec<Spike>> {
    let dt = TimeStep::from_millis(0.1); // 0.1ms time step
    let total_steps = (duration / dt.as_secs_f64()) as usize;
    
    let mut output_spikes = Vec::new();
    let mut spike_buffer = SpikeBuffer::new(10000);
    let mut current_time = TimeStep::zero();
    
    // Add input spikes to buffer
    for spike in input_spikes {
        spike_buffer.add_spike(spike.clone());
    }

    println!("  Simulating {} time steps...", total_steps);
    
    for step in 0..total_steps {
        if step % (total_steps / 10) == 0 {
            let progress = (step as f64 / total_steps as f64) * 100.0;
            print!("  Progress: {:.1}%\r", progress);
        }

        // Get spikes for current time window
        let window_start = current_time;
        let window_end = current_time + dt;
        
        let current_spikes = spike_buffer.get_spikes_in_window(
            SpikeTime::from_time_step(window_start),
            SpikeTime::from_time_step(window_end),
        )?;

        // Process spikes through network
        let step_output = network.process_step(&current_spikes, dt)?;
        output_spikes.extend(step_output);

        current_time = current_time + dt;
    }
    
    println!("  Progress: 100.0%");

    Ok(output_spikes)
}

/// Analyze simulation results
fn analyze_results(
    input_spikes: &[Spike],
    output_spikes: &[Spike],
    duration: f64,
    verbose: bool,
) -> Result<()> {
    let stats = Statistics::new();

    // Basic statistics
    println!("📊 Simulation Results:");
    println!("  Input spikes:  {}", input_spikes.len());
    println!("  Output spikes: {}", output_spikes.len());
    
    // Calculate firing rates
    let input_rate = input_spikes.len() as f64 / duration;
    let output_rate = output_spikes.len() as f64 / duration;
    
    println!("  Input rate:    {:.1} spikes/s", input_rate);
    println!("  Output rate:   {:.1} spikes/s", output_rate);
    println!("  Amplification: {:.2}x", output_rate / input_rate.max(1.0));

    if verbose {
        // Detailed analysis
        println!("\n🔍 Detailed Analysis:");
        
        // Spike timing statistics
        if !output_spikes.is_empty() {
            let spike_times: Vec<f64> = output_spikes.iter()
                .map(|s| s.time.as_secs_f64())
                .collect();
            
            let mean_time = stats.mean(&spike_times);
            let std_time = stats.std(&spike_times);
            
            println!("  Mean spike time: {:.4}s", mean_time);
            println!("  Std spike time:  {:.4}s", std_time);
        }

        // Neuron activity distribution
        let mut neuron_counts = std::collections::HashMap::new();
        for spike in output_spikes {
            *neuron_counts.entry(spike.neuron_id).or_insert(0) += 1;
        }
        
        let active_neurons = neuron_counts.len();
        println!("  Active neurons: {}", active_neurons);
        
        if !neuron_counts.is_empty() {
            let counts: Vec<f64> = neuron_counts.values()
                .map(|&count| count as f64)
                .collect();
            
            let mean_activity = stats.mean(&counts);
            let std_activity = stats.std(&counts);
            
            println!("  Mean activity:  {:.2} spikes/neuron", mean_activity);
            println!("  Std activity:   {:.2} spikes/neuron", std_activity);
        }

        // Inter-spike intervals
        if output_spikes.len() > 1 {
            let mut isis = Vec::new();
            for window in output_spikes.windows(2) {
                let isi = window[1].time.as_secs_f64() - window[0].time.as_secs_f64();
                isis.push(isi * 1000.0); // Convert to milliseconds
            }
            
            let mean_isi = stats.mean(&isis);
            let std_isi = stats.std(&isis);
            
            println!("  Mean ISI:       {:.2}ms", mean_isi);
            println!("  Std ISI:        {:.2}ms", std_isi);
        }
    }

    Ok(())
}

/// Save results to a file
fn save_results(spikes: &[Spike], output_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# SHNN Simple Network Results")?;
    writeln!(writer, "# Format: neuron_id,time_s,amplitude")?;
    
    for spike in spikes {
        writeln!(
            writer,
            "{},{:.6},{}",
            spike.neuron_id,
            spike.time.as_secs_f64(),
            spike.amplitude
        )?;
    }

    Ok(())
}

/// Simple network implementation for demonstration
struct SimpleNetwork {
    neurons: Vec<Option<Box<dyn NeuronModel>>>,
    connectivity: Vec<Vec<(usize, f32)>>, // (target_neuron, weight)
    plasticity_rules: Vec<Box<dyn PlasticityRule>>,
}

impl SimpleNetwork {
    fn new(num_neurons: usize) -> Self {
        Self {
            neurons: vec![None; num_neurons],
            connectivity: vec![Vec::new(); num_neurons],
            plasticity_rules: Vec::new(),
        }
    }

    fn add_neuron(&mut self, id: usize, neuron: Box<dyn NeuronModel>) {
        if id < self.neurons.len() {
            self.neurons[id] = Some(neuron);
        }
    }

    fn add_connection(&mut self, source: usize, target: usize, weight: f32) {
        if source < self.connectivity.len() {
            self.connectivity[source].push((target, weight));
        }
    }

    fn add_plasticity_rule(&mut self, rule: Box<dyn PlasticityRule>) {
        self.plasticity_rules.push(rule);
    }

    fn process_step(&self, input_spikes: &[Spike], dt: TimeStep) -> Result<Vec<Spike>> {
        let mut output_spikes = Vec::new();
        
        // Create synaptic input for each neuron
        let mut synaptic_inputs = vec![0.0f32; self.neurons.len()];
        
        // Process input spikes
        for spike in input_spikes {
            let source_id = spike.neuron_id as usize;
            if source_id < self.connectivity.len() {
                for &(target_id, weight) in &self.connectivity[source_id] {
                    if target_id < synaptic_inputs.len() {
                        synaptic_inputs[target_id] += spike.amplitude * weight;
                    }
                }
            }
        }

        // Update each neuron
        for (neuron_id, neuron_opt) in self.neurons.iter().enumerate() {
            if let Some(neuron) = neuron_opt {
                // Note: This is a simplified version - in reality we'd need proper
                // synaptic input structures and more sophisticated neuron updating
                // For this example, we'll create a simple mock update
                
                let input_current = synaptic_inputs[neuron_id];
                if input_current > 0.5 && rand::random::<f32>() < 0.1 {
                    // Simple probabilistic spiking for demonstration
                    let spike = Spike {
                        neuron_id: neuron_id as u32,
                        time: SpikeTime::from_time_step(TimeStep::zero()), // Simplified timing
                        amplitude: 1.0,
                        payload: Vec::new(),
                    };
                    output_spikes.push(spike);
                }
            }
        }

        Ok(output_spikes)
    }
}

/// Trait for neuron models (simplified for this example)
trait NeuronModel {
    fn update(&mut self, input: f32, dt: TimeStep) -> Result<Option<Spike>>;
    fn reset(&mut self);
}

/// Create random connectivity in the network
fn create_connectivity(
    network: &mut SimpleNetwork,
    probability: f32,
    rng: &mut StdRng,
) -> Result<()> {
    let num_neurons = network.neurons.len();
    
    for source in 0..num_neurons {
        for target in 0..num_neurons {
            if source != target && rng.gen::<f32>() < probability {
                let weight = rng.gen_range(0.1..1.0);
                network.add_connection(source, target, weight);
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let mut rng = StdRng::seed_from_u64(42);
        let network = create_network(10, &mut rng).unwrap();
        // Basic test that network was created
        assert_eq!(network.neurons.len(), 10);
    }

    #[test]
    fn test_spike_generation() {
        let spikes = generate_input_spikes(5, 0.1, 10.0, 42).unwrap();
        assert!(!spikes.is_empty());
        
        // Check spikes are within time bounds
        for spike in &spikes {
            assert!(spike.time.as_secs_f64() >= 0.0);
            assert!(spike.time.as_secs_f64() <= 0.1);
        }
    }

    #[test]
    fn test_analysis() {
        let spikes = vec![
            Spike {
                neuron_id: 0,
                time: SpikeTime::from_secs_f64(0.001),
                amplitude: 1.0,
                payload: Vec::new(),
            },
            Spike {
                neuron_id: 1,
                time: SpikeTime::from_secs_f64(0.002),
                amplitude: 1.0,
                payload: Vec::new(),
            },
        ];
        
        // Should not panic
        analyze_results(&spikes, &spikes, 0.1, false).unwrap();
    }
}