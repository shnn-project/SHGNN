//! Integration tests for shnn-core crate
//! 
//! These tests verify the interaction between different components of the core
//! neuromorphic computing framework, ensuring proper spike propagation,
//! plasticity mechanisms, and network dynamics.

use shnn_core::prelude::*;
use shnn_core::error::ShnnError;
use shnn_core::time::{SimulationTime, TimeStep};
use shnn_core::spike::{Spike, SpikeId, SpikeTrain};
use shnn_core::neuron::{NeuronId, NeuronState, LIFNeuron, AdExNeuron, IzhikevichNeuron};
use shnn_core::hypergraph::{HypergraphNetwork, HyperedgeId, ConnectionWeight};
use shnn_core::plasticity::{STDPRule, PlasticityUpdate, HomeostasisRule};
use shnn_core::encoding::{RateEncoder, TemporalEncoder, PopulationEncoder};

#[test]
fn test_basic_network_simulation() {
    let mut network = HypergraphNetwork::new();
    
    // Add neurons
    let neuron1 = network.add_neuron(LIFNeuron::default()).unwrap();
    let neuron2 = network.add_neuron(LIFNeuron::default()).unwrap();
    
    // Connect neurons
    let edge = network.add_hyperedge(vec![neuron1], vec![neuron2], 1.0).unwrap();
    
    // Create input spike
    let spike = Spike::new(neuron1, TimeStep::from_ms(1.0));
    
    // Process spike
    let result = network.process_spike(spike);
    assert!(result.is_ok());
    
    // Verify network state
    assert_eq!(network.neuron_count(), 2);
    assert_eq!(network.hyperedge_count(), 1);
}

#[test]
fn test_lif_neuron_dynamics() {
    let mut neuron = LIFNeuron::default();
    let dt = TimeStep::from_ms(0.1);
    
    // Test membrane potential integration
    neuron.integrate(1.0, dt);
    assert!(neuron.membrane_potential() > 0.0);
    
    // Test spike generation
    let threshold = neuron.threshold();
    neuron.set_membrane_potential(threshold + 0.1);
    
    let spike_result = neuron.update(dt);
    assert!(spike_result.is_some());
    
    // Test reset after spike
    assert!(neuron.membrane_potential() < threshold);
}

#[test]
fn test_adex_neuron_adaptation() {
    let mut neuron = AdExNeuron::default();
    let dt = TimeStep::from_ms(0.1);
    
    // Test adaptation current
    let initial_adaptation = neuron.adaptation_current();
    
    // Stimulate neuron
    neuron.integrate(5.0, dt);
    
    // Adaptation should change
    assert_ne!(neuron.adaptation_current(), initial_adaptation);
    
    // Test spike-triggered adaptation
    neuron.set_membrane_potential(neuron.threshold() + 1.0);
    let spike_result = neuron.update(dt);
    
    if spike_result.is_some() {
        assert!(neuron.adaptation_current() > initial_adaptation);
    }
}

#[test]
fn test_izhikevich_neuron_patterns() {
    // Test different firing patterns
    let patterns = [
        (0.02, 0.2, -65.0, 8.0),   // Regular spiking
        (0.02, 0.25, -65.0, 2.0),  // Intrinsically bursting
        (0.02, 0.2, -50.0, 2.0),   // Chattering
        (0.1, 0.2, -65.0, 2.0),    // Fast spiking
    ];
    
    for (a, b, c, d) in patterns.iter() {
        let mut neuron = IzhikevichNeuron::new(*a, *b, *c, *d);
        let dt = TimeStep::from_ms(0.1);
        
        // Stimulate and check for spikes
        let mut spike_count = 0;
        for _ in 0..1000 {
            neuron.integrate(10.0, dt);
            if neuron.update(dt).is_some() {
                spike_count += 1;
            }
        }
        
        // Should generate some spikes with sufficient stimulation
        assert!(spike_count > 0, "Pattern ({}, {}, {}, {}) didn't spike", a, b, c, d);
    }
}

#[test]
fn test_hypergraph_connectivity() {
    let mut network = HypergraphNetwork::new();
    
    // Create a small network
    let neurons: Vec<_> = (0..5).map(|_| network.add_neuron(LIFNeuron::default()).unwrap()).collect();
    
    // Test one-to-many connection
    let edge1 = network.add_hyperedge(
        vec![neurons[0]], 
        vec![neurons[1], neurons[2]], 
        0.5
    ).unwrap();
    
    // Test many-to-one connection
    let edge2 = network.add_hyperedge(
        vec![neurons[1], neurons[2]], 
        vec![neurons[3]], 
        0.8
    ).unwrap();
    
    // Test many-to-many connection
    let edge3 = network.add_hyperedge(
        vec![neurons[0], neurons[1]], 
        vec![neurons[3], neurons[4]], 
        1.0
    ).unwrap();
    
    // Verify connections
    assert_eq!(network.hyperedge_count(), 3);
    
    // Test spike propagation through hyperedges
    let spike = Spike::new(neurons[0], TimeStep::from_ms(1.0));
    let result = network.process_spike(spike);
    assert!(result.is_ok());
    
    // Check that connected neurons received input
    let neuron1_state = network.get_neuron_state(neurons[1]).unwrap();
    let neuron2_state = network.get_neuron_state(neurons[2]).unwrap();
    
    assert!(neuron1_state.membrane_potential() > 0.0);
    assert!(neuron2_state.membrane_potential() > 0.0);
}

#[test]
fn test_stdp_learning() {
    let mut stdp = STDPRule::default();
    
    // Test LTP (long-term potentiation)
    let pre_spike_time = TimeStep::from_ms(10.0);
    let post_spike_time = TimeStep::from_ms(15.0);
    
    let weight_change = stdp.update_weight(
        1.0, 
        pre_spike_time, 
        post_spike_time
    );
    
    // Pre before post should increase weight (LTP)
    assert!(weight_change > 0.0);
    
    // Test LTD (long-term depression)
    let weight_change = stdp.update_weight(
        1.0, 
        post_spike_time, 
        pre_spike_time
    );
    
    // Post before pre should decrease weight (LTD)
    assert!(weight_change < 0.0);
}

#[test]
fn test_homeostasis_regulation() {
    let mut homeostasis = HomeostasisRule::new(10.0, 0.01); // Target 10Hz, learning rate 0.01
    
    // Test with high firing rate
    let high_rate = 20.0;
    let adjustment = homeostasis.regulate_activity(high_rate);
    
    // Should decrease excitability
    assert!(adjustment < 0.0);
    
    // Test with low firing rate
    let low_rate = 5.0;
    let adjustment = homeostasis.regulate_activity(low_rate);
    
    // Should increase excitability
    assert!(adjustment > 0.0);
}

#[test]
fn test_rate_encoding() {
    let encoder = RateEncoder::new(0.0, 100.0, 100.0); // 0-100 value range, 100Hz max
    
    // Test encoding
    let spike_train = encoder.encode(50.0, TimeStep::from_ms(100.0));
    
    // Should generate approximately 50% of max rate
    let expected_spikes = (50.0 * 0.1) as usize; // 50Hz for 100ms
    let actual_spikes = spike_train.spike_count();
    
    // Allow some variance due to stochastic nature
    assert!((actual_spikes as f64 - expected_spikes as f64).abs() <= 3.0);
}

#[test]
fn test_temporal_encoding() {
    let encoder = TemporalEncoder::new(TimeStep::from_ms(1.0), TimeStep::from_ms(10.0));
    
    // Test first-spike timing
    let value = 0.5; // Middle of range
    let spike_time = encoder.encode_first_spike(value);
    
    // Should be between min and max delay
    assert!(spike_time >= TimeStep::from_ms(1.0));
    assert!(spike_time <= TimeStep::from_ms(10.0));
    
    // Higher values should have shorter delays
    let high_value_time = encoder.encode_first_spike(0.9);
    let low_value_time = encoder.encode_first_spike(0.1);
    
    assert!(high_value_time < low_value_time);
}

#[test]
fn test_population_encoding() {
    let encoder = PopulationEncoder::new(10, 0.0, 100.0); // 10 neurons, 0-100 range
    
    // Test encoding
    let activations = encoder.encode(50.0);
    
    // Should have 10 activation values
    assert_eq!(activations.len(), 10);
    
    // Middle value should activate middle neurons most
    let max_activation_idx = activations
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    
    // Should be around the middle (indices 4-6 for 10 neurons)
    assert!(max_activation_idx >= 3 && max_activation_idx <= 6);
}

#[test]
fn test_spike_train_analysis() {
    let mut spike_train = SpikeTrain::new();
    
    // Add spikes at regular intervals
    for i in 0..10 {
        let spike = Spike::new(NeuronId(0), TimeStep::from_ms(i as f64 * 10.0));
        spike_train.add_spike(spike);
    }
    
    // Test spike count
    assert_eq!(spike_train.spike_count(), 10);
    
    // Test firing rate calculation
    let rate = spike_train.firing_rate(TimeStep::from_ms(100.0));
    assert!((rate - 100.0).abs() < 1.0); // Should be ~100Hz
    
    // Test inter-spike intervals
    let isis = spike_train.inter_spike_intervals();
    assert_eq!(isis.len(), 9); // n-1 intervals for n spikes
    
    for isi in isis {
        assert!((isi.as_ms() - 10.0).abs() < 0.1); // Should be ~10ms intervals
    }
}

#[test]
fn test_network_plasticity_integration() {
    let mut network = HypergraphNetwork::new();
    
    // Create neurons
    let pre_neuron = network.add_neuron(LIFNeuron::default()).unwrap();
    let post_neuron = network.add_neuron(LIFNeuron::default()).unwrap();
    
    // Connect with plastic connection
    let edge = network.add_hyperedge(vec![pre_neuron], vec![post_neuron], 0.5).unwrap();
    network.enable_plasticity(edge, STDPRule::default()).unwrap();
    
    // Stimulate pre-neuron to spike
    let pre_spike = Spike::new(pre_neuron, TimeStep::from_ms(10.0));
    network.process_spike(pre_spike).unwrap();
    
    // Stimulate post-neuron to spike shortly after
    let post_spike = Spike::new(post_neuron, TimeStep::from_ms(15.0));
    network.process_spike(post_spike).unwrap();
    
    // Weight should have increased due to STDP
    let updated_weight = network.get_hyperedge_weight(edge).unwrap();
    assert!(updated_weight > 0.5, "STDP should have increased weight");
}

#[test]
fn test_error_handling() {
    let mut network = HypergraphNetwork::new();
    
    // Test invalid neuron access
    let invalid_neuron = NeuronId(999);
    let result = network.get_neuron_state(invalid_neuron);
    assert!(matches!(result, Err(ShnnError::InvalidNeuronId(_))));
    
    // Test invalid hyperedge access
    let invalid_edge = HyperedgeId(999);
    let result = network.get_hyperedge_weight(invalid_edge);
    assert!(matches!(result, Err(ShnnError::InvalidHyperedgeId(_))));
    
    // Test empty hyperedge creation
    let result = network.add_hyperedge(vec![], vec![], 1.0);
    assert!(matches!(result, Err(ShnnError::EmptyHyperedge)));
}

#[test]
fn test_simulation_time_operations() {
    let time1 = TimeStep::from_ms(10.0);
    let time2 = TimeStep::from_ms(5.0);
    
    // Test arithmetic operations
    let sum = time1 + time2;
    assert_eq!(sum.as_ms(), 15.0);
    
    let diff = time1 - time2;
    assert_eq!(diff.as_ms(), 5.0);
    
    // Test comparisons
    assert!(time1 > time2);
    assert!(time2 < time1);
    
    // Test conversions
    let microseconds = time1.as_us();
    assert_eq!(microseconds, 10000.0);
    
    let seconds = time1.as_s();
    assert_eq!(seconds, 0.01);
}

#[test]
fn test_memory_management() {
    let mut network = HypergraphNetwork::new();
    
    // Add many neurons and connections
    let neurons: Vec<_> = (0..1000)
        .map(|_| network.add_neuron(LIFNeuron::default()).unwrap())
        .collect();
    
    // Create random connections
    for i in 0..500 {
        let pre = neurons[i % neurons.len()];
        let post = neurons[(i + 1) % neurons.len()];
        network.add_hyperedge(vec![pre], vec![post], 0.5).unwrap();
    }
    
    // Process many spikes
    for i in 0..10000 {
        let neuron = neurons[i % neurons.len()];
        let spike = Spike::new(neuron, TimeStep::from_ms(i as f64 * 0.1));
        network.process_spike(spike).unwrap();
    }
    
    // Network should still be functional
    assert_eq!(network.neuron_count(), 1000);
    assert_eq!(network.hyperedge_count(), 500);
}

#[cfg(feature = "std")]
#[test]
fn test_multithreaded_access() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let network = Arc::new(Mutex::new(HypergraphNetwork::new()));
    
    // Create neurons in multiple threads
    let mut handles = vec![];
    
    for i in 0..4 {
        let network_clone = Arc::clone(&network);
        let handle = thread::spawn(move || {
            for j in 0..100 {
                let mut net = network_clone.lock().unwrap();
                let neuron = net.add_neuron(LIFNeuron::default()).unwrap();
                
                // Process a spike
                let spike = Spike::new(neuron, TimeStep::from_ms((i * 100 + j) as f64));
                net.process_spike(spike).unwrap();
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state
    let final_network = network.lock().unwrap();
    assert_eq!(final_network.neuron_count(), 400);
}

#[test]
fn test_numerical_stability() {
    let mut neuron = LIFNeuron::default();
    let very_small_dt = TimeStep::from_ms(0.001);
    let very_large_input = 1000.0;
    
    // Test with extreme values
    for _ in 0..10000 {
        neuron.integrate(very_large_input, very_small_dt);
        neuron.update(very_small_dt);
        
        // Membrane potential should remain finite
        assert!(neuron.membrane_potential().is_finite());
        assert!(!neuron.membrane_potential().is_nan());
    }
}