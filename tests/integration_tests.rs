Colonial//! Integration tests for SHNN zero-dependency refactoring
//!
//! This test suite validates that all zero-dependency components work together
//! correctly and that the migration from external dependencies is complete.

use shnn_core::{
    neuron::{LIFNeuron, IzhikevichNeuron, HHNeuron},
    network::{Network, NetworkBuilder},
    spike::{SpikeEvent, SpikeTime},
    plasticity::{STDPRule, HomeostaticPlasticity},
    learning::{HebbianLearning, ReinforcementLearning},
};
use shnn_async_runtime::{SHNNRuntime, TaskPriority};
use shnn_math::{
    Vector, Matrix, SparseMatrix,
    activation::{sigmoid, tanh, relu},
    math::{exp_approx, ln_approx, sqrt_approx},
};
use shnn_serialize::{Serialize, Deserialize, BinaryEncoder, BinaryDecoder};
use shnn_lockfree::queue::MPSCQueue;

use std::{
    sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}},
    time::{Duration, Instant},
    thread,
};

/// Test end-to-end neuromorphic simulation with zero dependencies
#[test]
fn test_end_to_end_neuromorphic_simulation() {
    // Create a small neural network
    let mut network = NetworkBuilder::new()
        .with_neurons(100)
        .with_connectivity(0.1) // 10% connectivity
        .with_learning_rule(Box::new(STDPRule::new(0.01, 20.0)))
        .build();
    
    // Create runtime for async processing
    let runtime = SHNNRuntime::new(4, 1024);
    let spike_count = Arc::new(AtomicU64::new(0));
    let simulation_time = Duration::from_millis(100);
    
    // Run simulation
    let spike_count_clone = spike_count.clone();
    let network_clone = Arc::new(std::sync::Mutex::new(network));
    let network_sim = network_clone.clone();
    
    let sim_handle = runtime.spawn_task(async move {
        let start = Instant::now();
        while start.elapsed() < simulation_time {
            let mut net = network_sim.lock().unwrap();
            
            // Inject random spikes
            for neuron_id in 0..10 {
                if rand::random::<f32>() < 0.1 {
                    let spike = SpikeEvent::new(neuron_id, SpikeTime::now(), 1.0);
                    net.process_spike(spike);
                    spike_count_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
            
            // Update network state
            net.update(0.001); // 1ms time step
            
            // Yield to allow other tasks
            shnn_async_runtime::yield_now().await;
        }
    }, TaskPriority::High);
    
    // Wait for simulation to complete
    let start = Instant::now();
    while sim_handle.try_get_result().is_none() {
        if start.elapsed() > Duration::from_secs(10) {
            panic!("Simulation timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
    
    // Verify simulation ran
    assert!(spike_count.load(Ordering::SeqCst) > 0);
    
    // Test serialization of network state
    let network = network_clone.lock().unwrap();
    let network_state = network.get_state();
    
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    network_state.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let _restored_state = shnn_core::network::NetworkState::deserialize(&mut decoder).unwrap();
    
    println!("✅ End-to-end neuromorphic simulation completed successfully");
}

/// Test concurrent spike processing with lock-free queues
#[test]
fn test_concurrent_spike_processing() {
    let spike_queue = Arc::new(MPSCQueue::new());
    let runtime = SHNNRuntime::new(4, 1024);
    let processed_spikes = Arc::new(AtomicU64::new(0));
    let total_spikes = 10000;
    
    // Spawn spike generators
    let mut generator_handles = Vec::new();
    for producer_id in 0..4 {
        let queue_clone = spike_queue.clone();
        let handle = runtime.spawn_task(async move {
            for i in 0..total_spikes / 4 {
                let spike = SpikeEvent::new(
                    producer_id * 1000 + i,
                    SpikeTime::from_millis(i as f64 * 0.1),
                    rand::random::<f32>(),
                );
                
                while queue_clone.push(spike).is_err() {
                    shnn_async_runtime::yield_now().await;
                }
            }
        }, TaskPriority::Normal);
        generator_handles.push(handle);
    }
    
    // Spawn spike processors
    let mut processor_handles = Vec::new();
    for _ in 0..2 {
        let queue_clone = spike_queue.clone();
        let processed_clone = processed_spikes.clone();
        let handle = runtime.spawn_task(async move {
            let mut local_processed = 0;
            
            loop {
                if let Some(spike) = queue_clone.pop() {
                    // Process spike (simulate computation)
                    let _activation = sigmoid(spike.amplitude);
                    local_processed += 1;
                    processed_clone.fetch_add(1, Ordering::SeqCst);
                } else {
                    shnn_async_runtime::yield_now().await;
                }
                
                if local_processed >= total_spikes / 2 {
                    break;
                }
            }
        }, TaskPriority::High);
        processor_handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let start = Instant::now();
    let all_handles: Vec<_> = generator_handles.into_iter().chain(processor_handles).collect();
    
    while all_handles.iter().any(|h| h.try_get_result().is_none()) {
        if start.elapsed() > Duration::from_secs(30) {
            panic!("Concurrent spike processing timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
    
    assert_eq!(processed_spikes.load(Ordering::SeqCst), total_spikes as u64);
    println!("✅ Processed {} spikes concurrently", total_spikes);
}

/// Test mathematical operations in neuromorphic context
#[test]
fn test_neuromorphic_math_operations() {
    let runtime = SHNNRuntime::new(2, 512);
    
    // Test neural network computations using custom math
    let weights = Matrix::from_rows(&[
        &[0.1, 0.2, 0.3],
        &[0.4, 0.5, 0.6],
        &[0.7, 0.8, 0.9],
    ]);
    
    let input = Vector::from_slice(&[1.0, 0.5, 0.2]);
    let completed = Arc::new(AtomicBool::new(false));
    let completed_clone = completed.clone();
    
    let math_handle = runtime.spawn_task(async move {
        // Matrix-vector multiplication
        let output = weights.multiply_vector(&input);
        
        // Apply activation functions
        let activated: Vec<f32> = output.data().iter()
            .map(|&x| sigmoid(x))
            .collect();
        
        // Verify results
        assert_eq!(activated.len(), 3);
        assert!(activated.iter().all(|&x| x >= 0.0 && x <= 1.0));
        
        // Test approximation functions
        for &value in &activated {
            let exp_val = exp_approx(value);
            let ln_val = if value > 0.0 { ln_approx(value) } else { 0.0 };
            let sqrt_val = sqrt_approx(value);
            
            assert!(exp_val > 0.0);
            assert!(sqrt_val >= 0.0);
            if value > 0.0 {
                assert!(ln_val.is_finite());
            }
        }
        
        completed_clone.store(true, Ordering::SeqCst);
    }, TaskPriority::Normal);
    
    // Wait for completion
    let start = Instant::now();
    while !completed.load(Ordering::SeqCst) {
        if start.elapsed() > Duration::from_secs(5) {
            panic!("Math operations timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
    
    // Ensure task completed successfully
    assert!(math_handle.try_get_result().is_some());
    println!("✅ Neuromorphic math operations completed successfully");
}

/// Test learning algorithms with zero dependencies
#[test]
fn test_learning_algorithms() {
    let mut network = NetworkBuilder::new()
        .with_neurons(50)
        .with_connectivity(0.2)
        .build();
    
    // Test Hebbian learning
    let mut hebbian = HebbianLearning::new(0.01);
    
    // Simulate training patterns
    for episode in 0..100 {
        // Create input pattern
        let input_pattern: Vec<f32> = (0..50)
            .map(|i| if (episode + i) % 10 < 5 { 1.0 } else { 0.0 })
            .collect();
        
        // Apply pattern to network
        for (neuron_id, &activation) in input_pattern.iter().enumerate() {
            if activation > 0.5 {
                let spike = SpikeEvent::new(
                    neuron_id,
                    SpikeTime::from_millis(episode as f64),
                    activation,
                );
                network.process_spike(spike);
            }
        }
        
        // Update learning
        hebbian.update(&mut network, 0.001);
        network.update(0.001);
    }
    
    // Test STDP
    let mut stdp = STDPRule::new(0.005, 20.0);
    
    // Create spike pairs for STDP
    for i in 0..10 {
        let pre_spike = SpikeEvent::new(0, SpikeTime::from_millis(i as f64 * 10.0), 1.0);
        let post_spike = SpikeEvent::new(1, SpikeTime::from_millis(i as f64 * 10.0 + 5.0), 1.0);
        
        network.process_spike(pre_spike);
        network.process_spike(post_spike);
        stdp.update_weight(0, 1, &network);
    }
    
    // Test homeostatic plasticity
    let mut homeostatic = HomeostaticPlasticity::new(0.1, 10.0);
    homeostatic.update(&mut network);
    
    println!("✅ Learning algorithms test completed successfully");
}

/// Test performance under realistic neuromorphic workload
#[test]
fn test_realistic_performance() {
    let runtime = SHNNRuntime::new(8, 2048);
    let network_size = 1000;
    let simulation_duration = Duration::from_millis(200);
    
    // Create large network
    let network = Arc::new(std::sync::Mutex::new(
        NetworkBuilder::new()
            .with_neurons(network_size)
            .with_connectivity(0.05) // 5% connectivity
            .with_learning_rule(Box::new(STDPRule::new(0.001, 50.0)))
            .build()
    ));
    
    let metrics = Arc::new(std::sync::Mutex::new(PerformanceMetrics::new()));
    let start_time = Instant::now();
    
    // Spawn multiple concurrent tasks
    let mut handles = Vec::new();
    
    // Spike generation task
    let network_clone = network.clone();
    let metrics_clone = metrics.clone();
    let spike_gen_handle = runtime.spawn_task(async move {
        let mut spike_count = 0;
        while start_time.elapsed() < simulation_duration {
            // Generate background activity
            for _ in 0..10 {
                let neuron_id = rand::random::<usize>() % network_size;
                if rand::random::<f32>() < 0.01 {
                    let spike = SpikeEvent::new(
                        neuron_id,
                        SpikeTime::now(),
                        rand::random::<f32>() * 0.5 + 0.5,
                    );
                    
                    network_clone.lock().unwrap().process_spike(spike);
                    spike_count += 1;
                }
            }
            
            metrics_clone.lock().unwrap().spikes_generated += spike_count;
            spike_count = 0;
            
            shnn_async_runtime::yield_now().await;
        }
    }, TaskPriority::High);
    handles.push(spike_gen_handle);
    
    // Network update task
    let network_clone = network.clone();
    let metrics_clone = metrics.clone();
    let update_handle = runtime.spawn_task(async move {
        let mut updates = 0;
        while start_time.elapsed() < simulation_duration {
            network_clone.lock().unwrap().update(0.001);
            updates += 1;
            
            if updates % 100 == 0 {
                metrics_clone.lock().unwrap().network_updates += 100;
            }
            
            shnn_async_runtime::yield_now().await;
        }
    }, TaskPriority::Normal);
    handles.push(update_handle);
    
    // Math computation task
    let metrics_clone = metrics.clone();
    let math_handle = runtime.spawn_task(async move {
        let mut computations = 0;
        while start_time.elapsed() < simulation_duration {
            // Simulate heavy math computations
            let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
            
            for &x in &data {
                let _result = sigmoid(x) + tanh(x) + relu(x) + exp_approx(x);
                computations += 1;
            }
            
            if computations % 1000 == 0 {
                metrics_clone.lock().unwrap().math_operations += 1000;
            }
            
            shnn_async_runtime::yield_now().await;
        }
    }, TaskPriority::Low);
    handles.push(math_handle);
    
    // Wait for all tasks to complete
    let wait_start = Instant::now();
    while handles.iter().any(|h| h.try_get_result().is_none()) {
        if wait_start.elapsed() > Duration::from_secs(30) {
            panic!("Performance test timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
    
    let final_metrics = metrics.lock().unwrap();
    let total_time = start_time.elapsed();
    
    println!("🚀 Performance Test Results:");
    println!("  Total simulation time: {:?}", total_time);
    println!("  Spikes generated: {}", final_metrics.spikes_generated);
    println!("  Network updates: {}", final_metrics.network_updates);
    println!("  Math operations: {}", final_metrics.math_operations);
    println!("  Spikes/sec: {:.0}", final_metrics.spikes_generated as f64 / total_time.as_secs_f64());
    println!("  Updates/sec: {:.0}", final_metrics.network_updates as f64 / total_time.as_secs_f64());
    println!("  Math ops/sec: {:.0}", final_metrics.math_operations as f64 / total_time.as_secs_f64());
    
    // Verify reasonable performance
    assert!(final_metrics.spikes_generated > 0);
    assert!(final_metrics.network_updates > 0);
    assert!(final_metrics.math_operations > 0);
    
    println!("✅ Realistic performance test completed successfully");
}

/// Test memory efficiency and cleanup
#[test]
fn test_memory_efficiency() {
    let initial_memory = get_memory_usage();
    
    {
        // Create and destroy many objects
        let runtime = SHNNRuntime::new(4, 1024);
        let mut networks = Vec::new();
        
        for _ in 0..10 {
            let network = NetworkBuilder::new()
                .with_neurons(100)
                .with_connectivity(0.1)
                .build();
            networks.push(network);
        }
        
        // Process some data
        for network in &mut networks {
            for i in 0..100 {
                let spike = SpikeEvent::new(i % 100, SpikeTime::from_millis(i as f64), 0.5);
                network.process_spike(spike);
            }
            network.update(0.001);
        }
        
        // Test serialization/deserialization
        for network in &networks {
            let state = network.get_state();
            let mut buffer = Vec::new();
            let mut encoder = BinaryEncoder::new(&mut buffer);
            state.serialize(&mut encoder).unwrap();
            
            let mut decoder = BinaryDecoder::new(&buffer);
            let _restored = shnn_core::network::NetworkState::deserialize(&mut decoder).unwrap();
        }
        
        // Objects should be dropped here
    }
    
    // Force garbage collection if possible
    thread::sleep(Duration::from_millis(100));
    
    let final_memory = get_memory_usage();
    
    println!("Memory usage - Initial: {} KB, Final: {} KB", 
             initial_memory / 1024, final_memory / 1024);
    
    // Memory should not have grown excessively
    let memory_growth = final_memory.saturating_sub(initial_memory);
    assert!(memory_growth < 100 * 1024 * 1024, // Less than 100MB growth
            "Excessive memory growth: {} bytes", memory_growth);
    
    println!("✅ Memory efficiency test completed successfully");
}

/// Test zero-dependency compilation
#[test]
fn test_zero_dependency_compilation() {
    // This test verifies that we can use all features without external dependencies
    
    // Async runtime
    let runtime = SHNNRuntime::new(1, 256);
    
    // Math operations
    let vector = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let matrix = Matrix::zeros(3, 3);
    let sparse = SparseMatrix::new(10, 10);
    
    // Activation functions
    let _sig = sigmoid(0.5);
    let _tanh_val = tanh(0.5);
    let _relu_val = relu(0.5);
    
    // Approximations
    let _exp_val = exp_approx(1.0);
    let _ln_val = ln_approx(2.0);
    let _sqrt_val = sqrt_approx(4.0);
    
    // Lock-free structures
    let queue = MPSCQueue::new();
    assert!(queue.push(42).is_ok());
    assert_eq!(queue.pop(), Some(42));
    
    // Serialization
    let data = vec![1u32, 2, 3, 4, 5];
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    data.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded = Vec::<u32>::deserialize(&mut decoder).unwrap();
    assert_eq!(data, decoded);
    
    // Neuromorphic components
    let neuron = LIFNeuron::new();
    let spike = SpikeEvent::new(0, SpikeTime::now(), 1.0);
    
    println!("✅ Zero-dependency compilation test completed successfully");
}

/// Performance metrics for testing
#[derive(Debug, Default)]
struct PerformanceMetrics {
    spikes_generated: u64,
    network_updates: u64,
    math_operations: u64,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self::default()
    }
}

/// Get current memory usage (approximation)
fn get_memory_usage() -> usize {
    // This is a simplified memory usage estimation
    // In a real implementation, you might use more sophisticated memory tracking
    
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(statm) = fs::read_to_string("/proc/self/statm") {
            if let Some(pages) = statm.split_whitespace().next() {
                if let Ok(pages_num) = pages.parse::<usize>() {
                    return pages_num * 4096; // Assume 4KB pages
                }
            }
        }
    }
    
    // Fallback: return a dummy value
    0
}

/// Fake random implementation for testing (since we're zero-dependency)
mod rand {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    static SEED: AtomicU64 = AtomicU64::new(1);
    
    pub fn random<T>() -> T 
    where 
        T: From<u32>,
    {
        // Simple LCG for testing
        let current = SEED.load(Ordering::SeqCst);
        let next = current.wrapping_mul(1103515245).wrapping_add(12345);
        SEED.store(next, Ordering::SeqCst);
        T::from((next >> 16) as u32)
    }
    
    pub fn random_usize() -> usize {
        random::<u32>() as usize
    }
}

/// Test that all major SHNN features work together
#[test]
fn test_comprehensive_integration() {
    println!("🧠 Starting comprehensive SHNN integration test...");
    
    // Test 1: Zero-dependency compilation
    test_zero_dependency_compilation();
    
    // Test 2: End-to-end simulation
    test_end_to_end_neuromorphic_simulation();
    
    // Test 3: Concurrent processing
    test_concurrent_spike_processing();
    
    // Test 4: Math operations
    test_neuromorphic_math_operations();
    
    // Test 5: Learning algorithms
    test_learning_algorithms();
    
    // Test 6: Memory efficiency
    test_memory_efficiency();
    
    println!("🎉 All integration tests passed!");
    println!("✅ SHNN zero-dependency refactoring validation complete!");
}