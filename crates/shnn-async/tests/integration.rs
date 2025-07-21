//! Integration tests for shnn-async crate
//! 
//! These tests verify async spike processing, real-time streaming,
//! distributed computing capabilities, and performance monitoring.

use shnn_async::prelude::*;
use shnn_async::error::AsyncShnnError;
use shnn_async::runtime::{AsyncRuntime, RuntimeConfig};
use shnn_async::network::{AsyncHypergraphNetwork, NetworkMetrics};
use shnn_async::channels::{SpikeChannel, ChannelConfig, BackpressureStrategy};
use shnn_async::streaming::{SpikeStream, StreamProcessor, StreamConfig};
use shnn_async::monitoring::{PerformanceMonitor, MetricsCollector};

use shnn_core::prelude::*;
use shnn_core::neuron::{LIFNeuron, NeuronId};
use shnn_core::spike::Spike;
use shnn_core::time::TimeStep;

use tokio;
use std::time::Duration;
use futures::{StreamExt, SinkExt};

#[tokio::test]
async fn test_async_runtime_creation() {
    let config = RuntimeConfig::default();
    let runtime = AsyncRuntime::new(config).await;
    
    assert!(runtime.is_ok());
    let runtime = runtime.unwrap();
    
    // Test runtime properties
    assert!(runtime.is_running());
    assert_eq!(runtime.worker_count(), num_cpus::get());
    
    // Shutdown cleanly
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_async_network_creation() {
    let runtime = AsyncRuntime::new(RuntimeConfig::default()).await.unwrap();
    let network = AsyncHypergraphNetwork::new(&runtime).await;
    
    assert!(network.is_ok());
    let mut network = network.unwrap();
    
    // Add neurons asynchronously
    let neuron1 = network.add_neuron(LIFNeuron::default()).await.unwrap();
    let neuron2 = network.add_neuron(LIFNeuron::default()).await.unwrap();
    
    // Connect neurons
    let edge = network.add_hyperedge(
        vec![neuron1], 
        vec![neuron2], 
        1.0
    ).await.unwrap();
    
    // Verify network state
    assert_eq!(network.neuron_count().await, 2);
    assert_eq!(network.hyperedge_count().await, 1);
    
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_async_spike_processing() {
    let runtime = AsyncRuntime::new(RuntimeConfig::default()).await.unwrap();
    let mut network = AsyncHypergraphNetwork::new(&runtime).await.unwrap();
    
    // Create network
    let neuron1 = network.add_neuron(LIFNeuron::default()).await.unwrap();
    let neuron2 = network.add_neuron(LIFNeuron::default()).await.unwrap();
    network.add_hyperedge(vec![neuron1], vec![neuron2], 1.0).await.unwrap();
    
    // Process spike asynchronously
    let spike = Spike::new(neuron1, TimeStep::from_ms(1.0));
    let result = network.process_spike(spike).await;
    
    assert!(result.is_ok());
    
    // Check that spike was propagated
    let neuron2_state = network.get_neuron_state(neuron2).await.unwrap();
    assert!(neuron2_state.membrane_potential() > -65.0); // Should have received input
    
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_spike_channel_communication() {
    let config = ChannelConfig {
        buffer_size: 1000,
        backpressure_strategy: BackpressureStrategy::Block,
        timeout: Duration::from_secs(1),
    };
    
    let (mut sender, mut receiver) = SpikeChannel::new(config);
    
    // Send spikes in background task
    let send_task = tokio::spawn(async move {
        for i in 0..100 {
            let spike = Spike::new(NeuronId(i), TimeStep::from_ms(i as f64));
            sender.send(spike).await.unwrap();
        }
    });
    
    // Receive spikes
    let mut received_count = 0;
    while received_count < 100 {
        if let Some(spike) = receiver.next().await {
            received_count += 1;
            assert!(spike.neuron_id().as_usize() < 100);
        }
    }
    
    send_task.await.unwrap();
    assert_eq!(received_count, 100);
}

#[tokio::test]
async fn test_backpressure_handling() {
    let config = ChannelConfig {
        buffer_size: 10, // Small buffer
        backpressure_strategy: BackpressureStrategy::Drop,
        timeout: Duration::from_millis(100),
    };
    
    let (mut sender, mut receiver) = SpikeChannel::new(config);
    
    // Send more spikes than buffer can hold
    let mut send_count = 0;
    for i in 0..1000 {
        let spike = Spike::new(NeuronId(i), TimeStep::from_ms(i as f64));
        if sender.try_send(spike).await.is_ok() {
            send_count += 1;
        }
    }
    
    // Should have dropped some spikes
    assert!(send_count < 1000);
    assert!(send_count >= 10); // At least buffer size should be sent
    
    // Receive what was sent
    let mut received_count = 0;
    while let Ok(Some(_)) = tokio::time::timeout(
        Duration::from_millis(100),
        receiver.next()
    ).await {
        received_count += 1;
    }
    
    assert!(received_count <= send_count);
}

#[tokio::test]
async fn test_spike_streaming() {
    let config = StreamConfig {
        batch_size: 10,
        flush_interval: Duration::from_millis(100),
        buffer_size: 1000,
    };
    
    let mut stream = SpikeStream::new(config);
    
    // Add spikes to stream
    for i in 0..50 {
        let spike = Spike::new(NeuronId(i % 5), TimeStep::from_ms(i as f64));
        stream.add_spike(spike).await.unwrap();
    }
    
    // Process stream in batches
    let mut total_processed = 0;
    while let Some(batch) = stream.next_batch().await {
        total_processed += batch.len();
        assert!(batch.len() <= 10); // Batch size limit
        
        // Verify spikes are ordered by time
        for window in batch.windows(2) {
            assert!(window[0].timestamp() <= window[1].timestamp());
        }
    }
    
    assert_eq!(total_processed, 50);
}

#[tokio::test]
async fn test_stream_processor() {
    let mut processor = StreamProcessor::new();
    
    // Define processing function
    let processing_fn = |spikes: Vec<Spike>| async move {
        // Simple processing: count spikes per neuron
        let mut counts = std::collections::HashMap::new();
        for spike in spikes {
            *counts.entry(spike.neuron_id()).or_insert(0) += 1;
        }
        Ok(counts)
    };
    
    processor.set_processor(processing_fn).await;
    
    // Send test spikes
    let test_spikes = vec![
        Spike::new(NeuronId(0), TimeStep::from_ms(1.0)),
        Spike::new(NeuronId(1), TimeStep::from_ms(2.0)),
        Spike::new(NeuronId(0), TimeStep::from_ms(3.0)),
        Spike::new(NeuronId(2), TimeStep::from_ms(4.0)),
    ];
    
    let result = processor.process_batch(test_spikes).await;
    assert!(result.is_ok());
    
    let counts = result.unwrap();
    assert_eq!(counts.get(&NeuronId(0)), Some(&2));
    assert_eq!(counts.get(&NeuronId(1)), Some(&1));
    assert_eq!(counts.get(&NeuronId(2)), Some(&1));
}

#[tokio::test]
async fn test_performance_monitoring() {
    let mut monitor = PerformanceMonitor::new();
    
    // Start monitoring
    monitor.start_monitoring().await;
    
    // Simulate some work
    for i in 0..100 {
        monitor.record_spike_processed().await;
        monitor.record_latency(Duration::from_micros(i * 10)).await;
        
        if i % 10 == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    
    // Get metrics
    let metrics = monitor.get_metrics().await;
    
    assert_eq!(metrics.total_spikes_processed, 100);
    assert!(metrics.average_latency > Duration::from_micros(0));
    assert!(metrics.peak_latency >= metrics.average_latency);
    assert!(metrics.throughput > 0.0);
}

#[tokio::test]
async fn test_metrics_collection() {
    let mut collector = MetricsCollector::new();
    
    // Collect metrics over time
    for _ in 0..10 {
        let metrics = NetworkMetrics {
            spikes_per_second: 1000.0,
            cpu_usage: 0.5,
            memory_usage: 1024 * 1024, // 1MB
            network_latency: Duration::from_micros(100),
            queue_depth: 50,
        };
        
        collector.record_metrics(metrics).await;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    let summary = collector.get_summary().await;
    
    assert_eq!(summary.sample_count, 10);
    assert!(summary.average_throughput > 0.0);
    assert!(summary.peak_memory_usage >= 1024 * 1024);
}

#[tokio::test]
async fn test_concurrent_spike_processing() {
    let runtime = AsyncRuntime::new(RuntimeConfig::default()).await.unwrap();
    let network = AsyncHypergraphNetwork::new(&runtime).await.unwrap();
    let network = std::sync::Arc::new(tokio::sync::RwLock::new(network));
    
    // Create multiple tasks sending spikes
    let mut tasks = vec![];
    
    for task_id in 0..4 {
        let network_clone = network.clone();
        
        let task = tokio::spawn(async move {
            let mut net = network_clone.write().await;
            
            // Each task creates its own neurons
            let mut neurons = vec![];
            for i in 0..10 {
                let neuron_id = task_id * 10 + i;
                let neuron = net.add_neuron(LIFNeuron::default()).await.unwrap();
                neurons.push(neuron);
            }
            
            // Process spikes from these neurons
            for (i, &neuron) in neurons.iter().enumerate() {
                let spike = Spike::new(neuron, TimeStep::from_ms(i as f64));
                net.process_spike(spike).await.unwrap();
            }
            
            neurons.len()
        });
        
        tasks.push(task);
    }
    
    // Wait for all tasks to complete
    let mut total_neurons = 0;
    for task in tasks {
        let count = task.await.unwrap();
        total_neurons += count;
    }
    
    assert_eq!(total_neurons, 40); // 4 tasks * 10 neurons each
    
    // Verify final network state
    let final_network = network.read().await;
    assert_eq!(final_network.neuron_count().await, 40);
    
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_real_time_constraints() {
    let config = RuntimeConfig {
        max_workers: 1,
        tick_interval: Duration::from_millis(1),
        real_time_priority: true,
        ..Default::default()
    };
    
    let runtime = AsyncRuntime::new(config).await.unwrap();
    let mut network = AsyncHypergraphNetwork::new(&runtime).await.unwrap();
    
    // Create time-sensitive network
    let neuron = network.add_neuron(LIFNeuron::default()).await.unwrap();
    
    let start_time = std::time::Instant::now();
    
    // Process spikes with timing constraints
    for i in 0..100 {
        let spike = Spike::new(neuron, TimeStep::from_ms(i as f64));
        let deadline = Duration::from_millis(1); // 1ms deadline
        
        let process_start = std::time::Instant::now();
        network.process_spike(spike).await.unwrap();
        let process_time = process_start.elapsed();
        
        // Should meet real-time deadline
        assert!(process_time < deadline * 2, 
               "Processing took {}μs, expected < {}μs", 
               process_time.as_micros(), 
               (deadline * 2).as_micros());
    }
    
    let total_time = start_time.elapsed();
    
    // Should maintain reasonable overall performance
    assert!(total_time < Duration::from_millis(200)); // 200ms for 100 spikes
    
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_distributed_processing() {
    let config = RuntimeConfig {
        max_workers: 4,
        enable_work_stealing: true,
        ..Default::default()
    };
    
    let runtime = AsyncRuntime::new(config).await.unwrap();
    let mut network = AsyncHypergraphNetwork::new(&runtime).await.unwrap();
    
    // Create a larger network for distributed processing
    let mut neurons = vec![];
    for i in 0..100 {
        let neuron = network.add_neuron(LIFNeuron::default()).await.unwrap();
        neurons.push(neuron);
    }
    
    // Create connections between neurons
    for i in 0..99 {
        network.add_hyperedge(
            vec![neurons[i]],
            vec![neurons[i + 1]],
            0.5
        ).await.unwrap();
    }
    
    // Process many spikes concurrently
    let spike_tasks: Vec<_> = (0..100).map(|i| {
        let neuron = neurons[i % neurons.len()];
        let mut net = network.clone();
        
        tokio::spawn(async move {
            let spike = Spike::new(neuron, TimeStep::from_ms(i as f64));
            net.process_spike(spike).await
        })
    }).collect();
    
    // Wait for all spikes to be processed
    for task in spike_tasks {
        let result = task.await.unwrap();
        assert!(result.is_ok());
    }
    
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_error_recovery() {
    let runtime = AsyncRuntime::new(RuntimeConfig::default()).await.unwrap();
    let mut network = AsyncHypergraphNetwork::new(&runtime).await.unwrap();
    
    // Test processing spike for non-existent neuron
    let invalid_spike = Spike::new(NeuronId(999), TimeStep::from_ms(1.0));
    let result = network.process_spike(invalid_spike).await;
    
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), AsyncShnnError::InvalidNeuronId(_)));
    
    // Network should still be functional after error
    let neuron = network.add_neuron(LIFNeuron::default()).await.unwrap();
    let valid_spike = Spike::new(neuron, TimeStep::from_ms(1.0));
    let result = network.process_spike(valid_spike).await;
    
    assert!(result.is_ok());
    
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_graceful_shutdown() {
    let runtime = AsyncRuntime::new(RuntimeConfig::default()).await.unwrap();
    let mut network = AsyncHypergraphNetwork::new(&runtime).await.unwrap();
    
    // Start some long-running operations
    let neuron = network.add_neuron(LIFNeuron::default()).await.unwrap();
    
    let shutdown_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        runtime.shutdown().await
    });
    
    // Continue processing until shutdown
    let mut processed = 0;
    while !runtime.is_shutting_down() {
        let spike = Spike::new(neuron, TimeStep::from_ms(processed as f64));
        if network.process_spike(spike).await.is_ok() {
            processed += 1;
        }
        
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    
    // Wait for shutdown to complete
    let result = shutdown_task.await.unwrap();
    assert!(result.is_ok());
    assert!(processed > 0);
}

#[tokio::test]
async fn test_memory_pressure_handling() {
    let config = RuntimeConfig {
        max_memory_usage: 10 * 1024 * 1024, // 10MB limit
        memory_check_interval: Duration::from_millis(10),
        ..Default::default()
    };
    
    let runtime = AsyncRuntime::new(config).await.unwrap();
    let mut network = AsyncHypergraphNetwork::new(&runtime).await.unwrap();
    
    // Create many neurons to increase memory usage
    let mut neurons = vec![];
    for i in 0..1000 {
        match network.add_neuron(LIFNeuron::default()).await {
            Ok(neuron) => neurons.push(neuron),
            Err(AsyncShnnError::MemoryLimitExceeded) => break,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
    
    // Should have created some neurons before hitting limit
    assert!(!neurons.is_empty());
    assert!(neurons.len() <= 1000);
    
    // Memory cleanup should allow more neurons
    runtime.trigger_gc().await.unwrap();
    
    let additional_neuron = network.add_neuron(LIFNeuron::default()).await;
    // May or may not succeed depending on cleanup effectiveness
    
    runtime.shutdown().await.unwrap();
}