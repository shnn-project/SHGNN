//! Async neural network management
//!
//! This module provides high-level async interfaces for managing neural networks,
//! including distributed processing, load balancing, and real-time operation.

use crate::{
    error::{AsyncError, AsyncResult},
    runtime::{AsyncRuntime, RuntimeManager},
    channels::{SpikeChannel, ChannelConfig},
    monitoring::{NetworkMonitor, ProcessingMetrics},
    scheduling::{TaskScheduler, TaskPriority},
};

use shnn_core::{
    spike::{NeuronId, Spike, TimedSpike, SpikeTrain},
    time::{Time, Duration},
    neuron::{LIFNeuron, LIFConfig, NeuronPool},
    hypergraph::{HypergraphNetwork, Hyperedge, HyperedgeId},
    plasticity::{PlasticityManager, STDPConfig},
    memory::{SpikeQueue, SpikeBuffer},
};

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Instant,
};

use futures::{future::BoxFuture, stream::BoxStream, StreamExt};
use dashmap::DashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error, instrument};

/// Configuration for async processing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProcessingConfig {
    /// Maximum concurrent spike processing tasks
    pub max_concurrent_spikes: usize,
    /// Batch size for spike processing
    pub batch_size: usize,
    /// Processing timeout
    pub processing_timeout: Duration,
    /// Buffer capacity for spike queues
    pub buffer_capacity: usize,
    /// Enable real-time processing
    pub real_time: bool,
    /// Target processing latency (for real-time mode)
    pub target_latency: Duration,
    /// Enable plasticity processing
    pub enable_plasticity: bool,
    /// Plasticity update interval
    pub plasticity_interval: Duration,
    /// Enable monitoring
    pub enable_monitoring: bool,
    /// Statistics collection interval
    pub stats_interval: Duration,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            max_concurrent_spikes: 1000,
            batch_size: 100,
            processing_timeout: Duration::from_millis(100),
            buffer_capacity: 10000,
            real_time: false,
            target_latency: Duration::from_micros(100), // 100μs target
            enable_plasticity: true,
            plasticity_interval: Duration::from_millis(10),
            enable_monitoring: true,
            stats_interval: Duration::from_millis(1000),
        }
    }
}

/// Network state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkState {
    /// Network is initializing
    Initializing,
    /// Network is ready for processing
    Ready,
    /// Network is actively processing
    Processing,
    /// Network is paused
    Paused,
    /// Network is shutting down
    Shutdown,
    /// Network encountered an error
    Error(String),
}

impl Default for NetworkState {
    fn default() -> Self {
        Self::Initializing
    }
}

/// Handle to an async neural network
#[derive(Debug, Clone)]
pub struct NetworkHandle {
    /// Unique network identifier
    pub id: u64,
    /// Network name
    pub name: String,
    /// Current state
    pub state: Arc<RwLock<NetworkState>>,
    /// Configuration
    pub config: ProcessingConfig,
    /// Creation timestamp
    pub created_at: Instant,
}

impl NetworkHandle {
    /// Create a new network handle
    pub fn new(id: u64, name: String, config: ProcessingConfig) -> Self {
        Self {
            id,
            name,
            state: Arc::new(RwLock::new(NetworkState::default())),
            config,
            created_at: Instant::now(),
        }
    }
    
    /// Get current state
    pub fn state(&self) -> NetworkState {
        self.state.read().unwrap().clone()
    }
    
    /// Set state
    pub fn set_state(&self, state: NetworkState) {
        *self.state.write().unwrap() = state;
    }
    
    /// Check if network is ready
    pub fn is_ready(&self) -> bool {
        matches!(self.state(), NetworkState::Ready)
    }
    
    /// Check if network is processing
    pub fn is_processing(&self) -> bool {
        matches!(self.state(), NetworkState::Processing)
    }
    
    /// Get uptime
    pub fn uptime(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

/// Internal network representation
struct AsyncNetworkInner {
    /// Network handle
    handle: NetworkHandle,
    /// Neuron pool
    neurons: NeuronPool,
    /// Hypergraph connectivity
    hypergraph: HypergraphNetwork,
    /// Spike processing queue
    spike_queue: SpikeQueue,
    /// Spike history buffer
    spike_buffer: SpikeBuffer,
    /// Plasticity manager
    plasticity: PlasticityManager,
    /// Communication channels
    channels: HashMap<String, SpikeChannel>,
    /// Task scheduler
    scheduler: TaskScheduler,
    /// Network monitor
    monitor: Option<NetworkMonitor>,
    /// Current simulation time
    current_time: Time,
    /// Processing metrics
    metrics: ProcessingMetrics,
}

impl AsyncNetworkInner {
    fn new(handle: NetworkHandle) -> Self {
        Self {
            neurons: NeuronPool::new(),
            hypergraph: HypergraphNetwork::new(),
            spike_queue: SpikeQueue::with_capacity(handle.config.buffer_capacity),
            spike_buffer: SpikeBuffer::new(handle.config.buffer_capacity),
            plasticity: PlasticityManager::new(),
            channels: HashMap::new(),
            scheduler: TaskScheduler::new(),
            monitor: if handle.config.enable_monitoring {
                Some(NetworkMonitor::new())
            } else {
                None
            },
            current_time: Time::ZERO,
            metrics: ProcessingMetrics::new(),
            handle,
        }
    }
}

/// Async neural network manager
pub struct AsyncNetworkManager {
    /// Runtime manager
    runtime: Arc<RuntimeManager>,
    /// Active networks
    networks: Arc<DashMap<u64, Arc<RwLock<AsyncNetworkInner>>>>,
    /// Network ID counter
    next_network_id: Arc<std::sync::atomic::AtomicU64>,
    /// Global configuration
    global_config: ProcessingConfig,
    /// Manager state
    state: Arc<RwLock<ManagerState>>,
}

/// Manager state
#[derive(Debug, Clone)]
struct ManagerState {
    /// Whether manager is running
    running: bool,
    /// Number of active networks
    active_networks: usize,
    /// Total spikes processed
    total_spikes_processed: u64,
    /// Manager start time
    start_time: Instant,
}

impl AsyncNetworkManager {
    /// Create a new async network manager
    pub async fn new(config: ProcessingConfig) -> AsyncResult<Self> {
        let runtime = Arc::new(RuntimeManager::with_config(Default::default())?);
        
        Ok(Self {
            runtime,
            networks: Arc::new(DashMap::new()),
            next_network_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            global_config: config,
            state: Arc::new(RwLock::new(ManagerState {
                running: true,
                active_networks: 0,
                total_spikes_processed: 0,
                start_time: Instant::now(),
            })),
        })
    }
    
    /// Create a new neural network
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub async fn create_network(&self) -> AsyncResult<u64> {
        self.create_network_with_name("default".to_string()).await
    }
    
    /// Create a new neural network with name
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub async fn create_network_with_name(&self, name: String) -> AsyncResult<u64> {
        let network_id = self.next_network_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let handle = NetworkHandle::new(network_id, name.clone(), self.global_config.clone());
        let network = Arc::new(RwLock::new(AsyncNetworkInner::new(handle.clone())));
        
        self.networks.insert(network_id, network.clone());
        
        // Update manager state
        {
            let mut state = self.state.write().unwrap();
            state.active_networks += 1;
        }
        
        // Initialize network
        self.initialize_network(network_id).await?;
        
        #[cfg(feature = "tracing")]
        info!(network_id, name, "Created new neural network");
        
        Ok(network_id)
    }
    
    /// Initialize a network
    async fn initialize_network(&self, network_id: u64) -> AsyncResult<()> {
        let network = self.get_network(network_id)?;
        
        // Set state to ready
        {
            let net = network.read().unwrap();
            net.handle.set_state(NetworkState::Ready);
        }
        
        // Start background tasks if enabled
        if self.global_config.enable_monitoring {
            self.start_monitoring_task(network_id).await?;
        }
        
        if self.global_config.enable_plasticity {
            self.start_plasticity_task(network_id).await?;
        }
        
        Ok(())
    }
    
    /// Get network reference
    fn get_network(&self, network_id: u64) -> AsyncResult<Arc<RwLock<AsyncNetworkInner>>> {
        self.networks.get(&network_id)
            .map(|entry| entry.clone())
            .ok_or_else(|| AsyncError::network_with_id(network_id, "Network not found"))
    }
    
    /// Add a neuron to a network
    #[cfg_attr(feature = "tracing", instrument(skip(self, neuron)))]
    pub async fn add_neuron(&self, network_id: u64, neuron: LIFNeuron) -> AsyncResult<()> {
        let network = self.get_network(network_id)?;
        let mut net = network.write().unwrap();
        
        net.neurons.add_lif_neuron(neuron)
            .map_err(AsyncError::from)?;
        
        #[cfg(feature = "tracing")]
        debug!(network_id, "Added neuron to network");
        
        Ok(())
    }
    
    /// Add a hyperedge to a network
    #[cfg_attr(feature = "tracing", instrument(skip(self, hyperedge)))]
    pub async fn add_hyperedge(&self, network_id: u64, hyperedge: Hyperedge) -> AsyncResult<()> {
        let network = self.get_network(network_id)?;
        let mut net = network.write().unwrap();
        
        net.hypergraph.add_hyperedge(hyperedge)
            .map_err(AsyncError::from)?;
        
        #[cfg(feature = "tracing")]
        debug!(network_id, "Added hyperedge to network");
        
        Ok(())
    }
    
    /// Process a single spike
    #[cfg_attr(feature = "tracing", instrument(skip(self, spike)))]
    pub async fn process_spike(&self, network_id: u64, spike: Spike) -> AsyncResult<Vec<Spike>> {
        let network = self.get_network(network_id)?;
        
        // Set state to processing
        {
            let net = network.read().unwrap();
            net.handle.set_state(NetworkState::Processing);
        }
        
        let result = self.process_spike_internal(network.clone(), spike).await;
        
        // Reset state to ready
        {
            let net = network.read().unwrap();
            if net.handle.state() == NetworkState::Processing {
                net.handle.set_state(NetworkState::Ready);
            }
        }
        
        result
    }
    
    /// Internal spike processing
    async fn process_spike_internal(
        &self,
        network: Arc<RwLock<AsyncNetworkInner>>,
        spike: Spike,
    ) -> AsyncResult<Vec<Spike>> {
        let start_time = Instant::now();
        
        // Route spike through hypergraph
        let routes = {
            let net = network.read().unwrap();
            net.hypergraph.route_spike(&spike, net.current_time)
        };
        
        let mut output_spikes = Vec::new();
        
        // Process each route
        for route in routes {
            for &target_id in &route.targets {
                let mut net = network.write().unwrap();
                
                if let Some(neuron) = net.neurons.get_lif_neuron_mut(target_id) {
                    if let Some(output) = neuron.process_spike(&spike, route.delivery_time) {
                        output_spikes.push(output.clone());
                        
                        // Add to spike buffer
                        let timed_spike = TimedSpike::new(output, route.delivery_time);
                        net.spike_buffer.push(timed_spike.clone());
                        
                        // Schedule delayed delivery if needed
                        if route.delivery_time > net.current_time {
                            net.spike_queue.push(timed_spike);
                        }
                        
                        // Update metrics
                        net.metrics.spikes_processed += 1;
                        net.metrics.total_processing_time += start_time.elapsed();
                    }
                }
            }
        }
        
        // Update global stats
        {
            let mut state = self.state.write().unwrap();
            state.total_spikes_processed += 1;
        }
        
        #[cfg(feature = "tracing")]
        debug!(
            spike_source = spike.source.raw(),
            output_count = output_spikes.len(),
            processing_time_us = start_time.elapsed().as_micros(),
            "Processed spike"
        );
        
        Ok(output_spikes)
    }
    
    /// Process a batch of spikes
    #[cfg_attr(feature = "tracing", instrument(skip(self, spikes)))]
    pub async fn process_batch(&self, network_id: u64, spikes: Vec<Spike>) -> AsyncResult<Vec<Spike>> {
        let network = self.get_network(network_id)?;
        let mut all_outputs = Vec::new();
        
        // Process spikes in parallel batches
        let batch_size = self.global_config.batch_size;
        for chunk in spikes.chunks(batch_size) {
            let mut batch_outputs = Vec::new();
            
            // Process each spike in the batch
            for spike in chunk {
                let outputs = self.process_spike_internal(network.clone(), spike.clone()).await?;
                batch_outputs.extend(outputs);
            }
            
            all_outputs.extend(batch_outputs);
            
            // Yield to allow other tasks to run
            #[cfg(feature = "tokio-runtime")]
            tokio::task::yield_now().await;
        }
        
        #[cfg(feature = "tracing")]
        info!(
            network_id,
            input_count = spikes.len(),
            output_count = all_outputs.len(),
            "Processed spike batch"
        );
        
        Ok(all_outputs)
    }
    
    /// Update network simulation time
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub async fn update_time(&self, network_id: u64, new_time: Time) -> AsyncResult<()> {
        let network = self.get_network(network_id)?;
        let mut net = network.write().unwrap();
        
        let dt = new_time - net.current_time;
        net.current_time = new_time;
        
        // Update all neurons
        net.neurons.update_all(new_time, dt);
        
        // Process any ready spikes from queue
        let ready_spikes = net.spike_queue.pop_ready(new_time);
        drop(net); // Release lock before recursive processing
        
        for timed_spike in ready_spikes {
            self.process_spike_internal(network.clone(), timed_spike.spike).await?;
        }
        
        Ok(())
    }
    
    /// Start monitoring task for a network
    async fn start_monitoring_task(&self, network_id: u64) -> AsyncResult<()> {
        let interval = self.global_config.stats_interval;
        let networks = self.networks.clone();
        
        self.runtime.spawn(async move {
            let mut interval_timer = 
                #[cfg(feature = "tokio-runtime")]
                tokio::time::interval(interval.into());
                
            loop {
                #[cfg(feature = "tokio-runtime")]
                interval_timer.tick().await;
                
                if let Some(network) = networks.get(&network_id) {
                    let net = network.read().unwrap();
                    if let Some(ref monitor) = net.monitor {
                        monitor.collect_metrics(&net.metrics).await;
                    }
                } else {
                    break; // Network was removed
                }
            }
        })?;
        
        Ok(())
    }
    
    /// Start plasticity update task
    async fn start_plasticity_task(&self, network_id: u64) -> AsyncResult<()> {
        let interval = self.global_config.plasticity_interval;
        let networks = self.networks.clone();
        
        self.runtime.spawn(async move {
            let mut interval_timer = 
                #[cfg(feature = "tokio-runtime")]
                tokio::time::interval(interval.into());
                
            loop {
                #[cfg(feature = "tokio-runtime")]
                interval_timer.tick().await;
                
                if let Some(network) = networks.get(&network_id) {
                    let mut net = network.write().unwrap();
                    // Update plasticity (implementation would go here)
                    // net.plasticity.update(net.current_time);
                } else {
                    break; // Network was removed
                }
            }
        })?;
        
        Ok(())
    }
    
    /// Get network handle
    pub async fn get_network_handle(&self, network_id: u64) -> AsyncResult<NetworkHandle> {
        let network = self.get_network(network_id)?;
        let net = network.read().unwrap();
        Ok(net.handle.clone())
    }
    
    /// List all networks
    pub async fn list_networks(&self) -> Vec<NetworkHandle> {
        self.networks.iter()
            .map(|entry| {
                let net = entry.read().unwrap();
                net.handle.clone()
            })
            .collect()
    }
    
    /// Remove a network
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub async fn remove_network(&self, network_id: u64) -> AsyncResult<()> {
        if let Some((_, network)) = self.networks.remove(&network_id) {
            // Set network state to shutdown
            {
                let net = network.read().unwrap();
                net.handle.set_state(NetworkState::Shutdown);
            }
            
            // Update manager state
            {
                let mut state = self.state.write().unwrap();
                state.active_networks = state.active_networks.saturating_sub(1);
            }
            
            #[cfg(feature = "tracing")]
            info!(network_id, "Removed neural network");
            
            Ok(())
        } else {
            Err(AsyncError::network_with_id(network_id, "Network not found"))
        }
    }
    
    /// Get manager statistics
    pub async fn stats(&self) -> ManagerStats {
        let state = self.state.read().unwrap();
        ManagerStats {
            active_networks: state.active_networks,
            total_spikes_processed: state.total_spikes_processed,
            uptime: state.start_time.elapsed(),
            runtime_stats: self.runtime.stats(),
        }
    }
    
    /// Shutdown the manager
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub async fn shutdown(self) -> AsyncResult<()> {
        // Mark as not running
        {
            let mut state = self.state.write().unwrap();
            state.running = false;
        }
        
        // Shutdown all networks
        let network_ids: Vec<_> = self.networks.iter().map(|entry| *entry.key()).collect();
        for network_id in network_ids {
            let _ = self.remove_network(network_id).await;
        }
        
        #[cfg(feature = "tracing")]
        info!("Async network manager shutdown complete");
        
        Ok(())
    }
}

/// Manager statistics
#[derive(Debug, Clone)]
pub struct ManagerStats {
    /// Number of active networks
    pub active_networks: usize,
    /// Total spikes processed across all networks
    pub total_spikes_processed: u64,
    /// Manager uptime
    pub uptime: std::time::Duration,
    /// Runtime statistics
    pub runtime_stats: crate::runtime::RuntimeStats,
}

impl std::fmt::Display for ManagerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Manager Stats: {} networks, {} spikes processed, uptime: {:.1}s",
            self.active_networks,
            self.total_spikes_processed,
            self.uptime.as_secs_f64()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shnn_core::neuron::LIFConfig;
    
    #[tokio::test]
    async fn test_network_manager_creation() {
        let config = ProcessingConfig::default();
        let manager = AsyncNetworkManager::new(config).await.unwrap();
        
        let stats = manager.stats().await;
        assert_eq!(stats.active_networks, 0);
    }
    
    #[tokio::test]
    async fn test_create_network() {
        let config = ProcessingConfig::default();
        let manager = AsyncNetworkManager::new(config).await.unwrap();
        
        let network_id = manager.create_network().await.unwrap();
        assert_eq!(network_id, 1);
        
        let stats = manager.stats().await;
        assert_eq!(stats.active_networks, 1);
    }
    
    #[tokio::test]
    async fn test_add_neuron() {
        let config = ProcessingConfig::default();
        let manager = AsyncNetworkManager::new(config).await.unwrap();
        let network_id = manager.create_network().await.unwrap();
        
        let neuron = LIFNeuron::new(NeuronId::new(0), LIFConfig::default()).unwrap();
        let result = manager.add_neuron(network_id, neuron).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_process_spike() {
        let config = ProcessingConfig::default();
        let manager = AsyncNetworkManager::new(config).await.unwrap();
        let network_id = manager.create_network().await.unwrap();
        
        let spike = Spike::binary(NeuronId::new(1), Time::from_millis(1)).unwrap();
        let outputs = manager.process_spike(network_id, spike).await.unwrap();
        
        // Should succeed even with empty network
        assert!(outputs.is_empty());
    }
    
    #[tokio::test]
    async fn test_network_handle() {
        let config = ProcessingConfig::default();
        let handle = NetworkHandle::new(1, "test".to_string(), config);
        
        assert_eq!(handle.id, 1);
        assert_eq!(handle.name, "test");
        assert!(matches!(handle.state(), NetworkState::Initializing));
        
        handle.set_state(NetworkState::Ready);
        assert!(handle.is_ready());
    }
}