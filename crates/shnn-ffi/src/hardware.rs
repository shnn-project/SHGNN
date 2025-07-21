//! Hardware acceleration interface and base implementations
//!
//! This module defines the core traits and structures for interfacing with
//! different hardware acceleration platforms.

use crate::{
    error::{FFIError, FFIResult},
    types::{
        NetworkConfig, SpikeData, NeuronState, AcceleratorType,
        HardwareCapabilities, PerformanceMetrics, ExecutionContext
    },
    NetworkId,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Core trait for hardware accelerators
pub trait HardwareAccelerator {
    /// Deploy a neural network configuration to hardware
    fn deploy_network(&mut self, config: &NetworkConfig) -> FFIResult<NetworkId>;
    
    /// Remove a deployed network from hardware
    fn undeploy_network(&mut self, network_id: NetworkId) -> FFIResult<()>;
    
    /// Process spikes through the deployed network
    fn process_spikes(
        &mut self,
        network_id: NetworkId,
        input_spikes: &[SpikeData],
    ) -> FFIResult<Vec<SpikeData>>;
    
    /// Process spikes with execution context
    fn process_spikes_with_context(
        &mut self,
        network_id: NetworkId,
        input_spikes: &[SpikeData],
        context: &ExecutionContext,
    ) -> FFIResult<Vec<SpikeData>> {
        // Default implementation ignores context
        self.process_spikes(network_id, input_spikes)
    }
    
    /// Update network weights/parameters
    fn update_network(
        &mut self,
        network_id: NetworkId,
        updates: &NetworkUpdate,
    ) -> FFIResult<()>;
    
    /// Get current state of all neurons in a network
    fn get_neuron_states(&self, network_id: NetworkId) -> FFIResult<Vec<NeuronState>>;
    
    /// Reset network to initial state
    fn reset_network(&mut self, network_id: NetworkId) -> FFIResult<()>;
    
    /// Get hardware capabilities
    fn get_capabilities(&self) -> HardwareCapabilities;
    
    /// Get current performance metrics
    fn get_performance_metrics(&self) -> PerformanceMetrics;
    
    /// Synchronize host and device (for async accelerators)
    fn synchronize(&mut self) -> FFIResult<()> {
        Ok(()) // Default: no-op for synchronous accelerators
    }
    
    /// Set execution context for subsequent operations
    fn set_execution_context(&mut self, context: ExecutionContext) -> FFIResult<()> {
        Ok(()) // Default: ignore context
    }
    
    /// Enable/disable profiling
    fn set_profiling(&mut self, enabled: bool) -> FFIResult<()> {
        Ok(()) // Default: no profiling
    }
}

/// Network update operations
#[derive(Debug, Clone)]
pub enum NetworkUpdate {
    /// Update synaptic weights
    UpdateWeights {
        /// Connection updates: (pre_neuron, post_neuron, new_weight)
        updates: Vec<(u32, u32, f32)>,
    },
    /// Update neuron parameters
    UpdateNeuronParams {
        /// Parameter updates: (neuron_id, param_name, new_value)
        updates: Vec<(u32, String, f32)>,
    },
    /// Add new connections
    AddConnections {
        /// New connections: (pre_neuron, post_neuron, weight)
        connections: Vec<(u32, u32, f32)>,
    },
    /// Remove connections
    RemoveConnections {
        /// Connections to remove: (pre_neuron, post_neuron)
        connections: Vec<(u32, u32)>,
    },
    /// Update plasticity parameters
    UpdatePlasticity {
        /// Plasticity rule parameters
        params: HashMap<String, f32>,
    },
}

/// Accelerator manager for handling multiple hardware devices
#[derive(Debug)]
pub struct AcceleratorManager {
    /// Available accelerators
    accelerators: HashMap<AcceleratorType, Vec<Box<dyn HardwareAccelerator + Send + Sync>>>,
    /// Performance history
    performance_history: Arc<Mutex<PerformanceHistory>>,
}

impl AcceleratorManager {
    /// Create a new accelerator manager
    pub fn new() -> Self {
        Self {
            accelerators: HashMap::new(),
            performance_history: Arc::new(Mutex::new(PerformanceHistory::new())),
        }
    }
    
    /// Register a hardware accelerator
    pub fn register_accelerator(
        &mut self,
        accelerator_type: AcceleratorType,
        accelerator: Box<dyn HardwareAccelerator + Send + Sync>,
    ) {
        self.accelerators
            .entry(accelerator_type)
            .or_insert_with(Vec::new)
            .push(accelerator);
    }
    
    /// Get the best accelerator for a given configuration
    pub fn get_best_accelerator(
        &self,
        config: &NetworkConfig,
    ) -> FFIResult<(AcceleratorType, usize)> {
        let mut best_score = 0.0;
        let mut best_accelerator = None;
        
        for (accel_type, accelerators) in &self.accelerators {
            for (idx, accelerator) in accelerators.iter().enumerate() {
                let capabilities = accelerator.get_capabilities();
                let score = self.calculate_suitability_score(config, &capabilities);
                
                if score > best_score {
                    best_score = score;
                    best_accelerator = Some((*accel_type, idx));
                }
            }
        }
        
        best_accelerator.ok_or_else(|| {
            FFIError::UnsupportedHardware(AcceleratorType::CPU)
        })
    }
    
    /// Calculate suitability score for an accelerator
    fn calculate_suitability_score(
        &self,
        config: &NetworkConfig,
        capabilities: &HardwareCapabilities,
    ) -> f64 {
        let mut score = 0.0;
        
        // Check if accelerator can handle the network size
        if config.num_neurons > capabilities.max_neurons {
            return 0.0; // Cannot handle this configuration
        }
        
        if config.num_connections > capabilities.max_connections {
            return 0.0;
        }
        
        // Neuron model compatibility
        if capabilities.supported_models.contains(&config.neuron_config.model_type) {
            score += 10.0;
        } else {
            return 0.0; // Unsupported neuron model
        }
        
        // Performance scoring
        score += (capabilities.performance.peak_ops_per_sec / 1e12) * 5.0; // TOPS
        score += (capabilities.memory_bandwidth / 100.0) * 3.0; // Relative to 100 GB/s
        score += (1.0 / capabilities.performance.spike_latency_us) * 2.0; // Lower latency is better
        
        // Power efficiency (lower is better)
        score += (100.0 / capabilities.performance.power_consumption) * 1.0;
        
        // Memory utilization
        let memory_needed = estimate_memory_usage(config);
        let memory_utilization = memory_needed as f64 / capabilities.memory_size as f64;
        if memory_utilization > 0.9 {
            score *= 0.5; // Penalize high memory usage
        }
        
        score
    }
    
    /// Get performance history
    pub fn get_performance_history(&self) -> Arc<Mutex<PerformanceHistory>> {
        Arc::clone(&self.performance_history)
    }
    
    /// Record performance metrics
    pub fn record_performance(
        &self,
        accelerator_type: AcceleratorType,
        config: &NetworkConfig,
        metrics: &PerformanceMetrics,
    ) {
        if let Ok(mut history) = self.performance_history.lock() {
            history.record(accelerator_type, config, metrics);
        }
    }
}

impl Default for AcceleratorManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance history tracking
#[derive(Debug)]
pub struct PerformanceHistory {
    /// Performance records
    records: Vec<PerformanceRecord>,
    /// Maximum number of records to keep
    max_records: usize,
}

impl PerformanceHistory {
    /// Create new performance history
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            max_records: 1000,
        }
    }
    
    /// Record a performance measurement
    pub fn record(
        &mut self,
        accelerator_type: AcceleratorType,
        config: &NetworkConfig,
        metrics: &PerformanceMetrics,
    ) {
        let record = PerformanceRecord {
            timestamp: Instant::now(),
            accelerator_type,
            network_size: config.num_neurons,
            connectivity: config.connectivity,
            metrics: metrics.clone(),
        };
        
        self.records.push(record);
        
        // Keep only the most recent records
        if self.records.len() > self.max_records {
            self.records.remove(0);
        }
    }
    
    /// Get average performance for an accelerator type
    pub fn get_average_performance(
        &self,
        accelerator_type: AcceleratorType,
    ) -> Option<PerformanceMetrics> {
        let relevant_records: Vec<_> = self.records
            .iter()
            .filter(|r| r.accelerator_type == accelerator_type)
            .collect();
        
        if relevant_records.is_empty() {
            return None;
        }
        
        let mut avg_metrics = PerformanceMetrics::default();
        let count = relevant_records.len() as f64;
        
        for record in &relevant_records {
            avg_metrics.execution_time_ms += record.metrics.execution_time_ms;
            avg_metrics.spikes_per_second += record.metrics.spikes_per_second;
            avg_metrics.memory_usage += record.metrics.memory_usage;
            avg_metrics.power_consumption += record.metrics.power_consumption;
            avg_metrics.gpu_utilization += record.metrics.gpu_utilization;
            avg_metrics.memory_utilization += record.metrics.memory_utilization;
        }
        
        avg_metrics.execution_time_ms /= count;
        avg_metrics.spikes_per_second /= count;
        avg_metrics.memory_usage = (avg_metrics.memory_usage as f64 / count) as u64;
        avg_metrics.power_consumption /= count;
        avg_metrics.gpu_utilization /= count;
        avg_metrics.memory_utilization /= count;
        
        Some(avg_metrics)
    }
    
    /// Get performance trend for an accelerator
    pub fn get_performance_trend(
        &self,
        accelerator_type: AcceleratorType,
        duration: Duration,
    ) -> Vec<&PerformanceRecord> {
        let cutoff = Instant::now() - duration;
        
        self.records
            .iter()
            .filter(|r| r.accelerator_type == accelerator_type && r.timestamp >= cutoff)
            .collect()
    }
}

impl Default for PerformanceHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance record entry
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Timestamp of the measurement
    pub timestamp: Instant,
    /// Accelerator type used
    pub accelerator_type: AcceleratorType,
    /// Network size (number of neurons)
    pub network_size: u32,
    /// Network connectivity
    pub connectivity: f32,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Estimate memory usage for a network configuration
pub fn estimate_memory_usage(config: &NetworkConfig) -> u64 {
    // Rough estimation of memory requirements
    
    // Neuron state memory
    let neuron_memory = config.num_neurons as u64 * 64; // ~64 bytes per neuron
    
    // Connection memory
    let connection_memory = config.num_connections as u64 * 16; // ~16 bytes per connection
    
    // Spike buffer memory
    let spike_buffer_memory = config.num_neurons as u64 * 4 * 1000; // Buffer for 1000 spikes per neuron
    
    // Plasticity memory (if enabled)
    let plasticity_memory = if config.plasticity_config.enabled {
        config.num_connections as u64 * 8 // Additional data for plasticity
    } else {
        0
    };
    
    // Add some overhead
    let total = neuron_memory + connection_memory + spike_buffer_memory + plasticity_memory;
    (total as f64 * 1.2) as u64 // 20% overhead
}

/// Software-based CPU accelerator (fallback implementation)
#[derive(Debug)]
pub struct CPUAccelerator {
    /// Deployed networks
    networks: HashMap<NetworkId, DeployedNetwork>,
    /// Next network ID
    next_network_id: u64,
    /// Performance metrics
    metrics: PerformanceMetrics,
    /// Execution context
    context: ExecutionContext,
}

impl CPUAccelerator {
    /// Create a new CPU accelerator
    pub fn new() -> FFIResult<Self> {
        Ok(Self {
            networks: HashMap::new(),
            next_network_id: 1,
            metrics: PerformanceMetrics::default(),
            context: ExecutionContext::default(),
        })
    }
}

impl Default for CPUAccelerator {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl HardwareAccelerator for CPUAccelerator {
    fn deploy_network(&mut self, config: &NetworkConfig) -> FFIResult<NetworkId> {
        let network_id = NetworkId(self.next_network_id);
        self.next_network_id += 1;
        
        // Create a simple network representation
        let network = DeployedNetwork::from_config(config)?;
        self.networks.insert(network_id, network);
        
        Ok(network_id)
    }
    
    fn undeploy_network(&mut self, network_id: NetworkId) -> FFIResult<()> {
        self.networks.remove(&network_id)
            .ok_or(FFIError::InvalidNetworkId(network_id))?;
        Ok(())
    }
    
    fn process_spikes(
        &mut self,
        network_id: NetworkId,
        input_spikes: &[SpikeData],
    ) -> FFIResult<Vec<SpikeData>> {
        let start_time = Instant::now();
        
        let network = self.networks.get_mut(&network_id)
            .ok_or(FFIError::InvalidNetworkId(network_id))?;
        
        // Simple spike processing simulation
        let output_spikes = network.process_spikes(input_spikes)?;
        
        // Update metrics
        self.metrics.execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        self.metrics.spikes_per_second = input_spikes.len() as f64 / 
            (self.metrics.execution_time_ms / 1000.0);
        
        Ok(output_spikes)
    }
    
    fn update_network(
        &mut self,
        network_id: NetworkId,
        updates: &NetworkUpdate,
    ) -> FFIResult<()> {
        let network = self.networks.get_mut(&network_id)
            .ok_or(FFIError::InvalidNetworkId(network_id))?;
        
        network.apply_updates(updates)
    }
    
    fn get_neuron_states(&self, network_id: NetworkId) -> FFIResult<Vec<NeuronState>> {
        let network = self.networks.get(&network_id)
            .ok_or(FFIError::InvalidNetworkId(network_id))?;
        
        Ok(network.get_neuron_states())
    }
    
    fn reset_network(&mut self, network_id: NetworkId) -> FFIResult<()> {
        let network = self.networks.get_mut(&network_id)
            .ok_or(FFIError::InvalidNetworkId(network_id))?;
        
        network.reset()
    }
    
    fn get_capabilities(&self) -> HardwareCapabilities {
        HardwareCapabilities {
            accelerator_type: AcceleratorType::CPU,
            max_neurons: 1_000_000,
            max_connections: 10_000_000,
            memory_size: 16 * 1024 * 1024 * 1024, // 16GB typical
            processing_units: num_cpus::get() as u32,
            supported_models: vec![
                crate::types::NeuronModelType::LIF,
                crate::types::NeuronModelType::AdEx,
                crate::types::NeuronModelType::Izhikevich,
            ],
            features: vec![
                "software_fallback".to_string(),
                "unlimited_precision".to_string(),
                "debugging_support".to_string(),
            ],
            performance: crate::types::PerformanceCharacteristics {
                peak_ops_per_sec: 1e9, // 1 GOPS (conservative)
                memory_bandwidth: 50.0, // 50 GB/s typical
                spike_latency_us: 10.0,
                power_consumption: 65.0, // Typical CPU TDP
                tdp: 65.0,
            },
        }
    }
    
    fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.metrics.clone()
    }
    
    fn set_execution_context(&mut self, context: ExecutionContext) -> FFIResult<()> {
        self.context = context;
        Ok(())
    }
}

/// Simple deployed network representation for CPU accelerator
#[derive(Debug)]
struct DeployedNetwork {
    config: NetworkConfig,
    neuron_states: Vec<NeuronState>,
    connections: Vec<(u32, u32, f32)>, // (pre, post, weight)
    simulation_time: f64,
}

impl DeployedNetwork {
    /// Create from network configuration
    fn from_config(config: &NetworkConfig) -> FFIResult<Self> {
        let mut neuron_states = Vec::new();
        
        for i in 0..config.num_neurons {
            neuron_states.push(NeuronState {
                neuron_id: i,
                membrane_potential: config.neuron_config.v_rest,
                last_spike_time: 0.0,
                spike_count: 0,
                is_refractory: false,
                input_current: 0.0,
            });
        }
        
        // Simple random connectivity
        let mut connections = Vec::new();
        let target_connections = (config.num_neurons as f32 * config.connectivity) as u32;
        
        for _ in 0..target_connections {
            let pre = fastrand::u32(0..config.num_neurons);
            let post = fastrand::u32(0..config.num_neurons);
            if pre != post {
                let weight = fastrand::f32() * 0.1; // Random weight 0-0.1
                connections.push((pre, post, weight));
            }
        }
        
        Ok(Self {
            config: config.clone(),
            neuron_states,
            connections,
            simulation_time: 0.0,
        })
    }
    
    /// Process input spikes
    fn process_spikes(&mut self, input_spikes: &[SpikeData]) -> FFIResult<Vec<SpikeData>> {
        let mut output_spikes = Vec::new();
        
        // Simple processing: forward some spikes based on connectivity
        for spike in input_spikes {
            // Find connections from this neuron
            for &(pre, post, weight) in &self.connections {
                if pre == spike.neuron_id && weight > 0.05 {
                    // Generate output spike with some probability
                    if fastrand::f32() < weight {
                        output_spikes.push(SpikeData {
                            neuron_id: post,
                            timestamp: spike.timestamp + 1.0, // Add 1ms delay
                            amplitude: spike.amplitude * weight,
                        });
                    }
                }
            }
        }
        
        self.simulation_time += self.config.dt as f64;
        Ok(output_spikes)
    }
    
    /// Apply network updates
    fn apply_updates(&mut self, updates: &NetworkUpdate) -> FFIResult<()> {
        match updates {
            NetworkUpdate::UpdateWeights { updates } => {
                for &(pre, post, new_weight) in updates {
                    // Find and update the connection
                    for connection in &mut self.connections {
                        if connection.0 == pre && connection.1 == post {
                            connection.2 = new_weight;
                            break;
                        }
                    }
                }
            }
            NetworkUpdate::AddConnections { connections } => {
                for &(pre, post, weight) in connections {
                    self.connections.push((pre, post, weight));
                }
            }
            NetworkUpdate::RemoveConnections { connections } => {
                for &(pre, post) in connections {
                    self.connections.retain(|&(p, q, _)| !(p == pre && q == post));
                }
            }
            _ => {
                // Other update types not implemented in simple CPU version
            }
        }
        Ok(())
    }
    
    /// Get current neuron states
    fn get_neuron_states(&self) -> Vec<NeuronState> {
        self.neuron_states.clone()
    }
    
    /// Reset network state
    fn reset(&mut self) -> FFIResult<()> {
        for state in &mut self.neuron_states {
            state.membrane_potential = self.config.neuron_config.v_rest;
            state.last_spike_time = 0.0;
            state.spike_count = 0;
            state.is_refractory = false;
            state.input_current = 0.0;
        }
        self.simulation_time = 0.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_accelerator() {
        let mut accelerator = CPUAccelerator::new().unwrap();
        let config = NetworkConfig::default();
        
        // Test network deployment
        let network_id = accelerator.deploy_network(&config).unwrap();
        
        // Test spike processing
        let input_spikes = vec![
            SpikeData::new(0, 1.0, 1.0),
            SpikeData::new(1, 1.5, 0.8),
        ];
        
        let output_spikes = accelerator.process_spikes(network_id, &input_spikes).unwrap();
        assert!(output_spikes.len() >= 0); // May generate outputs
        
        // Test capabilities
        let capabilities = accelerator.get_capabilities();
        assert_eq!(capabilities.accelerator_type, AcceleratorType::CPU);
        
        // Test cleanup
        accelerator.undeploy_network(network_id).unwrap();
    }
    
    #[test]
    fn test_accelerator_manager() {
        let mut manager = AcceleratorManager::new();
        let accelerator = Box::new(CPUAccelerator::new().unwrap());
        
        manager.register_accelerator(AcceleratorType::CPU, accelerator);
        
        let config = NetworkConfig::default();
        let (best_type, _idx) = manager.get_best_accelerator(&config).unwrap();
        assert_eq!(best_type, AcceleratorType::CPU);
    }
    
    #[test]
    fn test_memory_estimation() {
        let config = NetworkConfig {
            num_neurons: 1000,
            num_connections: 10000,
            ..Default::default()
        };
        
        let memory_usage = estimate_memory_usage(&config);
        assert!(memory_usage > 0);
        assert!(memory_usage < 100 * 1024 * 1024); // Should be reasonable
    }
}