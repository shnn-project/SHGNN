//! # SHNN WASM
//!
//! WebAssembly bindings for Spiking Hypergraph Neural Networks.
//!
//! This crate provides high-performance WebAssembly bindings that enable
//! neuromorphic computing directly in web browsers with JavaScript integration.
//!
//! ## Features
//!
//! - **Browser Integration**: Direct JavaScript API for neural networks
//! - **Real-time Processing**: Low-latency spike processing in browsers
//! - **Visualization**: Built-in support for network visualization
//! - **Web Workers**: Parallel processing using Web Workers
//! - **Audio Integration**: Real-time audio processing capabilities
//! - **WebGL Acceleration**: GPU-accelerated computation when available
//!
//! ## JavaScript Usage
//!
//! ```javascript
//! import init, { NeuralNetwork, Neuron, Spike } from './pkg/shnn_wasm.js';
//!
//! async function run() {
//!     await init();
//!     
//!     const network = new NeuralNetwork();
//!     const neuron = new Neuron(0, "LIF");
//!     network.addNeuron(neuron);
//!     
//!     const spike = new Spike(1, 1000, 1.0);
//!     const outputs = network.processSpike(spike);
//!     
//!     console.log(`Processed spike, got ${outputs.length} outputs`);
//! }
//! ```

use wasm_bindgen::prelude::*;
use js_sys::*;
use web_sys::*;

use shnn_core::{
    spike::{NeuronId, Spike as CoreSpike, SpikeTrain as CoreSpikeTrain},
    time::{Time, Duration},
    neuron::{LIFNeuron, LIFConfig, NeuronPool},
    hypergraph::{HypergraphNetwork, Hyperedge, HyperedgeId},
    plasticity::{PlasticityManager, STDPConfig},
    encoding::{RateEncoder, TemporalEncoder, RateEncodingConfig, TemporalEncodingConfig},
    error::SHNNError,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Global allocator for WASM
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    
    console_log!("SHNN WASM module initialized");
}

/// Logging macro for WASM
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => (web_sys::console::log_1(&format_args!($($t)*).to_string().into()))
}

/// Error handling for WASM
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmError {
    message: String,
}

impl From<SHNNError> for WasmError {
    fn from(err: SHNNError) -> Self {
        Self {
            message: err.to_string(),
        }
    }
}

#[wasm_bindgen]
impl WasmError {
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

/// WASM-compatible spike representation
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spike {
    source: u32,
    timestamp: f64,
    amplitude: f32,
}

#[wasm_bindgen]
impl Spike {
    /// Create a new spike
    #[wasm_bindgen(constructor)]
    pub fn new(source: u32, timestamp: f64, amplitude: f32) -> Self {
        Self {
            source,
            timestamp,
            amplitude,
        }
    }
    
    /// Create a binary spike (amplitude = 1.0)
    #[wasm_bindgen]
    pub fn binary(source: u32, timestamp: f64) -> Self {
        Self::new(source, timestamp, 1.0)
    }
    
    /// Get source neuron ID
    #[wasm_bindgen(getter)]
    pub fn source(&self) -> u32 {
        self.source
    }
    
    /// Get timestamp
    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }
    
    /// Get amplitude
    #[wasm_bindgen(getter)]
    pub fn amplitude(&self) -> f32 {
        self.amplitude
    }
    
    /// Convert to core spike
    pub(crate) fn to_core(&self) -> Result<CoreSpike, WasmError> {
        CoreSpike::new(
            NeuronId::new(self.source),
            Time::from_secs_f64(self.timestamp / 1000.0).map_err(|e| WasmError::from(e))?,
            self.amplitude,
        ).map_err(|e| WasmError::from(e))
    }
    
    /// Create from core spike
    pub(crate) fn from_core(spike: &CoreSpike) -> Self {
        Self {
            source: spike.source.raw(),
            timestamp: spike.timestamp.as_secs_f64() * 1000.0,
            amplitude: spike.amplitude,
        }
    }
}

/// WASM-compatible neuron configuration
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronConfig {
    tau_m: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    t_refrac: f32,
}

#[wasm_bindgen]
impl NeuronConfig {
    /// Create new neuron configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let config = LIFConfig::default();
        Self {
            tau_m: config.tau_m,
            v_rest: config.v_rest,
            v_thresh: config.v_thresh,
            v_reset: config.v_reset,
            t_refrac: config.t_refrac,
        }
    }
    
    /// Set membrane time constant
    #[wasm_bindgen]
    pub fn set_tau_m(&mut self, tau_m: f32) {
        self.tau_m = tau_m;
    }
    
    /// Set threshold voltage
    #[wasm_bindgen]
    pub fn set_threshold(&mut self, v_thresh: f32) {
        self.v_thresh = v_thresh;
    }
    
    /// Convert to core config
    pub(crate) fn to_core(&self) -> LIFConfig {
        LIFConfig {
            tau_m: self.tau_m,
            v_rest: self.v_rest,
            v_thresh: self.v_thresh,
            v_reset: self.v_reset,
            t_refrac: self.t_refrac,
            r_m: 10.0,  // Default values
            c_m: 2.0,
        }
    }
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// WASM-compatible neuron
#[wasm_bindgen]
pub struct Neuron {
    id: u32,
    neuron: LIFNeuron,
}

#[wasm_bindgen]
impl Neuron {
    /// Create a new neuron
    #[wasm_bindgen(constructor)]
    pub fn new(id: u32, config: Option<NeuronConfig>) -> Result<Neuron, WasmError> {
        let config = config.unwrap_or_default();
        let core_config = config.to_core();
        
        let neuron = LIFNeuron::new(NeuronId::new(id), core_config)
            .map_err(|e| WasmError::from(e))?;
        
        Ok(Self { id, neuron })
    }
    
    /// Get neuron ID
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> u32 {
        self.id
    }
    
    /// Get membrane potential
    #[wasm_bindgen]
    pub fn membrane_potential(&self) -> f32 {
        self.neuron.membrane_potential()
    }
    
    /// Get threshold
    #[wasm_bindgen]
    pub fn threshold(&self) -> f32 {
        self.neuron.threshold()
    }
    
    /// Set threshold
    #[wasm_bindgen]
    pub fn set_threshold(&mut self, threshold: f32) {
        self.neuron.set_threshold(threshold);
    }
    
    /// Process a spike
    #[wasm_bindgen]
    pub fn process_spike(&mut self, spike: &Spike, current_time: f64) -> Option<Spike> {
        let core_spike = spike.to_core().ok()?;
        let time = Time::from_secs_f64(current_time / 1000.0).ok()?;
        
        self.neuron.process_spike(&core_spike, time)
            .map(|s| Spike::from_core(&s))
    }
    
    /// Reset neuron state
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.neuron.reset();
    }
}

/// WASM-compatible neural network
#[wasm_bindgen]
pub struct NeuralNetwork {
    neurons: NeuronPool,
    hypergraph: HypergraphNetwork,
    plasticity: PlasticityManager,
    current_time: f64,
    spike_history: Vec<Spike>,
}

#[wasm_bindgen]
impl NeuralNetwork {
    /// Create a new neural network
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            neurons: NeuronPool::new(),
            hypergraph: HypergraphNetwork::new(),
            plasticity: PlasticityManager::new(),
            current_time: 0.0,
            spike_history: Vec::new(),
        }
    }
    
    /// Add a neuron to the network
    #[wasm_bindgen]
    pub fn add_neuron(&mut self, id: u32, config: Option<NeuronConfig>) -> Result<(), WasmError> {
        let config = config.unwrap_or_default();
        let core_config = config.to_core();
        
        let neuron = LIFNeuron::new(NeuronId::new(id), core_config)
            .map_err(|e| WasmError::from(e))?;
        
        self.neurons.add_lif_neuron(neuron)
            .map_err(|e| WasmError::from(e))?;
        
        Ok(())
    }
    
    /// Add a connection between neurons
    #[wasm_bindgen]
    pub fn add_connection(
        &mut self,
        source_id: u32,
        target_id: u32,
        weight: f32,
    ) -> Result<(), WasmError> {
        let edge_id = HyperedgeId::new(self.hypergraph.hyperedge_ids().len() as u32);
        
        let hyperedge = Hyperedge::pairwise(
            edge_id,
            NeuronId::new(source_id),
            NeuronId::new(target_id),
            weight,
        ).map_err(|e| WasmError::from(e))?;
        
        self.hypergraph.add_hyperedge(hyperedge)
            .map_err(|e| WasmError::from(e))?;
        
        Ok(())
    }
    
    /// Process a single spike
    #[wasm_bindgen]
    pub fn process_spike(&mut self, spike: &Spike) -> Result<Array, WasmError> {
        let core_spike = spike.to_core()?;
        let time = Time::from_secs_f64(spike.timestamp / 1000.0)
            .map_err(|e| WasmError::from(e))?;
        
        // Route spike through hypergraph
        let routes = self.hypergraph.route_spike(&core_spike, time);
        let mut output_spikes = Vec::new();
        
        // Process each route
        for route in routes {
            for &target_id in &route.targets {
                if let Some(neuron) = self.neurons.get_lif_neuron_mut(target_id) {
                    if let Some(output) = neuron.process_spike(&core_spike, route.delivery_time) {
                        let wasm_spike = Spike::from_core(&output);
                        output_spikes.push(wasm_spike);
                    }
                }
            }
        }
        
        // Add to history
        self.spike_history.push(spike.clone());
        
        // Convert to JavaScript Array
        let js_array = Array::new();
        for spike in output_spikes {
            js_array.push(&JsValue::from_serde(&spike).unwrap());
        }
        
        Ok(js_array)
    }
    
    /// Process a batch of spikes
    #[wasm_bindgen]
    pub fn process_batch(&mut self, spikes: &Array) -> Result<Array, WasmError> {
        let mut all_outputs = Vec::new();
        
        for i in 0..spikes.length() {
            let js_spike = spikes.get(i);
            let spike: Spike = js_spike.into_serde().map_err(|_| WasmError {
                message: "Failed to deserialize spike".to_string(),
            })?;
            
            let outputs = self.process_spike(&spike)?;
            for j in 0..outputs.length() {
                all_outputs.push(outputs.get(j));
            }
        }
        
        let js_array = Array::new();
        for output in all_outputs {
            js_array.push(&output);
        }
        
        Ok(js_array)
    }
    
    /// Update network time
    #[wasm_bindgen]
    pub fn update_time(&mut self, new_time: f64) {
        let dt = Duration::from_secs_f64((new_time - self.current_time) / 1000.0)
            .unwrap_or(Duration::from_millis(1));
        let time = Time::from_secs_f64(new_time / 1000.0)
            .unwrap_or(Time::ZERO);
        
        self.neurons.update_all(time, dt);
        self.current_time = new_time;
    }
    
    /// Get neuron count
    #[wasm_bindgen]
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }
    
    /// Get connection count
    #[wasm_bindgen]
    pub fn connection_count(&self) -> usize {
        self.hypergraph.hyperedge_ids().len()
    }
    
    /// Get spike history
    #[wasm_bindgen]
    pub fn spike_history(&self) -> Array {
        let js_array = Array::new();
        for spike in &self.spike_history {
            js_array.push(&JsValue::from_serde(spike).unwrap());
        }
        js_array
    }
    
    /// Clear spike history
    #[wasm_bindgen]
    pub fn clear_history(&mut self) {
        self.spike_history.clear();
    }
    
    /// Reset all neurons
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.neurons.reset_all();
        self.current_time = 0.0;
        self.spike_history.clear();
    }
    
    /// Get network statistics as JSON
    #[wasm_bindgen]
    pub fn get_stats(&self) -> JsValue {
        let stats = NetworkStats {
            neuron_count: self.neuron_count(),
            connection_count: self.connection_count(),
            current_time: self.current_time,
            spike_count: self.spike_history.len(),
        };
        
        JsValue::from_serde(&stats).unwrap()
    }
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Network statistics for JavaScript
#[derive(Serialize, Deserialize)]
struct NetworkStats {
    neuron_count: usize,
    connection_count: usize,
    current_time: f64,
    spike_count: usize,
}

/// Spike encoder for converting analog signals to spikes
#[wasm_bindgen]
pub struct SpikeEncoder {
    encoder_type: String,
    rate_encoder: RateEncoder,
    temporal_encoder: TemporalEncoder,
}

#[wasm_bindgen]
impl SpikeEncoder {
    /// Create a new spike encoder
    #[wasm_bindgen(constructor)]
    pub fn new(encoder_type: &str) -> Self {
        Self {
            encoder_type: encoder_type.to_string(),
            rate_encoder: RateEncoder::new(),
            temporal_encoder: TemporalEncoder::new(),
        }
    }
    
    /// Encode a value to spikes
    #[wasm_bindgen]
    pub fn encode(&self, value: f32, timestamp: f64, neuron_ids: &Array) -> Result<Array, WasmError> {
        let ids: Vec<NeuronId> = (0..neuron_ids.length())
            .map(|i| NeuronId::new(neuron_ids.get(i).as_f64().unwrap() as u32))
            .collect();
        
        let time = Time::from_secs_f64(timestamp / 1000.0)
            .map_err(|e| WasmError::from(e))?;
        
        let spike_trains = match self.encoder_type.as_str() {
            "rate" => {
                let config = RateEncodingConfig::default();
                self.rate_encoder.encode_value(value, time, &ids, &config)
                    .map_err(|e| WasmError::from(e))?
            }
            "temporal" => {
                let config = TemporalEncodingConfig::default();
                self.temporal_encoder.encode_value(value, time, &ids, &config)
                    .map_err(|e| WasmError::from(e))?
            }
            _ => return Err(WasmError {
                message: "Unknown encoder type".to_string(),
            }),
        };
        
        // Convert spike trains to JavaScript array
        let js_array = Array::new();
        for train in spike_trains {
            let spikes = train.to_spikes().map_err(|e| WasmError::from(e))?;
            for spike in spikes {
                let wasm_spike = Spike::from_core(&spike);
                js_array.push(&JsValue::from_serde(&wasm_spike).unwrap());
            }
        }
        
        Ok(js_array)
    }
}

/// Performance monitoring utilities
#[wasm_bindgen]
pub struct PerformanceMonitor {
    start_time: f64,
    spike_count: u32,
}

#[wasm_bindgen]
impl PerformanceMonitor {
    /// Create a new performance monitor
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        
        Self {
            start_time: performance.now(),
            spike_count: 0,
        }
    }
    
    /// Record a processed spike
    #[wasm_bindgen]
    pub fn record_spike(&mut self) {
        self.spike_count += 1;
    }
    
    /// Get processing rate (spikes per second)
    #[wasm_bindgen]
    pub fn get_rate(&self) -> f64 {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        let elapsed = (performance.now() - self.start_time) / 1000.0;
        
        if elapsed > 0.0 {
            self.spike_count as f64 / elapsed
        } else {
            0.0
        }
    }
    
    /// Reset monitor
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        
        self.start_time = performance.now();
        self.spike_count = 0;
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for JavaScript integration
#[wasm_bindgen]
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get WASM module version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Log a message to the browser console
#[wasm_bindgen]
pub fn log(message: &str) {
    console_log!("{}", message);
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    wasm_bindgen_test_configure!(run_in_browser);
    
    #[wasm_bindgen_test]
    fn test_spike_creation() {
        let spike = Spike::new(1, 1000.0, 1.0);
        assert_eq!(spike.source(), 1);
        assert_eq!(spike.timestamp(), 1000.0);
        assert_eq!(spike.amplitude(), 1.0);
    }
    
    #[wasm_bindgen_test]
    fn test_neuron_creation() {
        let config = NeuronConfig::new();
        let neuron = Neuron::new(0, Some(config));
        assert!(neuron.is_ok());
    }
    
    #[wasm_bindgen_test]
    fn test_network_creation() {
        let network = NeuralNetwork::new();
        assert_eq!(network.neuron_count(), 0);
        assert_eq!(network.connection_count(), 0);
    }
    
    #[wasm_bindgen_test]
    fn test_network_add_neuron() {
        let mut network = NeuralNetwork::new();
        let config = NeuronConfig::new();
        let result = network.add_neuron(0, Some(config));
        assert!(result.is_ok());
        assert_eq!(network.neuron_count(), 1);
    }
}