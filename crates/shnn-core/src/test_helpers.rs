//! Comprehensive test infrastructure and helpers for SHNN testing
//! 
//! This module provides test builders, custom assertions, mock implementations,
//! and utility functions to support robust testing across the SHNN codebase.

#[cfg(test)]
use crate::{
    error::{Result, SHNNError},
    neuron::{LIFNeuron, AdExNeuron, IzhikevichNeuron, Neuron, NeuronState},
    spike::{Spike, NeuronId, TimedSpike, SpikeTrain, SpikeTarget},
    time::{Time, Duration, TimeStep, TimeWindow},
    memory::{SpikeBuffer, NeuronPool},
    serialization::NetworkSnapshot,
    hypergraph::HyperedgeId,
    async_processing::{AsyncNeuralNetwork, ProcessingStats},
};

#[cfg(test)]
use std::collections::{HashMap, VecDeque};

// =============================================================================
// Test Data Builders
// =============================================================================

#[cfg(test)]
pub struct NeuronTestBuilder {
    id: NeuronId,
    threshold: f64,
    tau_membrane: f64,
    resting_potential: f64,
    reset_potential: f64,
}

#[cfg(test)]
impl NeuronTestBuilder {
    pub fn new() -> Self {
        Self {
            id: NeuronId::new(0),
            threshold: -55.0,
            tau_membrane: 20.0,
            resting_potential: -70.0,
            reset_potential: -70.0,
        }
    }
    
    pub fn with_id(mut self, id: u32) -> Self {
        self.id = NeuronId::new(id);
        self
    }
    
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
    
    pub fn with_tau_membrane(mut self, tau: f64) -> Self {
        self.tau_membrane = tau;
        self
    }
    
    pub fn with_resting_potential(mut self, potential: f64) -> Self {
        self.resting_potential = potential;
        self
    }
    
    pub fn with_reset_potential(mut self, potential: f64) -> Self {
        self.reset_potential = potential;
        self
    }
    
    pub fn build_lif(self) -> LIFNeuron {
        let mut neuron = LIFNeuron::new(self.id);
        neuron.tau_membrane = self.tau_membrane;
        neuron.threshold = self.threshold;
        neuron.resting_potential = self.resting_potential;
        neuron.reset_potential = self.reset_potential;
        neuron
    }
}

#[cfg(test)]
pub struct SpikeTestBuilder {
    source: NeuronId,
    amplitude: f32,
    timestamp: Time,
}

#[cfg(test)]
impl SpikeTestBuilder {
    pub fn new() -> Self {
        Self {
            source: NeuronId::new(0),
            amplitude: 1.0,
            timestamp: Time::from_millis(0),
        }
    }
    
    pub fn with_source(mut self, id: u32) -> Self {
        self.source = NeuronId::new(id);
        self
    }
    
    pub fn with_amplitude(mut self, amplitude: f64) -> Self {
        self.amplitude = amplitude as f32;
        self
    }
    
    pub fn with_timestamp_ms(mut self, ms: u64) -> Self {
        self.timestamp = Time::from_millis(ms);
        self
    }
    
    pub fn build(self) -> Result<Spike> {
        Spike::new(self.source, self.timestamp, self.amplitude)
    }
    
    pub fn build_binary(self) -> Result<Spike> {
        Spike::binary(self.source, self.timestamp)
    }
}

#[cfg(test)]
pub struct NetworkTestBuilder {
    neurons: Vec<LIFNeuron>,
    spikes: Vec<Spike>,
    connections: Vec<(NeuronId, NeuronId, f64)>,
}

#[cfg(test)]
impl NetworkTestBuilder {
    pub fn new() -> Self {
        Self {
            neurons: Vec::new(),
            spikes: Vec::new(),
            connections: Vec::new(),
        }
    }
    
    pub fn add_neuron(mut self, neuron: LIFNeuron) -> Self {
        self.neurons.push(neuron);
        self
    }
    
    pub fn add_spike(mut self, spike: Spike) -> Self {
        self.spikes.push(spike);
        self
    }
    
    pub fn add_connection(mut self, from: NeuronId, to: NeuronId, weight: f64) -> Self {
        self.connections.push((from, to, weight));
        self
    }
    
    pub fn build_async_network(self) -> AsyncNeuralNetwork {
        let mut network = AsyncNeuralNetwork::new();
        for neuron in self.neurons {
            let _ = network.add_neuron(neuron);
        }
        network
    }
}

// =============================================================================
// Custom Assertion Helpers
// =============================================================================

#[cfg(test)]
pub mod test_matchers {
    use super::*;
    
    pub fn assert_spike_eq(actual: &Spike, expected: &Spike) {
        assert_eq!(actual.source, expected.source, "Spike source mismatch");
        assert_eq!(actual.timestamp, expected.timestamp, "Spike timestamp mismatch");
        assert!((actual.amplitude - expected.amplitude).abs() < f32::EPSILON,
                "Spike amplitude mismatch: expected {}, got {}",
                expected.amplitude, actual.amplitude);
    }
    
    pub fn assert_neuron_voltage_approx(neuron: &LIFNeuron, expected_voltage: f64, tolerance: f64) {
        let actual = neuron.membrane_potential();
        assert!(
            (actual - expected_voltage).abs() < tolerance,
            "Expected voltage ~{} ± {}, got {}", expected_voltage, tolerance, actual
        );
    }
    
    pub fn assert_spike_train_timing(train: &SpikeTrain, expected_intervals: &[Duration]) {
        let intervals = train.inter_spike_intervals();
        assert_eq!(intervals.len(), expected_intervals.len(),
                   "Interval count mismatch");
        for (i, (actual, expected)) in intervals.iter().zip(expected_intervals).enumerate() {
            let diff = if *actual > *expected {
                *actual - *expected
            } else {
                *expected - *actual
            };
            assert!(diff < Duration::from_micros(1),
                    "Interval {} mismatch: expected {:?}, got {:?}", i, expected, actual);
        }
    }
    
    pub fn assert_time_approx(actual: Time, expected: Time, tolerance: Duration) {
        let diff = if actual > expected { 
            actual.elapsed_since(expected) 
        } else { 
            expected.elapsed_since(actual) 
        };
        assert!(diff <= tolerance, 
                "Time mismatch: expected {:?} ± {:?}, got {:?}", 
                expected, tolerance, actual);
    }
    
    pub fn assert_duration_approx(actual: Duration, expected: Duration, tolerance: Duration) {
        let diff = if actual > expected {
            actual - expected
        } else {
            expected - actual
        };
        assert!(diff <= tolerance,
                "Duration mismatch: expected {:?} ± {:?}, got {:?}",
                expected, tolerance, actual);
    }
}

// =============================================================================
// Validation Helpers
// =============================================================================

#[cfg(test)]
pub mod validators {
    use super::*;
    
    pub fn validate_network_state(snapshot: &NetworkSnapshot) -> Result<()> {
        if snapshot.version == 0 {
            return Err(SHNNError::invalid_neuron_config(0, "Invalid network version"));
        }
        
        for (id, state) in &snapshot.neuron_states {
            if state.membrane_potential.is_nan() {
                return Err(SHNNError::invalid_neuron_config(id.0, "NaN voltage in neuron"));
            }
            
            if state.membrane_potential.is_infinite() {
                return Err(SHNNError::invalid_neuron_config(id.0, "Infinite voltage in neuron"));
            }
        }
        
        Ok(())
    }
    
    pub fn validate_spike_timing(spikes: &[TimedSpike]) -> bool {
        spikes.windows(2).all(|pair| pair[0].delivery_time <= pair[1].delivery_time)
    }
    
    pub fn validate_neuron_parameters(neuron: &LIFNeuron) -> Result<()> {
        if neuron.tau_membrane <= 0.0 {
            return Err(SHNNError::invalid_neuron_config(neuron.id().0, "tau_membrane must be positive"));
        }
        if neuron.threshold <= neuron.resting_potential {
            return Err(SHNNError::invalid_neuron_config(
                neuron.id().0, "threshold must be above resting potential"
            ));
        }
        if neuron.reset_potential > neuron.threshold {
            return Err(SHNNError::invalid_neuron_config(
                neuron.id().0, "reset potential must be below threshold"
            ));
        }
        Ok(())
    }
    
    pub fn validate_spike_amplitude(spike: &Spike, min_amp: f32, max_amp: f32) -> bool {
        spike.amplitude >= min_amp && spike.amplitude <= max_amp && spike.amplitude.is_finite()
    }
    
    pub fn validate_time_ordering(times: &[Time]) -> bool {
        times.windows(2).all(|pair| pair[0] <= pair[1])
    }
}

// =============================================================================
// Mock Implementations
// =============================================================================

#[cfg(test)]
pub struct MockNeuron {
    id: NeuronId,
    voltage: f64,
    spike_count: usize,
    spike_threshold: f64,
    reset_voltage: f64,
}

#[cfg(test)]
impl MockNeuron {
    pub fn new(id: NeuronId) -> Self {
        Self {
            id,
            voltage: -70.0,
            spike_count: 0,
            spike_threshold: -55.0,
            reset_voltage: -70.0,
        }
    }
    
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.spike_threshold = threshold;
        self
    }
    
    pub fn spike_count(&self) -> usize {
        self.spike_count
    }
}

#[cfg(test)]
impl Neuron for MockNeuron {
    fn integrate(&mut self, input_current: f64, dt: TimeStep) {
        self.voltage += input_current * (dt as f64) / 1000.0;
    }
    
    fn update(&mut self, _dt: TimeStep) -> Option<Spike> {
        if self.voltage > self.spike_threshold {
            self.spike_count += 1;
            self.voltage = self.reset_voltage;
            Spike::binary(self.id, Time::from_millis(0)).ok()
        } else {
            None
        }
    }
    
    fn membrane_potential(&self) -> f64 { 
        self.voltage 
    }
    
    fn set_membrane_potential(&mut self, voltage: f64) { 
        self.voltage = voltage; 
    }
    
    fn threshold(&self) -> f64 { 
        self.spike_threshold 
    }
    
    fn reset(&mut self) { 
        self.voltage = self.reset_voltage; 
        self.spike_count = 0;
    }
    
    fn id(&self) -> NeuronId { 
        self.id 
    }
    
    fn set_id(&mut self, id: NeuronId) { 
        self.id = id; 
    }
}

// =============================================================================
// Test Configuration Helpers
// =============================================================================

#[cfg(test)]
pub struct TestConfig {
    pub neuron_count: usize,
    pub simulation_time: Duration,
    pub time_step: TimeStep,
    pub spike_rate: f64,
    pub connection_probability: f64,
}

#[cfg(test)]
impl TestConfig {
    pub fn default() -> Self {
        Self {
            neuron_count: 10,
            simulation_time: Duration::from_millis(100),
            time_step: 1000, // 1ms in microseconds
            spike_rate: 10.0, // Hz
            connection_probability: 0.1,
        }
    }
    
    pub fn fast_simulation() -> Self {
        Self {
            neuron_count: 5,
            simulation_time: Duration::from_millis(10),
            time_step: 100, // 0.1ms
            spike_rate: 5.0,
            connection_probability: 0.2,
        }
    }
    
    pub fn large_network() -> Self {
        Self {
            neuron_count: 1000,
            simulation_time: Duration::from_secs(1),
            time_step: 1000,
            spike_rate: 20.0,
            connection_probability: 0.05,
        }
    }
    
    pub fn minimal() -> Self {
        Self {
            neuron_count: 2,
            simulation_time: Duration::from_millis(1),
            time_step: 100,
            spike_rate: 1.0,
            connection_probability: 1.0,
        }
    }
}

// =============================================================================
// Cleanup and Resource Management
// =============================================================================

#[cfg(test)]
pub struct TestGuard<F: FnOnce()> {
    cleanup: Option<F>,
}

#[cfg(test)]
impl<F: FnOnce()> TestGuard<F> {
    pub fn new(cleanup: F) -> Self {
        Self { 
            cleanup: Some(cleanup) 
        }
    }
}

#[cfg(test)]
impl<F: FnOnce()> Drop for TestGuard<F> {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }
    }
}

// =============================================================================
// Test Data Generation
// =============================================================================

#[cfg(test)]
pub mod data_generators {
    use super::*;
    
    pub fn generate_regular_spike_train(
        neuron_id: NeuronId, 
        interval: Duration, 
        count: usize
    ) -> Result<SpikeTrain> {
        let timestamps: Vec<Time> = (0..count)
            .map(|i| Time::from_nanos(i as u64 * interval.as_nanos()))
            .collect();
        SpikeTrain::binary(neuron_id, timestamps)
    }
    
    pub fn generate_poisson_spike_train(
        neuron_id: NeuronId,
        rate: f64,
        duration: Duration,
        seed: u64,
    ) -> Result<SpikeTrain> {
        // Simple deterministic approximation for testing
        let avg_interval = Duration::from_secs_f64(1.0 / rate).unwrap();
        let count = (duration.as_secs_f64() * rate) as usize;
        generate_regular_spike_train(neuron_id, avg_interval, count)
    }
    
    pub fn generate_test_network(config: &TestConfig) -> Vec<LIFNeuron> {
        (0..config.neuron_count)
            .map(|i| NeuronTestBuilder::new()
                .with_id(i as u32)
                .with_threshold(-55.0 + (i as f64 * 0.1)) // Slight variation
                .build_lif())
            .collect()
    }
    
    pub fn generate_spike_burst(
        neuron_id: NeuronId,
        start_time: Time,
        burst_size: usize,
        inter_spike_interval: Duration,
    ) -> Result<Vec<Spike>> {
        (0..burst_size)
            .map(|i| {
                let timestamp = start_time + Duration::from_nanos(i as u64 * inter_spike_interval.as_nanos());
                Spike::binary(neuron_id, timestamp)
            })
            .collect()
    }
}

// =============================================================================
// Performance and Profiling Helpers
// =============================================================================

#[cfg(test)]
pub mod performance {
    use super::*;
    use std::time::Instant;
    
    pub struct BenchmarkResult {
        pub duration: std::time::Duration,
        pub operations_per_second: f64,
        pub memory_used: usize,
    }
    
    pub fn benchmark_function<F, R>(f: F) -> (R, BenchmarkResult)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        
        let bench_result = BenchmarkResult {
            duration,
            operations_per_second: 1.0 / duration.as_secs_f64(),
            memory_used: 0, // Simplified for now
        };
        
        (result, bench_result)
    }
    
    pub fn assert_performance_bounds(
        result: &BenchmarkResult,
        max_duration_ms: u64,
        min_ops_per_sec: f64,
    ) {
        assert!(
            result.duration.as_millis() <= max_duration_ms as u128,
            "Operation took {}ms, expected ≤{}ms",
            result.duration.as_millis(),
            max_duration_ms
        );
        
        assert!(
            result.operations_per_second >= min_ops_per_sec,
            "Performance {:.2} ops/sec, expected ≥{:.2} ops/sec",
            result.operations_per_second,
            min_ops_per_sec
        );
    }
}

// =============================================================================
// Error Handling Test Helpers
// =============================================================================

#[cfg(test)]
pub mod error_helpers {
    use super::*;
    
    pub fn assert_error_type<T, E: std::error::Error>(
        result: Result<T, E>,
        expected_msg_contains: &str
    ) {
        match result {
            Err(e) => assert!(
                e.to_string().contains(expected_msg_contains),
                "Error message '{}' does not contain '{}'",
                e.to_string(),
                expected_msg_contains
            ),
            Ok(_) => panic!("Expected error but got success"),
        }
    }
    
    pub fn assert_shnn_error(result: Result<()>, expected_type: &str) {
        match result {
            Err(e) => assert!(
                e.to_string().contains(expected_type),
                "Expected error type '{}', got '{}'",
                expected_type,
                e.to_string()
            ),
            Ok(_) => panic!("Expected SHNN error but got success"),
        }
    }
}

// =============================================================================
// Test Fixtures and Constants
// =============================================================================

#[cfg(test)]
pub mod fixtures {
    use super::*;
    
    pub const TEST_NEURON_ID: u32 = 42;
    pub const TEST_THRESHOLD: f64 = -55.0;
    pub const TEST_TAU_MEMBRANE: f64 = 20.0;
    pub const TEST_RESTING_POTENTIAL: f64 = -70.0;
    pub const TEST_SPIKE_AMPLITUDE: f64 = 1.0;
    
    pub fn create_default_lif_neuron() -> LIFNeuron {
        NeuronTestBuilder::new()
            .with_id(TEST_NEURON_ID)
            .with_threshold(TEST_THRESHOLD)
            .with_tau_membrane(TEST_TAU_MEMBRANE)
            .with_resting_potential(TEST_RESTING_POTENTIAL)
            .build_lif()
    }
    
    pub fn create_test_spike() -> Spike {
        SpikeTestBuilder::new()
            .with_source(0)
            .with_amplitude(TEST_SPIKE_AMPLITUDE)
            .with_timestamp_ms(100)
            .build()
            .expect("Failed to create test spike")
    }
    
    pub fn create_test_time_window() -> TimeWindow {
        TimeWindow::new(
            Time::from_millis(0),
            Time::from_millis(100)
        ).expect("Failed to create test time window")
    }
}

            .with_resting_potential(TEST_RESTING_POTENTIAL)
            .build_lif()
    }
    
    pub fn create_test_spike() -> Spike {
        SpikeTestBuilder::new()
            .with_source(0)
            .with_target(1)
            .with_amplitude(TEST_SPIKE_AMPLITUDE)
            .with_timestamp_ms(100)
            .build()
            .expect("Failed to create test spike")
    }
    
    pub fn create_test_time_window() -> TimeWindow {
        TimeWindow::new(
            Time::from_millis(0),
            Time::from_millis(100)
        ).expect("Failed to create test time window")
    }
}