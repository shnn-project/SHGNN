//! Neuron models and dynamics for spiking neural networks
//! 
//! This module provides various biologically-inspired neuron models including
//! Leaky Integrate-and-Fire (LIF), Adaptive Exponential (AdEx), and Izhikevich neurons.
//! Each model implements the `Neuron` trait for consistent behavior across the framework.

use crate::spike::Spike;
use crate::error::{SHNNError, Result};
use crate::math::{exp_approx, safe_divide};
use crate::time::TimeStep;

#[cfg(not(feature = "std"))]
use libm::{exp, log};

// Re-export the canonical NeuronId from spike module to ensure type consistency
pub use crate::spike::NeuronId;

// Type alias for backward compatibility with usize-based APIs
pub type NeuronIdUsize = usize;

/// Enumeration of available neuron types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NeuronType {
    /// Leaky Integrate-and-Fire neuron
    LIF,
    /// Adaptive Exponential Integrate-and-Fire neuron
    AdEx,
    /// Izhikevich neuron model
    Izhikevich,
}

impl Default for NeuronType {
    fn default() -> Self {
        Self::LIF
    }
}

/// Collection of neurons for efficient management
#[derive(Debug, Clone)]
pub struct NeuronPool<T: Neuron> {
    neurons: Vec<T>,
    active_indices: Vec<usize>,
}

impl<T: Neuron> NeuronPool<T> {
    /// Create a new empty neuron pool
    pub fn new() -> Self {
        Self {
            neurons: Vec::new(),
            active_indices: Vec::new(),
        }
    }
    
    /// Create a neuron pool with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            neurons: Vec::with_capacity(capacity),
            active_indices: Vec::with_capacity(capacity),
        }
    }
    
    /// Add a neuron to the pool
    pub fn add_neuron(&mut self, neuron: T) -> usize {
        let index = self.neurons.len();
        self.neurons.push(neuron);
        self.active_indices.push(index);
        index
    }
    
    /// Get a reference to a neuron by index
    pub fn get_neuron(&self, index: usize) -> Option<&T> {
        self.neurons.get(index)
    }
    
    /// Get a mutable reference to a neuron by index
    pub fn get_neuron_mut(&mut self, index: usize) -> Option<&mut T> {
        self.neurons.get_mut(index)
    }
    
    /// Get the number of neurons in the pool
    pub fn len(&self) -> usize {
        self.neurons.len()
    }
    
    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }
    
    /// Get iterator over all neurons
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.neurons.iter()
    }
    
    /// Get mutable iterator over all neurons
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.neurons.iter_mut()
    }
    
    /// Update all neurons and collect generated spikes
    pub fn update_all(&mut self, dt: TimeStep) -> Vec<(usize, Spike)> {
        let mut spikes = Vec::new();
        for (index, neuron) in self.neurons.iter_mut().enumerate() {
            if let Some(spike) = neuron.update(dt) {
                spikes.push((index, spike));
            }
        }
        spikes
    }
}

impl<T: Neuron> Default for NeuronPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Conversion utilities for type safety
impl From<usize> for NeuronId {
    fn from(id: usize) -> Self {
        Self::new(id as u32)
    }
}

impl From<NeuronId> for usize {
    fn from(id: NeuronId) -> Self {
        id.raw() as usize
    }
}

/// Current state of a neuron including membrane potential and internal variables
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronState {
    pub membrane_potential: f64,
    pub refractory_timer: TimeStep,
    pub last_spike_time: Option<TimeStep>,
}

impl NeuronState {
    pub fn new() -> Self {
        Self {
            membrane_potential: -65.0, // Typical resting potential
            refractory_timer: TimeStep::zero(),
            last_spike_time: None,
        }
    }
    
    pub fn membrane_potential(&self) -> f64 {
        self.membrane_potential
    }
    
    pub fn is_refractory(&self) -> bool {
        self.refractory_timer > TimeStep::zero()
    }
}

impl Default for NeuronState {
    fn default() -> Self {
        Self::new()
    }
}

/// Core trait for all neuron models
pub trait Neuron: Send + Sync + Clone {
    /// Integrate input current over time step
    fn integrate(&mut self, input_current: f64, dt: TimeStep);
    
    /// Update neuron state and check for spike generation
    fn update(&mut self, dt: TimeStep) -> Option<Spike>;
    
    /// Get current membrane potential
    fn membrane_potential(&self) -> f64;
    
    /// Set membrane potential (for testing/initialization)
    fn set_membrane_potential(&mut self, voltage: f64);
    
    /// Get spike threshold
    fn threshold(&self) -> f64;
    
    /// Reset neuron to post-spike state
    fn reset(&mut self);
    
    /// Get neuron's unique identifier
    fn id(&self) -> NeuronId;
    
    /// Set neuron's identifier
    fn set_id(&mut self, id: NeuronId);
}

/// Leaky Integrate-and-Fire neuron model
/// 
/// The LIF model is the simplest spiking neuron model, where the membrane potential
/// integrates input current with exponential decay (leak).
#[derive(Debug, Clone, PartialEq)]
pub struct LIFNeuron {
    id: NeuronId,
    state: NeuronState,
    
    // Parameters
    pub tau_membrane: f64,      // Membrane time constant (ms)
    pub resistance: f64,        // Membrane resistance (MΩ)
    pub capacitance: f64,       // Membrane capacitance (nF)
    pub threshold: f64,         // Spike threshold (mV)
    pub reset_potential: f64,   // Reset potential (mV)
    pub resting_potential: f64, // Resting potential (mV)
    pub refractory_period: f64, // Refractory period (ms)
}

impl LIFNeuron {
    pub fn new(id: NeuronId) -> Self {
        Self {
            id,
            state: NeuronState::new(),
            tau_membrane: 20.0,      // 20ms time constant
            resistance: 10.0,        // 10 MΩ resistance  
            capacitance: 2.0,        // 2 nF capacitance
            threshold: -55.0,        // -55mV threshold
            reset_potential: -70.0,  // -70mV reset
            resting_potential: -65.0, // -65mV resting
            refractory_period: 2.0,   // 2ms refractory
        }
    }
    
    pub fn with_params(
        id: NeuronId,
        tau_membrane: f64,
        threshold: f64,
        reset_potential: f64,
        refractory_period: f64,
    ) -> Self {
        let mut neuron = Self::new(id);
        neuron.tau_membrane = tau_membrane;
        neuron.threshold = threshold;
        neuron.reset_potential = reset_potential;
        neuron.refractory_period = refractory_period;
        neuron
    }
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self::new(NeuronId(0))
    }
}

impl Neuron for LIFNeuron {
    fn integrate(&mut self, input_current: f64, dt: TimeStep) {
        if self.state.is_refractory() {
            // Update refractory timer
            self.state.refractory_timer = self.state.refractory_timer - dt;
            if self.state.refractory_timer < TimeStep::zero() {
                self.state.refractory_timer = TimeStep::zero();
            }
            return;
        }
        
        let dt_ms = dt.as_ms();
        
        // Membrane equation: dV/dt = (V_rest - V)/tau + I*R/tau
        let leak_current = (self.resting_potential - self.state.membrane_potential) / self.tau_membrane;
        let input_term = input_current * self.resistance / self.tau_membrane;
        
        let dv_dt = leak_current + input_term;
        self.state.membrane_potential += dv_dt * dt_ms;
    }
    
    fn update(&mut self, dt: TimeStep) -> Option<Spike> {
        if self.state.membrane_potential >= self.threshold {
            self.reset();
            self.state.last_spike_time = Some(TimeStep::zero()); // Would need current time
            self.state.refractory_timer = TimeStep::from_ms(self.refractory_period);
            
            // Create spike with proper type conversion and error handling
            match Spike::new(
                self.id.into(),
                crate::time::Time::from_nanos(0), // Convert TimeStep to Time
                1.0 // Default spike amplitude
            ) {
                Ok(spike) => Some(spike),
                Err(_) => None, // Log error in production code
            }
        } else {
            None
        }
    }
    
    fn membrane_potential(&self) -> f64 {
        self.state.membrane_potential
    }
    
    fn set_membrane_potential(&mut self, voltage: f64) {
        self.state.membrane_potential = voltage;
    }
    
    fn threshold(&self) -> f64 {
        self.threshold
    }
    
    fn reset(&mut self) {
        self.state.membrane_potential = self.reset_potential;
    }
    
    fn id(&self) -> NeuronId {
        self.id
    }
    
    fn set_id(&mut self, id: NeuronId) {
        self.id = id;
    }
}

/// Adaptive Exponential Integrate-and-Fire neuron model
/// 
/// The AdEx model includes an exponential term and adaptation current,
/// providing more realistic spike generation and frequency adaptation.
#[derive(Debug, Clone, PartialEq)]
pub struct AdExNeuron {
    id: NeuronId,
    state: NeuronState,
    adaptation_current: f64,
    
    // Parameters
    pub tau_membrane: f64,      // Membrane time constant (ms)
    pub tau_adaptation: f64,    // Adaptation time constant (ms)
    pub delta_t: f64,           // Slope factor (mV)
    pub conductance: f64,       // Leak conductance (nS)
    pub capacitance: f64,       // Membrane capacitance (pF)
    pub threshold: f64,         // Spike threshold (mV)
    pub reset_potential: f64,   // Reset potential (mV)
    pub resting_potential: f64, // Resting potential (mV)
    pub adaptation_increment: f64, // Spike-triggered adaptation increment (pA)
    pub refractory_period: f64, // Refractory period (ms)
}

impl AdExNeuron {
    pub fn new(id: NeuronId) -> Self {
        Self {
            id,
            state: NeuronState::new(),
            adaptation_current: 0.0,
            tau_membrane: 9.3,       // 9.3ms membrane time constant
            tau_adaptation: 144.0,   // 144ms adaptation time constant
            delta_t: 2.0,            // 2mV slope factor
            conductance: 30.0,       // 30nS leak conductance
            capacitance: 281.0,      // 281pF capacitance
            threshold: -50.4,        // -50.4mV threshold
            reset_potential: -70.6,  // -70.6mV reset
            resting_potential: -70.6, // -70.6mV resting
            adaptation_increment: 4.0, // 4pA adaptation increment
            refractory_period: 2.0,   // 2ms refractory
        }
    }
    
    pub fn adaptation_current(&self) -> f64 {
        self.adaptation_current
    }
}

impl Default for AdExNeuron {
    fn default() -> Self {
        Self::new(NeuronId(0))
    }
}

impl Neuron for AdExNeuron {
    fn integrate(&mut self, input_current: f64, dt: TimeStep) {
        if self.state.is_refractory() {
            self.state.refractory_timer = self.state.refractory_timer - dt;
            if self.state.refractory_timer < TimeStep::zero() {
                self.state.refractory_timer = TimeStep::zero();
            }
            return;
        }
        
        let dt_ms = dt.as_ms();
        let v = self.state.membrane_potential;
        
        // Exponential term for spike generation
        let exp_term = if v - self.threshold < 10.0 { // Avoid overflow
            self.delta_t * exp_approx((v - self.threshold) / self.delta_t)
        } else {
            self.delta_t * 1000.0 // Large value to trigger spike
        };
        
        // Membrane equation with exponential term
        let leak_current = self.conductance * (self.resting_potential - v);
        let adaptation_term = -self.adaptation_current;
        let exponential_current = self.conductance * exp_term;
        
        let dv_dt = (leak_current + adaptation_term + exponential_current + input_current) / self.capacitance;
        
        // Update membrane potential
        self.state.membrane_potential += dv_dt * dt_ms;
        
        // Update adaptation current
        let da_dt = -self.adaptation_current / self.tau_adaptation;
        self.adaptation_current += da_dt * dt_ms;
    }
    
    fn update(&mut self, dt: TimeStep) -> Option<Spike> {
        if self.state.membrane_potential >= self.threshold + 10.0 { // Spike condition
            self.reset();
            self.adaptation_current += self.adaptation_increment;
            self.state.refractory_timer = TimeStep::from_ms(self.refractory_period);
            
            // Create spike with proper type conversion and error handling
            match Spike::new(
                self.id.into(),
                crate::time::Time::from_nanos(0),
                1.0 // Default spike amplitude
            ) {
                Ok(spike) => Some(spike),
                Err(_) => None, // Log error in production code
            }
        } else {
            None
        }
    }
    
    fn membrane_potential(&self) -> f64 {
        self.state.membrane_potential
    }
    
    fn set_membrane_potential(&mut self, voltage: f64) {
        self.state.membrane_potential = voltage;
    }
    
    fn threshold(&self) -> f64 {
        self.threshold
    }
    
    fn reset(&mut self) {
        self.state.membrane_potential = self.reset_potential;
    }
    
    fn id(&self) -> NeuronId {
        self.id
    }
    
    fn set_id(&mut self, id: NeuronId) {
        self.id = id;
    }
}

/// Izhikevich neuron model
/// 
/// A computationally efficient model that can reproduce various firing patterns
/// depending on parameter values.
#[derive(Debug, Clone, PartialEq)]
pub struct IzhikevichNeuron {
    id: NeuronId,
    state: NeuronState,
    recovery_variable: f64,
    
    // Parameters
    pub a: f64, // Recovery time constant
    pub b: f64, // Recovery sensitivity  
    pub c: f64, // Reset potential
    pub d: f64, // Recovery increment
}

impl IzhikevichNeuron {
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            id: NeuronId(0),
            state: NeuronState::new(),
            recovery_variable: -14.0, // Typical initial value
            a,
            b,
            c,
            d,
        }
    }
    
    /// Create a regular spiking neuron
    pub fn regular_spiking(id: NeuronId) -> Self {
        let mut neuron = Self::new(0.02, 0.2, -65.0, 8.0);
        neuron.id = id;
        neuron
    }
    
    /// Create an intrinsically bursting neuron
    pub fn intrinsically_bursting(id: NeuronId) -> Self {
        let mut neuron = Self::new(0.02, 0.25, -65.0, 2.0);
        neuron.id = id;
        neuron
    }
    
    /// Create a chattering neuron
    pub fn chattering(id: NeuronId) -> Self {
        let mut neuron = Self::new(0.02, 0.2, -50.0, 2.0);
        neuron.id = id;
        neuron
    }
    
    /// Create a fast spiking neuron
    pub fn fast_spiking(id: NeuronId) -> Self {
        let mut neuron = Self::new(0.1, 0.2, -65.0, 2.0);
        neuron.id = id;
        neuron
    }
    
    pub fn recovery_variable(&self) -> f64 {
        self.recovery_variable
    }
}

impl Default for IzhikevichNeuron {
    fn default() -> Self {
        Self::regular_spiking(NeuronId(0))
    }
}

impl Neuron for IzhikevichNeuron {
    fn integrate(&mut self, input_current: f64, dt: TimeStep) {
        let dt_ms = dt.as_ms();
        let v = self.state.membrane_potential;
        let u = self.recovery_variable;
        
        // Izhikevich equations
        // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        // du/dt = a*(b*v - u)
        
        let dv_dt = 0.04 * v * v + 5.0 * v + 140.0 - u + input_current;
        let du_dt = self.a * (self.b * v - u);
        
        self.state.membrane_potential += dv_dt * dt_ms;
        self.recovery_variable += du_dt * dt_ms;
    }
    
    fn update(&mut self, _dt: TimeStep) -> Option<Spike> {
        if self.state.membrane_potential >= 30.0 { // Fixed threshold for Izhikevich
            self.state.membrane_potential = self.c;
            self.recovery_variable += self.d;
            
            // Create spike with proper type conversion and error handling
            match Spike::new(
                self.id.into(),
                crate::time::Time::from_nanos(0),
                1.0 // Default spike amplitude
            ) {
                Ok(spike) => Some(spike),
                Err(_) => None, // Log error in production code
            }
        } else {
            None
        }
    }
    
    fn membrane_potential(&self) -> f64 {
        self.state.membrane_potential
    }
    
    fn set_membrane_potential(&mut self, voltage: f64) {
        self.state.membrane_potential = voltage;
    }
    
    fn threshold(&self) -> f64 {
        30.0 // Fixed threshold for Izhikevich model
    }
    
    fn reset(&mut self) {
        self.state.membrane_potential = self.c;
        self.recovery_variable += self.d;
    }
    
    fn id(&self) -> NeuronId {
        self.id
    }
    
    fn set_id(&mut self, id: NeuronId) {
        self.id = id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::TimeStep;

    #[test]
    fn test_neuron_id() {
        let id = NeuronId::new(42);
        assert_eq!(id.as_usize(), 42);
        
        let id2 = NeuronId(123);
        assert_eq!(id2.as_usize(), 123);
        assert_ne!(id, id2);
    }

    #[test]
    fn test_neuron_state() {
        let state = NeuronState::new();
        assert_eq!(state.membrane_potential(), -65.0);
        assert!(!state.is_refractory());
        assert!(state.last_spike_time.is_none());
    }

    #[test]
    fn test_lif_neuron_creation() {
        let neuron = LIFNeuron::new(NeuronId(5));
        assert_eq!(neuron.id().as_usize(), 5);
        assert_eq!(neuron.membrane_potential(), -65.0);
        assert_eq!(neuron.threshold(), -55.0);
    }

    #[test]
    fn test_lif_neuron_integration() {
        let mut neuron = LIFNeuron::default();
        let dt = TimeStep::from_ms(0.1);
        
        // Apply positive current
        neuron.integrate(1.0, dt);
        
        // Membrane potential should increase
        assert!(neuron.membrane_potential() > -65.0);
    }

    #[test]
    fn test_lif_neuron_spike_generation() {
        let mut neuron = LIFNeuron::default();
        let dt = TimeStep::from_ms(0.1);
        
        // Set voltage above threshold
        neuron.set_membrane_potential(-50.0);
        
        // Should generate spike
        let spike = neuron.update(dt);
        assert!(spike.is_some());
        
        // Should be reset
        assert_eq!(neuron.membrane_potential(), neuron.reset_potential);
    }

    #[test]
    fn test_lif_neuron_refractory_period() {
        let mut neuron = LIFNeuron::default();
        let dt = TimeStep::from_ms(0.1);
        
        // Trigger spike
        neuron.set_membrane_potential(-50.0);
        neuron.update(dt);
        
        // During refractory period, integration should not change voltage
        let voltage_before = neuron.membrane_potential();
        neuron.integrate(10.0, dt); // Large current
        assert_eq!(neuron.membrane_potential(), voltage_before);
    }

    #[test]
    fn test_adex_neuron_creation() {
        let neuron = AdExNeuron::new(NeuronId(3));
        assert_eq!(neuron.id().as_usize(), 3);
        assert_eq!(neuron.adaptation_current(), 0.0);
    }

    #[test]
    fn test_adex_neuron_adaptation() {
        let mut neuron = AdExNeuron::default();
        let dt = TimeStep::from_ms(0.1);
        
        // Apply current and integrate
        neuron.integrate(100.0, dt);
        
        // Adaptation current should remain close to zero with no spikes
        assert!(neuron.adaptation_current().abs() < 0.1);
        
        // Trigger spike by setting high voltage
        neuron.set_membrane_potential(0.0);
        neuron.update(dt);
        
        // Adaptation current should increase after spike
        assert!(neuron.adaptation_current() > 0.0);
    }

    #[test]
    fn test_izhikevich_neuron_patterns() {
        // Test regular spiking
        let mut rs_neuron = IzhikevichNeuron::regular_spiking(NeuronId(1));
        let dt = TimeStep::from_ms(0.1);
        
        assert_eq!(rs_neuron.a, 0.02);
        assert_eq!(rs_neuron.b, 0.2);
        assert_eq!(rs_neuron.c, -65.0);
        assert_eq!(rs_neuron.d, 8.0);
        
        // Test intrinsically bursting
        let ib_neuron = IzhikevichNeuron::intrinsically_bursting(NeuronId(2));
        assert_eq!(ib_neuron.b, 0.25);
        assert_eq!(ib_neuron.d, 2.0);
    }

    #[test]
    fn test_izhikevich_neuron_dynamics() {
        let mut neuron = IzhikevichNeuron::default();
        let dt = TimeStep::from_ms(0.1);
        
        // Apply constant current
        for _ in 0..1000 {
            neuron.integrate(10.0, dt);
            if let Some(_spike) = neuron.update(dt) {
                // Should eventually spike
                break;
            }
        }
        
        // Voltage should have been reset after spike
        assert!(neuron.membrane_potential() < 0.0);
    }

    #[test]
    fn test_neuron_parameter_modification() {
        let mut neuron = LIFNeuron::with_params(
            NeuronId(10),
            15.0,  // tau_membrane
            -50.0, // threshold
            -75.0, // reset_potential
            3.0,   // refractory_period
        );
        
        assert_eq!(neuron.tau_membrane, 15.0);
        assert_eq!(neuron.threshold(), -50.0);
        assert_eq!(neuron.reset_potential, -75.0);
        assert_eq!(neuron.refractory_period, 3.0);
    }

    #[test]
    fn test_neuron_trait_consistency() {
        let mut lif = LIFNeuron::default();
        let mut adex = AdExNeuron::default();
        let mut izh = IzhikevichNeuron::default();
        
        let dt = TimeStep::from_ms(0.1);
        let current = 5.0;
        
        // All should integrate current
        lif.integrate(current, dt);
        adex.integrate(current, dt);
        izh.integrate(current, dt);
        
        // All should respond to threshold crossing
        lif.set_membrane_potential(lif.threshold() + 1.0);
        adex.set_membrane_potential(adex.threshold() + 20.0); // AdEx needs higher for spike
        izh.set_membrane_potential(35.0); // Above Izhikevich threshold
        
        assert!(lif.update(dt).is_some());
        assert!(adex.update(dt).is_some());
        assert!(izh.update(dt).is_some());
    }
}