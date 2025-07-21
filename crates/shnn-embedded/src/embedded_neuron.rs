//! Embedded neuron implementations optimized for no-std environments
//!
//! This module provides memory-efficient neuron models that use fixed-point
//! arithmetic and are suitable for real-time execution on microcontrollers.

use crate::{
    error::{EmbeddedError, EmbeddedResult},
    fixed_point::{FixedPoint, Q16_16, FixedSpike},
};
use heapless::Vec;
use core::marker::PhantomData;

/// Maximum number of inputs per neuron in embedded systems
pub const MAX_INPUTS: usize = 32;

/// Maximum number of spikes in input buffer
pub const MAX_SPIKE_BUFFER: usize = 16;

/// Embedded neuron trait for fixed-point computation
pub trait EmbeddedNeuron<T: FixedPoint> {
    /// Update neuron state with input current
    fn update(&mut self, dt: T, input_current: T) -> EmbeddedResult<Option<FixedSpike<T>>>;
    
    /// Reset neuron to initial state
    fn reset(&mut self);
    
    /// Get current membrane potential
    fn membrane_potential(&self) -> T;
    
    /// Set membrane potential (for initialization or external control)
    fn set_membrane_potential(&mut self, potential: T);
    
    /// Check if neuron is in refractory period
    fn is_refractory(&self) -> bool;
    
    /// Get neuron ID
    fn id(&self) -> u16;
}

/// Leaky Integrate-and-Fire (LIF) neuron optimized for embedded systems
#[derive(Debug, Clone)]
pub struct EmbeddedLIFNeuron<T: FixedPoint> {
    /// Neuron ID
    id: u16,
    /// Current membrane potential
    potential: T,
    /// Resting potential
    v_rest: T,
    /// Threshold potential
    v_thresh: T,
    /// Reset potential
    v_reset: T,
    /// Membrane time constant
    tau_m: T,
    /// Refractory period duration
    tau_ref: T,
    /// Current refractory time remaining
    refractory_time: T,
    /// Current simulation time
    current_time: T,
    /// Spike counter for ID generation
    spike_count: u32,
}

impl<T: FixedPoint> EmbeddedLIFNeuron<T> {
    /// Create a new LIF neuron with default parameters
    pub fn new(id: u16) -> Self {
        Self {
            id,
            potential: T::from_float(-70.0),
            v_rest: T::from_float(-70.0),
            v_thresh: T::from_float(-55.0),
            v_reset: T::from_float(-75.0),
            tau_m: T::from_float(0.02),
            tau_ref: T::from_float(0.002),
            refractory_time: T::zero(),
            current_time: T::zero(),
            spike_count: 0,
        }
    }
    
    /// Create a new LIF neuron with custom parameters
    pub fn with_parameters(
        id: u16,
        v_rest: T,
        v_thresh: T,
        v_reset: T,
        tau_m: T,
        tau_ref: T,
    ) -> Self {
        Self {
            id,
            potential: v_rest,
            v_rest,
            v_thresh,
            v_reset,
            tau_m,
            tau_ref,
            refractory_time: T::zero(),
            current_time: T::zero(),
            spike_count: 0,
        }
    }
}

impl<T: FixedPoint> EmbeddedNeuron<T> for EmbeddedLIFNeuron<T> {
    fn update(&mut self, dt: T, input_current: T) -> EmbeddedResult<Option<FixedSpike<T>>> {
        self.current_time = self.current_time + dt;
        
        // Handle refractory period
        if self.refractory_time > T::zero() {
            self.refractory_time = self.refractory_time.saturating_sub(dt);
            if self.refractory_time <= T::zero() {
                self.potential = self.v_reset;
            }
            return Ok(None);
        }
        
        // Update membrane potential using Euler integration
        // dV/dt = (V_rest - V + R*I) / tau_m
        let dv_dt = (self.v_rest - self.potential + input_current) / self.tau_m;
        self.potential = self.potential + dv_dt * dt;
        
        // Check for spike threshold crossing
        if self.potential >= self.v_thresh {
            // Generate spike
            let spike = FixedSpike::new(
                self.id,
                self.current_time,
                T::one(),
            );
            
            // Reset neuron
            self.potential = self.v_reset;
            self.refractory_time = self.tau_ref;
            self.spike_count = self.spike_count.wrapping_add(1);
            
            Ok(Some(spike))
        } else {
            Ok(None)
        }
    }
    
    fn reset(&mut self) {
        self.potential = self.v_rest;
        self.refractory_time = T::zero();
        self.current_time = T::zero();
        self.spike_count = 0;
    }
    
    fn membrane_potential(&self) -> T {
        self.potential
    }
    
    fn set_membrane_potential(&mut self, potential: T) {
        self.potential = potential;
    }
    
    fn is_refractory(&self) -> bool {
        self.refractory_time > T::zero()
    }
    
    fn id(&self) -> u16 {
        self.id
    }
}

/// Izhikevich neuron model optimized for embedded systems
#[derive(Debug, Clone)]
pub struct EmbeddedIzhikevichNeuron<T: FixedPoint> {
    /// Neuron ID
    id: u16,
    /// Membrane potential
    v: T,
    /// Recovery variable
    u: T,
    /// Parameter a (time scale of recovery variable)
    a: T,
    /// Parameter b (sensitivity of recovery variable)
    b: T,
    /// Parameter c (after-spike reset value of v)
    c: T,
    /// Parameter d (after-spike reset parameter for u)
    d: T,
    /// Current simulation time
    current_time: T,
    /// Spike counter
    spike_count: u32,
}

impl<T: FixedPoint> EmbeddedIzhikevichNeuron<T> {
    /// Create regular spiking neuron
    pub fn regular_spiking(id: u16) -> Self {
        Self {
            id,
            v: T::from_float(-70.0),
            u: T::from_float(-14.0),
            a: T::from_float(0.02),
            b: T::from_float(0.2),
            c: T::from_float(-65.0),
            d: T::from_float(8.0),
            current_time: T::zero(),
            spike_count: 0,
        }
    }
    
    /// Create fast spiking neuron
    pub fn fast_spiking(id: u16) -> Self {
        Self {
            id,
            v: T::from_float(-70.0),
            u: T::from_float(-14.0),
            a: T::from_float(0.1),
            b: T::from_float(0.2),
            c: T::from_float(-65.0),
            d: T::from_float(2.0),
            current_time: T::zero(),
            spike_count: 0,
        }
    }
    
    /// Create custom Izhikevich neuron
    pub fn custom(id: u16, a: T, b: T, c: T, d: T) -> Self {
        Self {
            id,
            v: T::from_float(-70.0),
            u: b * T::from_float(-70.0),
            a,
            b,
            c,
            d,
            current_time: T::zero(),
            spike_count: 0,
        }
    }
}

impl<T: FixedPoint> EmbeddedNeuron<T> for EmbeddedIzhikevichNeuron<T> {
    fn update(&mut self, dt: T, input_current: T) -> EmbeddedResult<Option<FixedSpike<T>>> {
        self.current_time = self.current_time + dt;
        
        // Izhikevich model equations:
        // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        // du/dt = a*(b*v - u)
        
        let v_squared = self.v * self.v;
        let dv_dt = T::from_float(0.04) * v_squared + 
                   T::from_float(5.0) * self.v + 
                   T::from_float(140.0) - 
                   self.u + 
                   input_current;
        
        let du_dt = self.a * (self.b * self.v - self.u);
        
        // Update state variables
        self.v = self.v + dv_dt * dt;
        self.u = self.u + du_dt * dt;
        
        // Check for spike
        if self.v >= T::from_float(30.0) {
            let spike = FixedSpike::new(
                self.id,
                self.current_time,
                T::one(),
            );
            
            // Reset after spike
            self.v = self.c;
            self.u = self.u + self.d;
            self.spike_count = self.spike_count.wrapping_add(1);
            
            Ok(Some(spike))
        } else {
            Ok(None)
        }
    }
    
    fn reset(&mut self) {
        self.v = T::from_float(-70.0);
        self.u = self.b * self.v;
        self.current_time = T::zero();
        self.spike_count = 0;
    }
    
    fn membrane_potential(&self) -> T {
        self.v
    }
    
    fn set_membrane_potential(&mut self, potential: T) {
        self.v = potential;
    }
    
    fn is_refractory(&self) -> bool {
        false // Izhikevich model doesn't use explicit refractory period
    }
    
    fn id(&self) -> u16 {
        self.id
    }
}

/// Embedded synapse for connecting neurons
#[derive(Debug, Clone)]
pub struct EmbeddedSynapse<T: FixedPoint> {
    /// Pre-synaptic neuron ID
    pub pre_id: u16,
    /// Post-synaptic neuron ID
    pub post_id: u16,
    /// Synaptic weight
    pub weight: T,
    /// Synaptic delay (in time steps)
    pub delay: u16,
    /// Spike buffer for delayed transmission
    spike_buffer: Vec<(T, T), MAX_SPIKE_BUFFER>, // (timestamp, amplitude)
    /// Current buffer index
    buffer_index: usize,
}

impl<T: FixedPoint> EmbeddedSynapse<T> {
    /// Create a new synapse
    pub fn new(pre_id: u16, post_id: u16, weight: T, delay: u16) -> Self {
        Self {
            pre_id,
            post_id,
            weight,
            delay,
            spike_buffer: Vec::new(),
            buffer_index: 0,
        }
    }
    
    /// Process incoming spike
    pub fn receive_spike(&mut self, spike: &FixedSpike<T>) -> EmbeddedResult<()> {
        if spike.source == self.pre_id {
            // Add spike to buffer with current timestamp
            if self.spike_buffer.len() < MAX_SPIKE_BUFFER {
                self.spike_buffer.push((spike.timestamp, spike.amplitude))
                    .map_err(|_| EmbeddedError::BufferFull)?;
            } else {
                // Circular buffer behavior
                self.spike_buffer[self.buffer_index] = (spike.timestamp, spike.amplitude);
                self.buffer_index = (self.buffer_index + 1) % MAX_SPIKE_BUFFER;
            }
        }
        Ok(())
    }
    
    /// Get delayed output current for given time
    pub fn get_output_current(&self, current_time: T) -> T {
        let delay_time = T::from_int(self.delay as i32);
        let mut total_current = T::zero();
        
        for &(spike_time, amplitude) in &self.spike_buffer {
            let elapsed = current_time - spike_time;
            if elapsed >= delay_time && elapsed < delay_time + T::from_float(0.001) {
                total_current = total_current + self.weight * amplitude;
            }
        }
        
        total_current
    }
    
    /// Clear old spikes from buffer
    pub fn cleanup_buffer(&mut self, current_time: T, max_age: T) {
        self.spike_buffer.retain(|&(timestamp, _)| {
            current_time - timestamp <= max_age
        });
    }
}

/// Embedded neuron population for efficient batch processing
#[derive(Debug)]
pub struct EmbeddedNeuronPopulation<T: FixedPoint, N: EmbeddedNeuron<T>> {
    /// Collection of neurons
    neurons: Vec<N, 64>, // Support up to 64 neurons
    /// Current simulation time
    current_time: T,
    /// Time step
    dt: T,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T: FixedPoint, N: EmbeddedNeuron<T>> EmbeddedNeuronPopulation<T, N> {
    /// Create a new neuron population
    pub fn new(dt: T) -> Self {
        Self {
            neurons: Vec::new(),
            current_time: T::zero(),
            dt,
            _phantom: PhantomData,
        }
    }
    
    /// Add a neuron to the population
    pub fn add_neuron(&mut self, neuron: N) -> EmbeddedResult<()> {
        self.neurons.push(neuron)
            .map_err(|_| EmbeddedError::BufferFull)
    }
    
    /// Update all neurons in the population
    pub fn update(&mut self, inputs: &[T]) -> EmbeddedResult<Vec<FixedSpike<T>, 64>> {
        let mut spikes = Vec::new();
        
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let input_current = inputs.get(i).copied().unwrap_or(T::zero());
            
            if let Some(spike) = neuron.update(self.dt, input_current)? {
                spikes.push(spike).map_err(|_| EmbeddedError::BufferFull)?;
            }
        }
        
        self.current_time = self.current_time + self.dt;
        Ok(spikes)
    }
    
    /// Reset all neurons
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        self.current_time = T::zero();
    }
    
    /// Get membrane potentials of all neurons
    pub fn get_membrane_potentials(&self) -> Vec<T, 64> {
        let mut potentials = Vec::new();
        for neuron in &self.neurons {
            let _ = potentials.push(neuron.membrane_potential());
        }
        potentials
    }
    
    /// Get current simulation time
    pub fn current_time(&self) -> T {
        self.current_time
    }
    
    /// Get number of neurons
    pub fn len(&self) -> usize {
        self.neurons.len()
    }
    
    /// Check if population is empty
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_embedded_lif_neuron() {
        let mut neuron = EmbeddedLIFNeuron::<Q16_16>::new(0);
        let dt = Q16_16::from_float(0.001);
        let input = Q16_16::from_float(20.0);
        
        // Should not spike immediately
        let result = neuron.update(dt, Q16_16::zero()).unwrap();
        assert!(result.is_none());
        
        // Apply strong input to cause spike
        let mut spike_generated = false;
        for _ in 0..100 {
            if let Some(_spike) = neuron.update(dt, input).unwrap() {
                spike_generated = true;
                break;
            }
        }
        assert!(spike_generated);
    }
    
    #[test]
    fn test_embedded_izhikevich_neuron() {
        let mut neuron = EmbeddedIzhikevichNeuron::<Q16_16>::regular_spiking(1);
        let dt = Q16_16::from_float(0.001);
        let input = Q16_16::from_float(15.0);
        
        let mut spike_generated = false;
        for _ in 0..1000 {
            if let Some(_spike) = neuron.update(dt, input).unwrap() {
                spike_generated = true;
                break;
            }
        }
        assert!(spike_generated);
    }
    
    #[test]
    fn test_embedded_synapse() {
        let mut synapse = EmbeddedSynapse::<Q16_16>::new(
            0, 1, Q16_16::from_float(0.5), 5
        );
        
        let spike = FixedSpike::new(
            0, 
            Q16_16::from_float(1.0), 
            Q16_16::one()
        );
        
        synapse.receive_spike(&spike).unwrap();
        
        // Should produce output after delay
        let current_time = Q16_16::from_float(1.005);
        let output = synapse.get_output_current(current_time);
        assert!(output > Q16_16::zero());
    }
    
    #[test]
    fn test_neuron_population() {
        let mut population = EmbeddedNeuronPopulation::new(Q16_16::from_float(0.001));
        
        let neuron1 = EmbeddedLIFNeuron::<Q16_16>::new(0);
        let neuron2 = EmbeddedLIFNeuron::<Q16_16>::new(1);
        
        population.add_neuron(neuron1).unwrap();
        population.add_neuron(neuron2).unwrap();
        
        assert_eq!(population.len(), 2);
        
        let inputs = [Q16_16::from_float(20.0), Q16_16::from_float(25.0)];
        let _spikes = population.update(&inputs).unwrap();
        
        assert!(population.current_time() > Q16_16::zero());
    }
}