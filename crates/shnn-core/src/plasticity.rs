//! Synaptic plasticity mechanisms for learning in neuromorphic systems
//!
//! This module provides biologically inspired learning rules including
//! spike-timing dependent plasticity (STDP) and other adaptation mechanisms.

use crate::{
    error::{Result, SHNNError},
    spike::{NeuronId, Spike},
    time::{Time, Duration},
    hypergraph::HyperedgeId,
};
use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "math")]
use crate::math::{exponential_decay, sigmoid};

/// Trait defining synaptic plasticity rules
pub trait PlasticityRule {
    /// Configuration type for this plasticity rule
    type Config;
    
    /// State type for tracking plasticity variables
    type State;
    
    /// Update synaptic weight based on pre- and post-synaptic activity
    ///
    /// Returns the new weight value
    fn update_weight(
        &self,
        current_weight: f32,
        pre_spike_time: Time,
        post_spike_time: Time,
        config: &Self::Config,
        state: &mut Self::State,
    ) -> f32;
    
    /// Initialize plasticity state
    fn init_state(&self) -> Self::State;
    
    /// Reset plasticity state
    fn reset_state(&self, state: &mut Self::State);
    
    /// Get the name of this plasticity rule
    fn name(&self) -> &'static str;
}

/// Spike-timing dependent plasticity (STDP) configuration
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct STDPConfig {
    /// Maximum weight change for potentiation
    pub a_plus: f32,
    /// Maximum weight change for depression
    pub a_minus: f32,
    /// Time constant for potentiation (ms)
    pub tau_plus: f32,
    /// Time constant for depression (ms)
    pub tau_minus: f32,
    /// Minimum weight value
    pub w_min: f32,
    /// Maximum weight value
    pub w_max: f32,
    /// Learning rate scaling factor
    pub learning_rate: f32,
    /// Whether to use multiplicative STDP
    pub multiplicative: bool,
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            a_plus: 0.005,      // 0.5% potentiation
            a_minus: 0.00525,   // Slightly asymmetric depression
            tau_plus: 20.0,     // 20ms potentiation window
            tau_minus: 20.0,    // 20ms depression window
            w_min: 0.0,         // No negative weights
            w_max: 1.0,         // Normalized maximum
            learning_rate: 1.0,  // Full learning rate
            multiplicative: false, // Additive STDP by default
        }
    }
}

impl STDPConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.tau_plus <= 0.0 || self.tau_minus <= 0.0 {
            return Err(SHNNError::plasticity_error("Time constants must be positive"));
        }
        if self.w_min < 0.0 || self.w_max <= self.w_min {
            return Err(SHNNError::plasticity_error("Invalid weight bounds"));
        }
        if self.learning_rate < 0.0 || self.learning_rate > 10.0 {
            return Err(SHNNError::plasticity_error("Learning rate out of reasonable range"));
        }
        Ok(())
    }
    
    /// Create potentiation-dominant configuration
    pub fn potentiation_dominant() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.005,
            ..Default::default()
        }
    }
    
    /// Create depression-dominant configuration
    pub fn depression_dominant() -> Self {
        Self {
            a_plus: 0.005,
            a_minus: 0.01,
            ..Default::default()
        }
    }
    
    /// Create multiplicative STDP configuration
    pub fn multiplicative() -> Self {
        Self {
            multiplicative: true,
            ..Default::default()
        }
    }
}

/// STDP plasticity state
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct STDPState {
    /// Last presynaptic spike time
    pub last_pre_spike: Option<Time>,
    /// Last postsynaptic spike time
    pub last_post_spike: Option<Time>,
    /// Running average of weight changes (for stability)
    pub weight_change_avg: f32,
    /// Number of updates performed
    pub update_count: u32,
}

impl Default for STDPState {
    fn default() -> Self {
        Self {
            last_pre_spike: None,
            last_post_spike: None,
            weight_change_avg: 0.0,
            update_count: 0,
        }
    }
}

/// Spike-timing dependent plasticity implementation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct STDPRule;

impl STDPRule {
    /// Create a new STDP rule
    pub fn new() -> Self {
        Self
    }
    
    /// Compute STDP weight change
    fn compute_weight_change(
        &self,
        dt: f32, // Time difference (post - pre) in ms
        config: &STDPConfig,
        current_weight: f32,
    ) -> f32 {
        let weight_change = if dt > 0.0 {
            // Potentiation: post after pre
            let factor = (-dt / config.tau_plus).exp();
            if config.multiplicative {
                config.a_plus * factor * (config.w_max - current_weight)
            } else {
                config.a_plus * factor
            }
        } else {
            // Depression: pre after post
            let factor = (dt / config.tau_minus).exp(); // dt is negative
            if config.multiplicative {
                -config.a_minus * factor * (current_weight - config.w_min)
            } else {
                -config.a_minus * factor
            }
        };
        
        weight_change * config.learning_rate
    }
    
    /// Apply weight bounds
    fn bound_weight(&self, weight: f32, config: &STDPConfig) -> f32 {
        weight.max(config.w_min).min(config.w_max)
    }
}

impl PlasticityRule for STDPRule {
    type Config = STDPConfig;
    type State = STDPState;
    
    fn update_weight(
        &self,
        current_weight: f32,
        pre_spike_time: Time,
        post_spike_time: Time,
        config: &Self::Config,
        state: &mut Self::State,
    ) -> f32 {
        // Update spike times
        state.last_pre_spike = Some(pre_spike_time);
        state.last_post_spike = Some(post_spike_time);
        state.update_count += 1;
        
        // Calculate time difference in milliseconds
        let dt = (post_spike_time - pre_spike_time).as_secs_f64() as f32 * 1000.0;
        
        // Compute weight change
        let weight_change = self.compute_weight_change(dt, config, current_weight);
        
        // Update running average
        let alpha = 0.1; // Exponential moving average factor
        state.weight_change_avg = (1.0 - alpha) * state.weight_change_avg + alpha * weight_change.abs();
        
        // Apply weight change and bounds
        let new_weight = current_weight + weight_change;
        self.bound_weight(new_weight, config)
    }
    
    fn init_state(&self) -> Self::State {
        STDPState::default()
    }
    
    fn reset_state(&self, state: &mut Self::State) {
        *state = STDPState::default();
    }
    
    fn name(&self) -> &'static str {
        "STDP"
    }
}

impl Default for STDPRule {
    fn default() -> Self {
        Self::new()
    }
}

/// Homeostatic plasticity configuration
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HomeostaticConfig {
    /// Target firing rate (Hz)
    pub target_rate: f32,
    /// Scaling factor for weight adjustments
    pub scaling_factor: f32,
    /// Time window for rate measurement (ms)
    pub time_window: f32,
    /// Minimum scaling factor
    pub min_scaling: f32,
    /// Maximum scaling factor
    pub max_scaling: f32,
}

impl Default for HomeostaticConfig {
    fn default() -> Self {
        Self {
            target_rate: 5.0,      // 5 Hz target
            scaling_factor: 0.001, // Conservative scaling
            time_window: 1000.0,   // 1 second window
            min_scaling: 0.1,      // Don't scale below 10%
            max_scaling: 10.0,     // Don't scale above 1000%
        }
    }
}

/// Homeostatic plasticity state
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HomeostaticState {
    /// Recent spike times for rate calculation
    pub recent_spikes: Vec<Time>,
    /// Current firing rate estimate
    pub current_rate: f32,
    /// Last update time
    pub last_update: Time,
}

impl Default for HomeostaticState {
    fn default() -> Self {
        Self {
            recent_spikes: Vec::new(),
            current_rate: 0.0,
            last_update: Time::ZERO,
        }
    }
}

/// Homeostatic plasticity rule
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HomeostaticRule;

impl HomeostaticRule {
    /// Create a new homeostatic rule
    pub fn new() -> Self {
        Self
    }
    
    /// Update firing rate estimate
    pub fn update_firing_rate(
        &self,
        state: &mut HomeostaticState,
        current_time: Time,
        config: &HomeostaticConfig,
    ) {
        // Remove old spikes outside the time window
        let window_duration = Duration::from_secs_f64(config.time_window as f64 / 1000.0)
            .unwrap_or(Duration::from_millis(100)); // Fallback to 100ms if invalid
        let window_start = current_time - window_duration;
        state.recent_spikes.retain(|&spike_time| spike_time >= window_start);
        
        // Calculate current firing rate
        if !state.recent_spikes.is_empty() {
            let time_span = (current_time - state.recent_spikes[0]).as_secs_f64();
            if time_span > 0.0 {
                state.current_rate = state.recent_spikes.len() as f32 / time_span as f32;
            }
        } else {
            state.current_rate = 0.0;
        }
        
        state.last_update = current_time;
    }
    
    /// Add a spike to the history
    pub fn add_spike(&self, state: &mut HomeostaticState, spike_time: Time) {
        state.recent_spikes.push(spike_time);
        
        // Keep list manageable size
        if state.recent_spikes.len() > 1000 {
            state.recent_spikes.remove(0);
        }
    }
    
    /// Calculate homeostatic scaling factor
    pub fn compute_scaling_factor(&self, state: &HomeostaticState, config: &HomeostaticConfig) -> f32 {
        if state.current_rate == 0.0 {
            return 1.0; // No scaling if no activity
        }
        
        // Scaling factor inversely related to current rate vs target
        let rate_ratio = state.current_rate / config.target_rate;
        let scaling = 1.0 + config.scaling_factor * (1.0 / rate_ratio - 1.0);
        
        scaling.max(config.min_scaling).min(config.max_scaling)
    }
}

impl Default for HomeostaticRule {
    fn default() -> Self {
        Self::new()
    }
}

/// Metaplasticity configuration (plasticity of plasticity)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetaplasticityConfig {
    /// Base learning rate
    pub base_learning_rate: f32,
    /// Adaptation time constant
    pub adaptation_tau: f32,
    /// Threshold for adaptation
    pub adaptation_threshold: f32,
    /// Maximum learning rate multiplier
    pub max_rate_multiplier: f32,
}

impl Default for MetaplasticityConfig {
    fn default() -> Self {
        Self {
            base_learning_rate: 1.0,
            adaptation_tau: 100.0,     // 100ms adaptation
            adaptation_threshold: 0.1,  // 10% change threshold
            max_rate_multiplier: 5.0,   // Up to 5x learning rate
        }
    }
}

/// Combined plasticity mechanism supporting multiple rules
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PlasticityManager {
    /// STDP rule configuration and state
    stdp_config: STDPConfig,
    stdp_states: Vec<STDPState>,
    
    /// Homeostatic rule configuration and state
    homeostatic_config: HomeostaticConfig,
    homeostatic_states: Vec<HomeostaticState>,
    
    /// Whether STDP is enabled
    stdp_enabled: bool,
    
    /// Whether homeostatic plasticity is enabled
    homeostatic_enabled: bool,
    
    /// Global learning rate modifier
    global_learning_rate: f32,
}

impl PlasticityManager {
    /// Create a new plasticity manager
    pub fn new() -> Self {
        Self {
            stdp_config: STDPConfig::default(),
            stdp_states: Vec::new(),
            homeostatic_config: HomeostaticConfig::default(),
            homeostatic_states: Vec::new(),
            stdp_enabled: true,
            homeostatic_enabled: false,
            global_learning_rate: 1.0,
        }
    }
    
    /// Configure STDP parameters
    pub fn with_stdp_config(mut self, config: STDPConfig) -> Self {
        self.stdp_config = config;
        self
    }
    
    /// Configure homeostatic parameters
    pub fn with_homeostatic_config(mut self, config: HomeostaticConfig) -> Self {
        self.homeostatic_config = config;
        self
    }
    
    /// Enable or disable STDP
    pub fn set_stdp_enabled(&mut self, enabled: bool) {
        self.stdp_enabled = enabled;
    }
    
    /// Enable or disable homeostatic plasticity
    pub fn set_homeostatic_enabled(&mut self, enabled: bool) {
        self.homeostatic_enabled = enabled;
    }
    
    /// Set global learning rate
    pub fn set_global_learning_rate(&mut self, rate: f32) {
        self.global_learning_rate = rate.max(0.0).min(10.0);
    }
    
    /// Initialize plasticity state for a connection
    pub fn init_connection(&mut self, connection_id: usize) {
        // Ensure vectors are large enough
        while self.stdp_states.len() <= connection_id {
            self.stdp_states.push(STDPState::default());
        }
        
        while self.homeostatic_states.len() <= connection_id {
            self.homeostatic_states.push(HomeostaticState::default());
        }
    }
    
    /// Process a spike pair for plasticity updates
    pub fn process_spike_pair(
        &mut self,
        connection_id: usize,
        current_weight: f32,
        pre_spike_time: Time,
        post_spike_time: Time,
    ) -> f32 {
        self.init_connection(connection_id);
        
        let mut new_weight = current_weight;
        
        // Apply STDP
        if self.stdp_enabled {
            let stdp_rule = STDPRule::new();
            let mut config = self.stdp_config.clone();
            config.learning_rate *= self.global_learning_rate;
            
            new_weight = stdp_rule.update_weight(
                new_weight,
                pre_spike_time,
                post_spike_time,
                &config,
                &mut self.stdp_states[connection_id],
            );
        }
        
        // Apply homeostatic scaling
        if self.homeostatic_enabled && connection_id < self.homeostatic_states.len() {
            let homeostatic_rule = HomeostaticRule::new();
            let scaling = homeostatic_rule.compute_scaling_factor(
                &self.homeostatic_states[connection_id],
                &self.homeostatic_config,
            );
            new_weight *= scaling;
        }
        
        new_weight
    }
    
    /// Update homeostatic state with new spike
    pub fn record_spike(&mut self, neuron_id: usize, spike_time: Time) {
        if self.homeostatic_enabled {
            self.init_connection(neuron_id);
            
            let homeostatic_rule = HomeostaticRule::new();
            homeostatic_rule.add_spike(&mut self.homeostatic_states[neuron_id], spike_time);
            homeostatic_rule.update_firing_rate(
                &mut self.homeostatic_states[neuron_id],
                spike_time,
                &self.homeostatic_config,
            );
        }
    }
    
    /// Reset all plasticity states
    pub fn reset(&mut self) {
        for state in &mut self.stdp_states {
            *state = STDPState::default();
        }
        
        for state in &mut self.homeostatic_states {
            *state = HomeostaticState::default();
        }
    }
    
    /// Get STDP statistics
    pub fn stdp_stats(&self) -> STDPStats {
        let mut stats = STDPStats::default();
        
        for state in &self.stdp_states {
            stats.total_updates += state.update_count as usize;
            stats.avg_weight_change += state.weight_change_avg;
        }
        
        if !self.stdp_states.is_empty() {
            stats.avg_weight_change /= self.stdp_states.len() as f32;
        }
        
        stats.active_connections = self.stdp_states.len();
        
        stats
    }
}

impl Default for PlasticityManager {
    fn default() -> Self {
        Self::new()
    }
}

/// STDP statistics
#[derive(Debug, Clone, Default)]
pub struct STDPStats {
    /// Total number of weight updates
    pub total_updates: usize,
    /// Average weight change magnitude
    pub avg_weight_change: f32,
    /// Number of active connections
    pub active_connections: usize,
}

impl fmt::Display for STDPStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "STDP Stats: {} updates, {:.6} avg change, {} connections",
            self.total_updates, self.avg_weight_change, self.active_connections
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stdp_config_validation() {
        let mut config = STDPConfig::default();
        assert!(config.validate().is_ok());
        
        config.tau_plus = -1.0;
        assert!(config.validate().is_err());
        
        config.tau_plus = 20.0;
        config.w_min = 1.0;
        config.w_max = 0.5;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_stdp_rule() {
        let rule = STDPRule::new();
        let config = STDPConfig::default();
        let mut state = rule.init_state();
        
        let pre_time = Time::from_millis(100);
        let post_time = Time::from_millis(110); // 10ms later -> potentiation
        
        let initial_weight = 0.5;
        let new_weight = rule.update_weight(
            initial_weight,
            pre_time,
            post_time,
            &config,
            &mut state,
        );
        
        // Should potentiate (increase weight)
        assert!(new_weight > initial_weight);
        assert_eq!(state.update_count, 1);
    }
    
    #[test]
    fn test_stdp_depression() {
        let rule = STDPRule::new();
        let config = STDPConfig::default();
        let mut state = rule.init_state();
        
        let pre_time = Time::from_millis(110);
        let post_time = Time::from_millis(100); // 10ms before -> depression
        
        let initial_weight = 0.5;
        let new_weight = rule.update_weight(
            initial_weight,
            pre_time,
            post_time,
            &config,
            &mut state,
        );
        
        // Should depress (decrease weight)
        assert!(new_weight < initial_weight);
    }
    
    #[test]
    fn test_homeostatic_rule() {
        let rule = HomeostaticRule::new();
        let config = HomeostaticConfig::default();
        let mut state = HomeostaticState::default();
        
        // Add some spikes
        for i in 0..10 {
            rule.add_spike(&mut state, Time::from_millis(i * 100));
        }
        
        rule.update_firing_rate(&mut state, Time::from_millis(1000), &config);
        
        // Should have calculated a firing rate
        assert!(state.current_rate > 0.0);
        
        let scaling = rule.compute_scaling_factor(&state, &config);
        assert!(scaling > 0.0);
    }
    
    #[test]
    fn test_plasticity_manager() {
        let mut manager = PlasticityManager::new();
        
        let pre_time = Time::from_millis(100);
        let post_time = Time::from_millis(110);
        let initial_weight = 0.5;
        
        let new_weight = manager.process_spike_pair(0, initial_weight, pre_time, post_time);
        
        // Should have applied STDP
        assert_ne!(new_weight, initial_weight);
        
        let stats = manager.stdp_stats();
        assert_eq!(stats.total_updates, 1);
        assert_eq!(stats.active_connections, 1);
    }
    
    #[test]
    fn test_weight_bounds() {
        let rule = STDPRule::new();
        let mut config = STDPConfig::default();
        config.w_min = 0.0;
        config.w_max = 1.0;
        config.a_plus = 2.0; // Large change to test bounds
        
        let mut state = rule.init_state();
        
        let pre_time = Time::from_millis(100);
        let post_time = Time::from_millis(101); // Small positive dt
        
        let new_weight = rule.update_weight(0.9, pre_time, post_time, &config, &mut state);
        
        // Should be bounded by w_max
        assert!(new_weight <= config.w_max);
        assert!(new_weight >= config.w_min);
    }
}