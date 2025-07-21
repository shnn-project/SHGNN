//! Temporal spike encoding for sensory input processing
//!
//! This module provides various encoding schemes to convert analog signals
//! and discrete data into spike trains for neuromorphic processing.

use crate::{
    error::{Result, SHNNError},
    spike::{NeuronId, Spike, SpikeTrain},
    time::{Time, Duration},
};
use core::fmt;

#[cfg(feature = "std")]
use std::collections::VecDeque;

#[cfg(not(feature = "std"))]
use heapless::Vec as HeaplessVec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "math")]
use crate::math::{sigmoid, exponential_decay};

/// Spike encoding strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EncodingType {
    /// Rate-based encoding (Poisson process)
    Rate,
    /// Temporal encoding based on time-to-first-spike
    Temporal,
    /// Population vector encoding
    Population,
    /// Delta encoding (encode changes)
    Delta,
    /// Rank order encoding
    RankOrder,
    /// Phase encoding
    Phase,
    /// Burst encoding
    Burst,
    /// Binary encoding
    Binary,
}

impl fmt::Display for EncodingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rate => write!(f, "Rate"),
            Self::Temporal => write!(f, "Temporal"),
            Self::Population => write!(f, "Population"),
            Self::Delta => write!(f, "Delta"),
            Self::RankOrder => write!(f, "RankOrder"),
            Self::Phase => write!(f, "Phase"),
            Self::Burst => write!(f, "Burst"),
            Self::Binary => write!(f, "Binary"),
        }
    }
}

/// Trait for spike encoders
pub trait SpikeEncoder {
    /// Configuration type for this encoder
    type Config;
    
    /// Encode a single value into spike trains
    fn encode_value(
        &self,
        value: f32,
        time: Time,
        neuron_ids: &[NeuronId],
        config: &Self::Config,
    ) -> Result<Vec<SpikeTrain>>;
    
    /// Encode a sequence of values
    fn encode_sequence(
        &self,
        values: &[f32],
        start_time: Time,
        dt: Duration,
        neuron_ids: &[NeuronId],
        config: &Self::Config,
    ) -> Result<Vec<SpikeTrain>> {
        let mut all_trains: Vec<SpikeTrain> = Vec::new();
        
        for (i, &value) in values.iter().enumerate() {
            let time = start_time + Duration::from_nanos(dt.as_nanos() * i as u64);
            let trains = self.encode_value(value, time, neuron_ids, config)?;
            
            // Merge spike trains for same neuron IDs
            for train in trains {
                if let Some(existing) = all_trains.iter_mut().find(|t| t.source == train.source) {
                    existing.timestamps.extend(&train.timestamps);
                    if let (Some(ref mut existing_amps), Some(ref train_amps)) =
                        (&mut existing.amplitudes, &train.amplitudes) {
                        existing_amps.extend(train_amps);
                    }
                } else {
                    all_trains.push(train);
                }
            }
        }
        
        // Sort timestamps within each train
        for train in &mut all_trains {
            let mut paired: Vec<_> = train.timestamps.iter().enumerate().collect();
            paired.sort_by_key(|(_, &time)| time);
            
            let sorted_times: Vec<_> = paired.iter().map(|(_, &time)| time).collect();
            train.timestamps = sorted_times;
            
            if let Some(ref mut amplitudes) = train.amplitudes {
                let sorted_amps: Vec<_> = paired.iter().map(|(i, _)| amplitudes[*i]).collect();
                *amplitudes = sorted_amps;
            }
        }
        
        Ok(all_trains)
    }
    
    /// Get encoder type
    fn encoding_type(&self) -> EncodingType;
}

/// Rate-based encoding configuration
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RateEncodingConfig {
    /// Maximum firing rate (Hz)
    pub max_rate: f32,
    /// Minimum firing rate (Hz)
    pub min_rate: f32,
    /// Encoding window duration
    pub window_duration: Duration,
    /// Input value range
    pub value_range: (f32, f32),
    /// Whether to add noise
    pub add_noise: bool,
    /// Noise standard deviation
    pub noise_std: f32,
}

impl Default for RateEncodingConfig {
    fn default() -> Self {
        Self {
            max_rate: 100.0,        // 100 Hz maximum
            min_rate: 0.1,          // 0.1 Hz minimum
            window_duration: Duration::from_millis(100), // 100ms window
            value_range: (0.0, 1.0), // Normalized input
            add_noise: false,
            noise_std: 0.1,
        }
    }
}

/// Rate-based (Poisson) encoder
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RateEncoder {
    /// Random number generator state (simplified)
    #[cfg_attr(feature = "serde", serde(skip))]
    rng_state: u64,
}

impl RateEncoder {
    /// Create a new rate encoder
    pub fn new() -> Self {
        Self {
            rng_state: 12345, // Simple seed
        }
    }
    
    /// Create with custom seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng_state: seed,
        }
    }
    
    /// Simple linear congruential generator for deterministic random numbers
    fn next_random(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state >> 16) as f32 / 65535.0
    }
    
    /// Generate Poisson spike times
    fn generate_poisson_spikes(
        &mut self,
        rate: f32,
        start_time: Time,
        duration: Duration,
    ) -> Vec<Time> {
        let mut spikes = Vec::new();
        let mut current_time = start_time;
        let end_time = start_time + duration;
        
        while current_time < end_time {
            // Exponential inter-spike interval
            let u = self.next_random();
            let interval = -((1.0 - u).ln()) / rate * 1000.0; // Convert to ms
            
            if interval <= 0.0 {
                break;
            }
            
            let duration = Duration::from_secs_f64(interval as f64 / 1000.0)
                .unwrap_or(Duration::from_millis(1)); // Fallback to 1ms if invalid
            current_time = current_time + duration;
            
            if current_time < end_time {
                spikes.push(current_time);
            }
        }
        
        spikes
    }
    
    /// Normalize value to rate range
    fn value_to_rate(&self, value: f32, config: &RateEncodingConfig) -> f32 {
        let normalized = (value - config.value_range.0) / 
                        (config.value_range.1 - config.value_range.0);
        let clamped = normalized.max(0.0).min(1.0);
        
        config.min_rate + clamped * (config.max_rate - config.min_rate)
    }
}

impl SpikeEncoder for RateEncoder {
    type Config = RateEncodingConfig;
    
    fn encode_value(
        &self,
        value: f32,
        time: Time,
        neuron_ids: &[NeuronId],
        config: &Self::Config,
    ) -> Result<Vec<SpikeTrain>> {
        if neuron_ids.is_empty() {
            return Err(SHNNError::encoding_error("No neuron IDs provided"));
        }
        
        let mut encoder = self.clone(); // For mutable access to RNG
        let rate = encoder.value_to_rate(value, config);
        
        let mut trains = Vec::new();
        
        for &neuron_id in neuron_ids {
            let spike_times = encoder.generate_poisson_spikes(
                rate,
                time,
                config.window_duration,
            );
            
            if !spike_times.is_empty() {
                let train = SpikeTrain::binary(neuron_id, spike_times)?;
                trains.push(train);
            }
        }
        
        Ok(trains)
    }
    
    fn encoding_type(&self) -> EncodingType {
        EncodingType::Rate
    }
}

impl Default for RateEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporal encoding configuration
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TemporalEncodingConfig {
    /// Maximum delay time
    pub max_delay: Duration,
    /// Minimum delay time
    pub min_delay: Duration,
    /// Input value range
    pub value_range: (f32, f32),
    /// Whether to invert mapping (high values = short delays)
    pub invert_mapping: bool,
}

impl Default for TemporalEncodingConfig {
    fn default() -> Self {
        Self {
            max_delay: Duration::from_millis(50),
            min_delay: Duration::from_millis(1),
            value_range: (0.0, 1.0),
            invert_mapping: true, // High values fire first
        }
    }
}

/// Temporal (time-to-first-spike) encoder
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TemporalEncoder;

impl TemporalEncoder {
    /// Create a new temporal encoder
    pub fn new() -> Self {
        Self
    }
    
    /// Convert value to spike delay
    fn value_to_delay(&self, value: f32, config: &TemporalEncodingConfig) -> Duration {
        let normalized = (value - config.value_range.0) / 
                        (config.value_range.1 - config.value_range.0);
        let clamped = normalized.max(0.0).min(1.0);
        
        let delay_ratio = if config.invert_mapping {
            1.0 - clamped // High values = short delays
        } else {
            clamped // High values = long delays
        };
        
        let delay_range = config.max_delay - config.min_delay;
        config.min_delay + Duration::from_nanos(
            (delay_range.as_nanos() as f32 * delay_ratio) as u64
        )
    }
}

impl SpikeEncoder for TemporalEncoder {
    type Config = TemporalEncodingConfig;
    
    fn encode_value(
        &self,
        value: f32,
        time: Time,
        neuron_ids: &[NeuronId],
        config: &Self::Config,
    ) -> Result<Vec<SpikeTrain>> {
        if neuron_ids.is_empty() {
            return Err(SHNNError::encoding_error("No neuron IDs provided"));
        }
        
        let delay = self.value_to_delay(value, config);
        let spike_time = time + delay;
        
        let mut trains = Vec::new();
        
        for &neuron_id in neuron_ids {
            let train = SpikeTrain::binary(neuron_id, vec![spike_time])?;
            trains.push(train);
        }
        
        Ok(trains)
    }
    
    fn encoding_type(&self) -> EncodingType {
        EncodingType::Temporal
    }
}

impl Default for TemporalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Population vector encoding configuration
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PopulationEncodingConfig {
    /// Centers of receptive fields
    pub centers: Vec<f32>,
    /// Width of receptive fields
    pub width: f32,
    /// Maximum firing rate
    pub max_rate: f32,
    /// Encoding duration
    pub duration: Duration,
}

impl Default for PopulationEncodingConfig {
    fn default() -> Self {
        Self {
            centers: vec![0.1, 0.3, 0.5, 0.7, 0.9], // 5 neurons covering 0-1 range
            width: 0.3,
            max_rate: 100.0,
            duration: Duration::from_millis(100),
        }
    }
}

/// Population vector encoder
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PopulationEncoder {
    rate_encoder: RateEncoder,
}

impl PopulationEncoder {
    /// Create a new population encoder
    pub fn new() -> Self {
        Self {
            rate_encoder: RateEncoder::new(),
        }
    }
    
    /// Calculate Gaussian response
    fn gaussian_response(&self, value: f32, center: f32, width: f32) -> f32 {
        let diff = value - center;
        (-0.5 * (diff / width).powi(2)).exp()
    }
}

impl SpikeEncoder for PopulationEncoder {
    type Config = PopulationEncodingConfig;
    
    fn encode_value(
        &self,
        value: f32,
        time: Time,
        neuron_ids: &[NeuronId],
        config: &Self::Config,
    ) -> Result<Vec<SpikeTrain>> {
        if neuron_ids.len() != config.centers.len() {
            return Err(SHNNError::encoding_error(
                "Number of neurons must match number of population centers"
            ));
        }
        
        let mut trains = Vec::new();
        let mut encoder = self.rate_encoder.clone();
        
        for (i, &neuron_id) in neuron_ids.iter().enumerate() {
            let response = self.gaussian_response(value, config.centers[i], config.width);
            let rate = response * config.max_rate;
            
            let spike_times = encoder.generate_poisson_spikes(
                rate,
                time,
                config.duration,
            );
            
            if !spike_times.is_empty() {
                let train = SpikeTrain::binary(neuron_id, spike_times)?;
                trains.push(train);
            }
        }
        
        Ok(trains)
    }
    
    fn encoding_type(&self) -> EncodingType {
        EncodingType::Population
    }
}

impl Default for PopulationEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Delta encoding configuration
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeltaEncodingConfig {
    /// Threshold for positive changes
    pub positive_threshold: f32,
    /// Threshold for negative changes
    pub negative_threshold: f32,
    /// Whether to encode magnitude
    pub encode_magnitude: bool,
    /// Maximum magnitude scaling
    pub max_magnitude: f32,
}

impl Default for DeltaEncodingConfig {
    fn default() -> Self {
        Self {
            positive_threshold: 0.01,
            negative_threshold: -0.01,
            encode_magnitude: false,
            max_magnitude: 1.0,
        }
    }
}

/// Delta encoder (encodes changes)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeltaEncoder {
    /// Previous values for each input channel
    previous_values: Vec<f32>,
}

impl DeltaEncoder {
    /// Create a new delta encoder
    pub fn new() -> Self {
        Self {
            previous_values: Vec::new(),
        }
    }
    
    /// Reset encoder state
    pub fn reset(&mut self) {
        self.previous_values.clear();
    }
    
    /// Encode a single channel
    pub fn encode_channel(
        &mut self,
        channel: usize,
        value: f32,
        time: Time,
        pos_neuron: NeuronId,
        neg_neuron: NeuronId,
        config: &DeltaEncodingConfig,
    ) -> Result<Vec<Spike>> {
        // Ensure we have previous value storage
        while self.previous_values.len() <= channel {
            self.previous_values.push(0.0);
        }
        
        let delta = value - self.previous_values[channel];
        self.previous_values[channel] = value;
        
        let mut spikes = Vec::new();
        
        if delta > config.positive_threshold {
            let amplitude = if config.encode_magnitude {
                (delta / config.max_magnitude).min(1.0)
            } else {
                1.0
            };
            spikes.push(Spike::with_amplitude(pos_neuron, time, amplitude)?);
        } else if delta < config.negative_threshold {
            let amplitude = if config.encode_magnitude {
                (-delta / config.max_magnitude).min(1.0)
            } else {
                1.0
            };
            spikes.push(Spike::with_amplitude(neg_neuron, time, amplitude)?);
        }
        
        Ok(spikes)
    }
}

impl Default for DeltaEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-modal encoder combining different encoding strategies
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MultiModalEncoder {
    /// Rate encoder
    rate_encoder: RateEncoder,
    /// Temporal encoder
    temporal_encoder: TemporalEncoder,
    /// Population encoder
    population_encoder: PopulationEncoder,
    /// Delta encoder
    delta_encoder: DeltaEncoder,
}

impl MultiModalEncoder {
    /// Create a new multi-modal encoder
    pub fn new() -> Self {
        Self {
            rate_encoder: RateEncoder::new(),
            temporal_encoder: TemporalEncoder::new(),
            population_encoder: PopulationEncoder::new(),
            delta_encoder: DeltaEncoder::new(),
        }
    }
    
    /// Encode using specific strategy
    pub fn encode_with_strategy<C>(
        &mut self,
        encoding_type: EncodingType,
        value: f32,
        time: Time,
        neuron_ids: &[NeuronId],
        config: &C,
    ) -> Result<Vec<SpikeTrain>>
    where
        C: 'static,
    {
        match encoding_type {
            EncodingType::Rate => {
                if let Some(config) = (config as &dyn core::any::Any).downcast_ref::<RateEncodingConfig>() {
                    self.rate_encoder.encode_value(value, time, neuron_ids, config)
                } else {
                    Err(SHNNError::encoding_error("Invalid config type for rate encoding"))
                }
            }
            EncodingType::Temporal => {
                if let Some(config) = (config as &dyn core::any::Any).downcast_ref::<TemporalEncodingConfig>() {
                    self.temporal_encoder.encode_value(value, time, neuron_ids, config)
                } else {
                    Err(SHNNError::encoding_error("Invalid config type for temporal encoding"))
                }
            }
            EncodingType::Population => {
                if let Some(config) = (config as &dyn core::any::Any).downcast_ref::<PopulationEncodingConfig>() {
                    self.population_encoder.encode_value(value, time, neuron_ids, config)
                } else {
                    Err(SHNNError::encoding_error("Invalid config type for population encoding"))
                }
            }
            _ => Err(SHNNError::encoding_error("Encoding type not yet implemented")),
        }
    }
    
    /// Reset all encoder states
    pub fn reset(&mut self) {
        self.delta_encoder.reset();
        // Other encoders are stateless or have their own reset methods
    }
}

impl Default for MultiModalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Sensory preprocessing for encoding
#[derive(Debug, Clone)]
pub struct SensorPreprocessor {
    /// Input scaling factors
    scaling: Vec<f32>,
    /// Input offsets
    offsets: Vec<f32>,
    /// Low-pass filter coefficients
    filter_coeffs: Vec<f32>,
    /// Filter states
    filter_states: Vec<f32>,
}

impl SensorPreprocessor {
    /// Create a new preprocessor
    pub fn new(num_channels: usize) -> Self {
        Self {
            scaling: vec![1.0; num_channels],
            offsets: vec![0.0; num_channels],
            filter_coeffs: vec![1.0; num_channels], // No filtering by default
            filter_states: vec![0.0; num_channels],
        }
    }
    
    /// Set scaling and offset for a channel
    pub fn set_channel_params(&mut self, channel: usize, scale: f32, offset: f32) {
        if channel < self.scaling.len() {
            self.scaling[channel] = scale;
            self.offsets[channel] = offset;
        }
    }
    
    /// Set low-pass filter coefficient for a channel
    pub fn set_filter_coeff(&mut self, channel: usize, coeff: f32) {
        if channel < self.filter_coeffs.len() {
            self.filter_coeffs[channel] = coeff.max(0.0).min(1.0);
        }
    }
    
    /// Process a sample
    pub fn process_sample(&mut self, channel: usize, sample: f32) -> f32 {
        if channel >= self.scaling.len() {
            return sample;
        }
        
        // Scale and offset
        let scaled = sample * self.scaling[channel] + self.offsets[channel];
        
        // Low-pass filter
        let alpha = self.filter_coeffs[channel];
        self.filter_states[channel] = alpha * scaled + (1.0 - alpha) * self.filter_states[channel];
        
        self.filter_states[channel]
    }
    
    /// Process multiple samples
    pub fn process_samples(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter().enumerate()
            .map(|(i, &sample)| self.process_sample(i, sample))
            .collect()
    }
    
    /// Reset filter states
    pub fn reset(&mut self) {
        self.filter_states.fill(0.0);
    }
}

impl Default for SensorPreprocessor {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rate_encoder() {
        let encoder = RateEncoder::new();
        let config = RateEncodingConfig::default();
        let neuron_ids = vec![NeuronId::new(0)];
        
        let trains = encoder.encode_value(0.5, Time::ZERO, &neuron_ids, &config).unwrap();
        assert_eq!(trains.len(), 1);
        
        // Higher values should generally produce more spikes
        let trains_low = encoder.encode_value(0.1, Time::ZERO, &neuron_ids, &config).unwrap();
        let trains_high = encoder.encode_value(0.9, Time::ZERO, &neuron_ids, &config).unwrap();
        
        // This is probabilistic, so we can't guarantee exact counts
        // but we can check that encoding succeeds
        assert!(trains_low.len() <= 1);
        assert!(trains_high.len() <= 1);
    }
    
    #[test]
    fn test_temporal_encoder() {
        let encoder = TemporalEncoder::new();
        let config = TemporalEncodingConfig::default();
        let neuron_ids = vec![NeuronId::new(0)];
        
        let trains = encoder.encode_value(0.5, Time::ZERO, &neuron_ids, &config).unwrap();
        assert_eq!(trains.len(), 1);
        assert_eq!(trains[0].len(), 1);
        
        // High values should spike earlier than low values (with invert_mapping=true)
        let trains_low = encoder.encode_value(0.1, Time::ZERO, &neuron_ids, &config).unwrap();
        let trains_high = encoder.encode_value(0.9, Time::ZERO, &neuron_ids, &config).unwrap();
        
        if !trains_low.is_empty() && !trains_high.is_empty() {
            assert!(trains_high[0].timestamps[0] < trains_low[0].timestamps[0]);
        }
    }
    
    #[test]
    fn test_population_encoder() {
        let encoder = PopulationEncoder::new();
        let config = PopulationEncodingConfig::default();
        let neuron_ids: Vec<_> = (0..5).map(NeuronId::new).collect();
        
        let trains = encoder.encode_value(0.5, Time::ZERO, &neuron_ids, &config).unwrap();
        
        // Should produce spike trains for neurons near the center (0.5)
        // This is probabilistic, so we mainly check it doesn't crash
        assert!(trains.len() <= neuron_ids.len());
    }
    
    #[test]
    fn test_delta_encoder() {
        let mut encoder = DeltaEncoder::new();
        let config = DeltaEncodingConfig::default();
        
        // First value establishes baseline
        let spikes1 = encoder.encode_channel(
            0, 0.0, Time::ZERO, 
            NeuronId::new(0), NeuronId::new(1), 
            &config
        ).unwrap();
        assert!(spikes1.is_empty()); // No previous value
        
        // Positive change
        let spikes2 = encoder.encode_channel(
            0, 0.05, Time::from_millis(1),
            NeuronId::new(0), NeuronId::new(1),
            &config
        ).unwrap();
        assert_eq!(spikes2.len(), 1);
        assert_eq!(spikes2[0].source, NeuronId::new(0)); // Positive neuron
        
        // Negative change
        let spikes3 = encoder.encode_channel(
            0, -0.05, Time::from_millis(2),
            NeuronId::new(0), NeuronId::new(1),
            &config
        ).unwrap();
        assert_eq!(spikes3.len(), 1);
        assert_eq!(spikes3[0].source, NeuronId::new(1)); // Negative neuron
    }
    
    #[test]
    fn test_sensor_preprocessor() {
        let mut preprocessor = SensorPreprocessor::new(2);
        
        // Set scaling for channel 0
        preprocessor.set_channel_params(0, 2.0, 1.0);
        
        let result = preprocessor.process_sample(0, 1.0);
        assert_eq!(result, 3.0); // 1.0 * 2.0 + 1.0
        
        // Test filtering
        preprocessor.set_filter_coeff(0, 0.5);
        preprocessor.reset();
        
        let result1 = preprocessor.process_sample(0, 2.0); // (2*2+1) = 5
        let result2 = preprocessor.process_sample(0, 0.0); // (0*2+1) = 1
        
        // Should be filtered
        assert!(result1 < 5.0); // Due to filtering
        assert!(result2 > 1.0); // Due to filter memory
    }
    
    #[test]
    fn test_encoding_sequence() {
        let encoder = TemporalEncoder::new();
        let config = TemporalEncodingConfig::default();
        let neuron_ids = vec![NeuronId::new(0)];
        
        let values = vec![0.1, 0.5, 0.9];
        let trains = encoder.encode_sequence(
            &values,
            Time::ZERO,
            Duration::from_millis(10),
            &neuron_ids,
            &config,
        ).unwrap();
        
        assert_eq!(trains.len(), 1);
        assert_eq!(trains[0].len(), 3); // Three spikes for three values
        
        // Timestamps should be sorted
        for i in 1..trains[0].timestamps.len() {
            assert!(trains[0].timestamps[i] >= trains[0].timestamps[i-1]);
        }
    }
}