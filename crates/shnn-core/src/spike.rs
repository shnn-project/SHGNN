//! Spike representation and processing for neuromorphic computation
//!
//! This module provides the fundamental spike data structures and operations
//! for event-driven neural computation.

use crate::{
    error::{Result, SHNNError},
    time::Time,
};
use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Unique identifier for a neuron in the network
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuronId(pub u32);

impl NeuronId {
    /// Create a new neuron ID
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub const fn raw(&self) -> u32 {
        self.0
    }
    
    /// Invalid neuron ID constant
    pub const INVALID: Self = Self(u32::MAX);
    
    /// Check if this is a valid neuron ID
    pub const fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
    
    /// Convert to usize for array indexing (test helper)
    #[cfg(test)]
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for NeuronId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "N{}", self.0)
    }
}

impl From<u32> for NeuronId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<NeuronId> for u32 {
    fn from(id: NeuronId) -> Self {
        id.0
    }
}

/// A neural spike event
///
/// Represents a discrete spike event with source neuron, timestamp, and amplitude.
/// This is the fundamental unit of information in neuromorphic computation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Spike {
    /// The neuron that generated this spike
    pub source: NeuronId,
    /// When the spike occurred
    pub timestamp: Time,
    /// Spike amplitude (typically 1.0 for binary spikes)
    pub amplitude: f32,
}

impl Spike {
    /// Create a new spike
    pub fn new(source: NeuronId, timestamp: Time, amplitude: f32) -> Result<Self> {
        if !source.is_valid() {
            return Err(SHNNError::invalid_spike("Invalid source neuron ID"));
        }
        
        if !amplitude.is_finite() || amplitude < 0.0 {
            return Err(SHNNError::invalid_spike("Invalid spike amplitude"));
        }
        
        Ok(Self {
            source,
            timestamp,
            amplitude,
        })
    }
    
    /// Create a binary spike (amplitude = 1.0)
    pub fn binary(source: NeuronId, timestamp: Time) -> Result<Self> {
        Self::new(source, timestamp, 1.0)
    }
    
    /// Create a spike with custom amplitude
    pub fn with_amplitude(source: NeuronId, timestamp: Time, amplitude: f32) -> Result<Self> {
        Self::new(source, timestamp, amplitude)
    }
    
    /// Check if this is a binary spike
    pub fn is_binary(&self) -> bool {
        (self.amplitude - 1.0).abs() < f32::EPSILON
    }
    
    /// Get the energy of this spike (amplitude squared)
    pub fn energy(&self) -> f32 {
        self.amplitude * self.amplitude
    }
    
    /// Scale the spike amplitude
    pub fn scale(&mut self, factor: f32) -> Result<()> {
        if !factor.is_finite() || factor < 0.0 {
            return Err(SHNNError::invalid_spike("Invalid scale factor"));
        }
        self.amplitude *= factor;
        Ok(())
    }
    
    /// Create a scaled copy of this spike
    pub fn scaled(&self, factor: f32) -> Result<Self> {
        let mut spike = self.clone();
        spike.scale(factor)?;
        Ok(spike)
    }
}

impl fmt::Display for Spike {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Spike({} @ {} amp={:.3})",
            self.source, self.timestamp, self.amplitude
        )
    }
}

/// A spike with additional timing information for delayed processing
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimedSpike {
    /// The base spike
    pub spike: Spike,
    /// When this spike should be delivered
    pub delivery_time: Time,
    /// Processing delay
    pub delay: crate::time::Duration,
}

impl TimedSpike {
    /// Create a new timed spike
    pub fn new(spike: Spike, delivery_time: Time) -> Self {
        let delay = delivery_time - spike.timestamp; // This returns Duration
        Self {
            spike,
            delivery_time,
            delay,
        }
    }
    
    /// Create a timed spike with explicit delay
    pub fn with_delay(spike: Spike, delay: crate::time::Duration) -> Self {
        let delivery_time = spike.timestamp + delay; // Time + Duration = Time
        Self {
            spike,
            delivery_time,
            delay,
        }
    }
    
    /// Check if this spike is ready for delivery at the given time
    pub fn is_ready(&self, current_time: Time) -> bool {
        current_time >= self.delivery_time
    }
    
    /// Get remaining time until delivery
    pub fn time_until_delivery(&self, current_time: Time) -> crate::time::Duration {
        if current_time >= self.delivery_time {
            crate::time::Duration::ZERO
        } else {
            self.delivery_time - current_time
        }
    }
}

impl From<Spike> for TimedSpike {
    fn from(spike: Spike) -> Self {
        Self::new(spike.clone(), spike.timestamp)
    }
}

/// Target specification for spike routing
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SpikeTarget {
    /// Single neuron target
    Neuron(NeuronId),
    /// Multiple neuron targets
    Multiple(Vec<NeuronId>),
    /// Broadcast to all neurons in a range
    Range {
        /// Starting neuron ID of the connection
        start: NeuronId,
        /// Ending neuron ID of the connection
        end: NeuronId,
    },
    /// Custom target selection function (not serializable)
    #[cfg_attr(feature = "serde", serde(skip))]
    Custom(fn(NeuronId) -> bool),
}

impl SpikeTarget {
    /// Create a single neuron target
    pub fn neuron(id: NeuronId) -> Self {
        Self::Neuron(id)
    }
    
    /// Create a multiple neuron target
    pub fn multiple(ids: Vec<NeuronId>) -> Self {
        Self::Multiple(ids)
    }
    
    /// Create a range target
    pub fn range(start: NeuronId, end: NeuronId) -> Result<Self> {
        if start > end {
            return Err(SHNNError::invalid_spike("Invalid neuron range"));
        }
        Ok(Self::Range { start, end })
    }
    
    /// Create a custom target
    pub fn custom(selector: fn(NeuronId) -> bool) -> Self {
        Self::Custom(selector)
    }
    
    /// Check if a neuron ID matches this target
    pub fn matches(&self, neuron_id: NeuronId) -> bool {
        match self {
            Self::Neuron(id) => *id == neuron_id,
            Self::Multiple(ids) => ids.contains(&neuron_id),
            Self::Range { start, end } => neuron_id >= *start && neuron_id <= *end,
            Self::Custom(selector) => selector(neuron_id),
        }
    }
    
    /// Get all matching neuron IDs within a given range
    pub fn resolve(&self, max_id: NeuronId) -> Vec<NeuronId> {
        match self {
            Self::Neuron(id) => {
                if id.raw() <= max_id.raw() {
                    vec![*id]
                } else {
                    vec![]
                }
            }
            Self::Multiple(ids) => {
                ids.iter()
                    .filter(|id| id.raw() <= max_id.raw())
                    .cloned()
                    .collect()
            }
            Self::Range { start, end } => {
                let actual_end = NeuronId(end.raw().min(max_id.raw()));
                (start.raw()..=actual_end.raw())
                    .map(NeuronId::new)
                    .collect()
            }
            Self::Custom(selector) => {
                (0..=max_id.raw())
                    .map(NeuronId::new)
                    .filter(|id| selector(*id))
                    .collect()
            }
        }
    }
    
    /// Get estimated target count
    pub fn estimated_count(&self, max_id: NeuronId) -> usize {
        match self {
            Self::Neuron(_) => 1,
            Self::Multiple(ids) => ids.len(),
            Self::Range { start, end } => {
                let actual_end = end.raw().min(max_id.raw());
                if actual_end >= start.raw() {
                    (actual_end - start.raw() + 1) as usize
                } else {
                    0
                }
            }
            Self::Custom(_) => (max_id.raw() + 1) as usize, // Conservative estimate
        }
    }
}

/// Spike train - a sequence of spikes from the same source
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpikeTrain {
    /// Source neuron
    pub source: NeuronId,
    /// Sequence of spike times
    pub timestamps: Vec<Time>,
    /// Amplitude for each spike (None means all binary spikes)
    pub amplitudes: Option<Vec<f32>>,
}

impl SpikeTrain {
    /// Create a new spike train with binary spikes
    pub fn binary(source: NeuronId, timestamps: Vec<Time>) -> Result<Self> {
        if timestamps.is_empty() {
            return Err(SHNNError::invalid_spike("Empty spike train"));
        }
        
        // Check timestamps are sorted
        for window in timestamps.windows(2) {
            if window[1] < window[0] {
                return Err(SHNNError::invalid_spike("Spike train timestamps not sorted"));
            }
        }
        
        Ok(Self {
            source,
            timestamps,
            amplitudes: None,
        })
    }
    
    /// Create a new spike train with variable amplitudes
    pub fn with_amplitudes(
        source: NeuronId,
        timestamps: Vec<Time>,
        amplitudes: Vec<f32>,
    ) -> Result<Self> {
        if timestamps.len() != amplitudes.len() {
            return Err(SHNNError::invalid_spike(
                "Timestamp and amplitude vectors must have same length"
            ));
        }
        
        if timestamps.is_empty() {
            return Err(SHNNError::invalid_spike("Empty spike train"));
        }
        
        // Check timestamps are sorted
        for window in timestamps.windows(2) {
            if window[1] < window[0] {
                return Err(SHNNError::invalid_spike("Spike train timestamps not sorted"));
            }
        }
        
        // Check amplitudes are valid
        for &amp in &amplitudes {
            if !amp.is_finite() || amp < 0.0 {
                return Err(SHNNError::invalid_spike("Invalid spike amplitude"));
            }
        }
        
        Ok(Self {
            source,
            timestamps,
            amplitudes: Some(amplitudes),
        })
    }
    
    /// Get the number of spikes in the train
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }
    
    /// Check if the spike train is empty
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
    
    /// Get the spike at a specific index
    pub fn get_spike(&self, index: usize) -> Option<Spike> {
        if index >= self.timestamps.len() {
            return None;
        }
        
        let amplitude = self.amplitudes
            .as_ref()
            .map(|amps| amps[index])
            .unwrap_or(1.0);
        
        Spike::new(self.source, self.timestamps[index], amplitude).ok()
    }
    
    /// Convert to a vector of individual spikes
    pub fn to_spikes(&self) -> Result<Vec<Spike>> {
        let mut spikes = Vec::with_capacity(self.timestamps.len());
        
        for i in 0..self.timestamps.len() {
            let amplitude = self.amplitudes
                .as_ref()
                .map(|amps| amps[i])
                .unwrap_or(1.0);
            
            spikes.push(Spike::new(self.source, self.timestamps[i], amplitude)?);
        }
        
        Ok(spikes)
    }
    
    /// Get the inter-spike intervals
    pub fn inter_spike_intervals(&self) -> Vec<crate::time::Duration> {
        self.timestamps
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect()
    }
    
    /// Get the firing rate (spikes per second) over the total duration
    pub fn firing_rate(&self) -> f64 {
        if self.timestamps.len() < 2 {
            return 0.0;
        }
        
        let duration = (*self.timestamps.last().unwrap() - self.timestamps[0]).as_secs_f64();
        if duration > 0.0 {
            (self.timestamps.len() - 1) as f64 / duration
        } else {
            0.0
        }
    }
    
    /// Get the coefficient of variation of inter-spike intervals
    pub fn cv_isi(&self) -> f64 {
        let intervals = self.inter_spike_intervals();
        if intervals.len() < 2 {
            return 0.0;
        }
        
        let mean = intervals.iter().map(|d| d.as_secs_f64()).sum::<f64>() / intervals.len() as f64;
        
        if mean == 0.0 {
            return 0.0;
        }
        
        let variance = intervals
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>() / intervals.len() as f64;
        
        variance.sqrt() / mean
    }
}

impl fmt::Display for SpikeTrain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpikeTrain({}, {} spikes, rate={:.1} Hz)",
            self.source,
            self.len(),
            self.firing_rate()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neuron_id() {
        let id = NeuronId::new(42);
        assert_eq!(id.raw(), 42);
        assert!(id.is_valid());
        assert_eq!(format!("{}", id), "N42");
        
        assert!(!NeuronId::INVALID.is_valid());
    }
    
    #[test]
    fn test_spike_creation() {
        let spike = Spike::binary(NeuronId::new(1), Time::from_millis(100)).unwrap();
        assert_eq!(spike.source, NeuronId::new(1));
        assert_eq!(spike.timestamp, Time::from_millis(100));
        assert_eq!(spike.amplitude, 1.0);
        assert!(spike.is_binary());
    }
    
    #[test]
    fn test_spike_scaling() {
        let mut spike = Spike::binary(NeuronId::new(1), Time::from_millis(100)).unwrap();
        spike.scale(0.5).unwrap();
        assert_eq!(spike.amplitude, 0.5);
        
        let scaled = spike.scaled(2.0).unwrap();
        assert_eq!(scaled.amplitude, 1.0);
    }
    
    #[test]
    fn test_timed_spike() {
        let spike = Spike::binary(NeuronId::new(1), Time::from_millis(100)).unwrap();
        let timed = TimedSpike::with_delay(spike, Duration::from_millis(10));
        
        assert_eq!(timed.delivery_time, Time::from_millis(110));
        assert!(!timed.is_ready(Time::from_millis(105)));
        assert!(timed.is_ready(Time::from_millis(110)));
    }
    
    #[test]
    fn test_spike_target() {
        let target = SpikeTarget::range(NeuronId::new(10), NeuronId::new(20)).unwrap();
        assert!(target.matches(NeuronId::new(15)));
        assert!(!target.matches(NeuronId::new(5)));
        
        let resolved = target.resolve(NeuronId::new(100));
        assert_eq!(resolved.len(), 11); // 10-20 inclusive
    }
    
    #[test]
    fn test_spike_train() {
        let timestamps = vec![
            Time::from_millis(100),
            Time::from_millis(200),
            Time::from_millis(300),
        ];
        
        let train = SpikeTrain::binary(NeuronId::new(1), timestamps).unwrap();
        assert_eq!(train.len(), 3);
        assert_eq!(train.firing_rate(), 10.0); // 2 intervals over 0.2 seconds
        
        let spikes = train.to_spikes().unwrap();
        assert_eq!(spikes.len(), 3);
        assert!(spikes[0].is_binary());
    }
    
    #[test]
    fn test_spike_train_isi() {
        let timestamps = vec![
            Time::from_millis(0),
            Time::from_millis(100),
            Time::from_millis(200),
        ];
        
        let train = SpikeTrain::binary(NeuronId::new(1), timestamps).unwrap();
        let intervals = train.inter_spike_intervals();
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0], Duration::from_millis(100));
        assert_eq!(intervals[1], Duration::from_millis(100));
        
        // Perfectly regular -> CV should be 0
        assert_eq!(train.cv_isi(), 0.0);
    }
}