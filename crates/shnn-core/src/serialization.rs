//! Serialization utilities for SHNN data structures
//!
//! This module provides efficient serialization and deserialization
//! for neural network states, spike trains, and hypergraph structures
//! using our custom zero-dependency serialization library.

use crate::{
    error::{Result, SHNNError},
    spike::{Spike, SpikeTrain, NeuronId},
    time::Time,
    neuron::NeuronState,
};

// Use our custom zero-dependency serialization
#[cfg(feature = "serialize")]
use shnn_serialize::{
    Serialize, Deserialize, BinaryEncoder, BinaryDecoder,
    Buffer, BufferMut, ZeroCopyBuffer,
    neural::{SpikeEvent, WeightMatrix, LayerState, NeuralSerializer},
};

// Legacy compatibility for serde
#[cfg(feature = "legacy-serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use heapless::FnvIndexMap as HashMap;

/// Serializable network snapshot containing complete state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NetworkSnapshot {
    /// Timestamp when snapshot was taken
    pub timestamp: Time,
    /// All neuron states indexed by ID
    pub neuron_states: HashMap<NeuronId, NeuronState>,
    /// Active spike trains
    pub spike_trains: Vec<SpikeTrain>,
    /// Version for compatibility checking
    pub version: u32,
}

impl NetworkSnapshot {
    /// Create a new network snapshot
    pub fn new(timestamp: Time) -> Self {
        Self {
            timestamp,
            neuron_states: HashMap::new(),
            spike_trains: Vec::new(),
            version: 1,
        }
    }
    
    /// Add neuron state to snapshot
    pub fn add_neuron_state(&mut self, id: NeuronId, state: NeuronState) {
        #[cfg(feature = "std")]
        {
            self.neuron_states.insert(id, state);
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = self.neuron_states.insert(id, state);
        }
    }
    
    /// Add spike train to snapshot
    pub fn add_spike_train(&mut self, train: SpikeTrain) {
        self.spike_trains.push(train);
    }
    
    
    
    /// Validate snapshot integrity
    pub fn validate(&self) -> Result<()> {
        if self.version == 0 {
            return Err(SHNNError::generic("Invalid version"));
        }
        
        // Validate spike trains
        for train in &self.spike_trains {
            if !train.source.is_valid() {
                return Err(SHNNError::generic("Invalid spike train source"));
            }
        }
        
        Ok(())
    }
}

/// Compact binary format for spike data
#[derive(Debug, Clone)]
pub struct CompactSpikeData {
    /// Source neuron ID
    pub source: u32,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Amplitude as fixed-point u16
    pub amplitude_fp: u16,
}

impl CompactSpikeData {
    /// Convert from regular Spike
    pub fn from_spike(spike: &Spike) -> Self {
        Self {
            source: spike.source.raw(),
            timestamp_ns: spike.timestamp.as_nanos(),
            amplitude_fp: (spike.amplitude * 1000.0) as u16, // Fixed-point with 3 decimal places
        }
    }
    
    /// Convert to regular Spike
    pub fn to_spike(&self) -> Result<Spike> {
        Spike::new(
            NeuronId::new(self.source),
            Time::from_nanos(self.timestamp_ns),
            self.amplitude_fp as f32 / 1000.0,
        )
    }
}

/// Efficient spike train compression for storage
pub fn compress_spike_train(train: &SpikeTrain) -> Vec<CompactSpikeData> {
    train.timestamps.iter()
        .enumerate()
        .map(|(i, &timestamp)| {
            let amplitude = train.amplitudes
                .as_ref()
                .and_then(|amps| amps.get(i))
                .copied()
                .unwrap_or(1.0);
            
            // Create temporary spike for conversion
            let spike = Spike {
                source: train.source,
                timestamp,
                amplitude,
            };
            
            CompactSpikeData::from_spike(&spike)
        })
        .collect()
}

/// Decompress spike train from compact format
pub fn decompress_spike_train(
    source: NeuronId,
    compact_data: &[CompactSpikeData],
) -> Result<SpikeTrain> {
    let mut timestamps = Vec::new();
    let mut amplitudes = Vec::new();
    
    for compact in compact_data {
        let spike = compact.to_spike()?;
        timestamps.push(spike.timestamp);
        amplitudes.push(spike.amplitude);
    }
    
    Ok(SpikeTrain {
        source,
        timestamps,
        amplitudes: if amplitudes.iter().all(|&a| a == 1.0) {
            None // All amplitudes are default, save space
        } else {
            Some(amplitudes)
        },
    })
}

/// JSON serialization helpers
#[cfg(feature = "serde_json")]
pub mod json {
    use super::*;
    
    /// Serialize network snapshot to JSON string
    pub fn serialize_snapshot(snapshot: &NetworkSnapshot) -> Result<String> {
        serde_json::to_string(snapshot)
            .map_err(|e| SHNNError::serialization_error(format!("JSON serialization failed: {}", e)))
    }
    
    /// Deserialize network snapshot from JSON string
    pub fn deserialize_snapshot(json: &str) -> Result<NetworkSnapshot> {
        serde_json::from_str(json)
            .map_err(|e| SHNNError::serialization_error(format!("JSON deserialization failed: {}", e)))
    }
    
    /// Serialize spike train to JSON
    pub fn serialize_spike_train(train: &SpikeTrain) -> Result<String> {
        serde_json::to_string(train)
            .map_err(|e| SHNNError::serialization_error(format!("Spike train serialization failed: {}", e)))
    }
}

/// Binary serialization helpers
#[cfg(feature = "bincode")]
pub mod binary {
    use super::*;
    
    /// Serialize network snapshot to binary format
    pub fn serialize_snapshot(snapshot: &NetworkSnapshot) -> Result<Vec<u8>> {
        bincode::serialize(snapshot)
            .map_err(|e| SHNNError::serialization_error(format!("Binary serialization failed: {}", e)))
    }
    
    /// Deserialize network snapshot from binary format
    pub fn deserialize_snapshot(data: &[u8]) -> Result<NetworkSnapshot> {
        bincode::deserialize(data)
            .map_err(|e| SHNNError::serialization_error(format!("Binary deserialization failed: {}", e)))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_snapshot() {
        let mut snapshot = NetworkSnapshot::new(Time::from_millis(100));
        
        let neuron_id = NeuronId::new(1);
        let state = NeuronState::new();
        snapshot.add_neuron_state(neuron_id, state);
        
        assert!(snapshot.validate().is_ok());
        assert_eq!(snapshot.timestamp, Time::from_millis(100));
    }
    
    #[test]
    fn test_compact_spike_data() {
        let spike = Spike::new(
            NeuronId::new(42),
            Time::from_millis(1500),
            2.5,
        ).unwrap();
        
        let compact = CompactSpikeData::from_spike(&spike);
        let recovered = compact.to_spike().unwrap();
        
        assert_eq!(spike.source, recovered.source);
        assert_eq!(spike.timestamp, recovered.timestamp);
        assert!((spike.amplitude - recovered.amplitude).abs() < 0.001);
    }
}