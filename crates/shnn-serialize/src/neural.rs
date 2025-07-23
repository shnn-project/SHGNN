//! Neural network specific serialization
//! 
//! Provides specialized serialization for neuromorphic data structures including
//! spike events, weight matrices, and neural network states with zero-copy optimizations.

use crate::{
    Buffer, BufferMut, Result, SerializeError, Serialize, Deserialize,
    BinaryEncoder, BinaryDecoder, ZeroCopySerialize,
};
#[cfg(feature = "std")]
use std::{vec::Vec, string::String};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String};

/// Spike event for neuromorphic computation
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeEvent {
    /// Neuron ID that fired
    pub neuron_id: u32,
    /// Timestamp of the spike (in simulation time units)
    pub timestamp: f32,
    /// Spike amplitude/weight
    pub amplitude: f32,
    /// Layer index
    pub layer: u16,
    /// Reserved for future use
    pub reserved: u16,
}

impl ZeroCopySerialize for SpikeEvent {}

impl Serialize for SpikeEvent {
    fn serialize(&self, buffer: &mut [u8]) -> Result<usize> {
        if buffer.len() < self.serialized_size() {
            return Err(SerializeError::BufferTooSmall);
        }
        
        let bytes = self.as_bytes();
        buffer[..bytes.len()].copy_from_slice(bytes);
        Ok(bytes.len())
    }
    
    fn serialized_size(&self) -> usize {
        core::mem::size_of::<Self>()
    }
}

impl<'a> Deserialize<'a> for SpikeEvent {
    fn deserialize(buffer: &'a [u8]) -> Result<Self> {
        let spike_ref = Self::from_bytes(buffer)?;
        Ok(*spike_ref)
    }
}

/// Weight matrix for neural connections
#[derive(Debug, Clone)]
pub struct WeightMatrix {
    /// Matrix dimensions (rows, cols)
    pub shape: (u32, u32),
    /// Flattened weight data
    pub weights: Vec<f32>,
    /// Learning rate for this matrix
    pub learning_rate: f32,
}

impl WeightMatrix {
    /// Create a new weight matrix
    pub fn new(rows: u32, cols: u32) -> Self {
        let size = (rows * cols) as usize;
        Self {
            shape: (rows, cols),
            weights: vec![0.0; size],
            learning_rate: 0.01,
        }
    }

    /// Get weight at position (row, col)
    pub fn get(&self, row: u32, col: u32) -> Result<f32> {
        let idx = (row * self.shape.1 + col) as usize;
        self.weights.get(idx).copied()
            .ok_or(SerializeError::IndexOutOfBounds)
    }

    /// Set weight at position (row, col)
    pub fn set(&mut self, row: u32, col: u32, value: f32) -> Result<()> {
        let idx = (row * self.shape.1 + col) as usize;
        if idx < self.weights.len() {
            self.weights[idx] = value;
            Ok(())
        } else {
            Err(SerializeError::IndexOutOfBounds)
        }
    }

    /// Get total number of weights
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if matrix is empty
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

impl Serialize for WeightMatrix {
    fn serialize(&self, buffer: &mut [u8]) -> Result<usize> {
        let mut encoder = BinaryEncoder::new(BufferMut::new(buffer));
        
        // Write shape
        encoder.encode(&self.shape.0)?;
        encoder.encode(&self.shape.1)?;
        
        // Write learning rate
        encoder.encode(&self.learning_rate)?;
        
        // Write weights
        encoder.encode_slice(&self.weights)?;
        
        Ok(encoder.position())
    }
    
    fn serialized_size(&self) -> usize {
        4 + 4 + 4 + 4 + (self.weights.len() * 4) // shape + lr + len + weights
    }
}

impl<'a> Deserialize<'a> for WeightMatrix {
    fn deserialize(buffer: &'a [u8]) -> Result<Self> {
        let mut decoder = BinaryDecoder::new(Buffer::new(buffer));
        
        // Read shape
        let rows: u32 = decoder.decode()?;
        let cols: u32 = decoder.decode()?;
        
        // Read learning rate
        let learning_rate: f32 = decoder.decode()?;
        
        // Read weights
        let weights_slice: &[f32] = decoder.decode_slice()?;
        let weights = weights_slice.to_vec();
        
        Ok(Self {
            shape: (rows, cols),
            weights,
            learning_rate,
        })
    }
}

/// Neural network layer state
#[derive(Debug, Clone)]
pub struct LayerState {
    /// Layer index
    pub layer_id: u32,
    /// Neuron membrane potentials
    pub potentials: Vec<f32>,
    /// Neuron activation states
    pub activations: Vec<f32>,
    /// Spike history buffer
    pub spike_history: Vec<SpikeEvent>,
    /// Last update timestamp
    pub last_update: f32,
}

impl LayerState {
    /// Create new layer state
    pub fn new(layer_id: u32, neuron_count: usize) -> Self {
        Self {
            layer_id,
            potentials: vec![0.0; neuron_count],
            activations: vec![0.0; neuron_count],
            spike_history: Vec::new(),
            last_update: 0.0,
        }
    }

    /// Add spike to history
    pub fn add_spike(&mut self, spike: SpikeEvent) {
        self.spike_history.push(spike);
    }

    /// Clear old spikes before timestamp
    pub fn clear_old_spikes(&mut self, before_time: f32) {
        self.spike_history.retain(|spike| spike.timestamp >= before_time);
    }

    /// Get neuron count
    pub fn neuron_count(&self) -> usize {
        self.potentials.len()
    }
}

impl Serialize for LayerState {
    fn serialize(&self, buffer: &mut [u8]) -> Result<usize> {
        let mut encoder = BinaryEncoder::new(BufferMut::new(buffer));
        
        // Write layer ID
        encoder.encode(&self.layer_id)?;
        
        // Write last update time
        encoder.encode(&self.last_update)?;
        
        // Write potentials
        encoder.encode_slice(&self.potentials)?;
        
        // Write activations  
        encoder.encode_slice(&self.activations)?;
        
        // Write spike history
        encoder.encode_slice(&self.spike_history)?;
        
        Ok(encoder.position())
    }
    
    fn serialized_size(&self) -> usize {
        4 + 4 + // layer_id + last_update
        4 + (self.potentials.len() * 4) + // potentials
        4 + (self.activations.len() * 4) + // activations
        4 + (self.spike_history.len() * core::mem::size_of::<SpikeEvent>()) // spike_history
    }
}

impl<'a> Deserialize<'a> for LayerState {
    fn deserialize(buffer: &'a [u8]) -> Result<Self> {
        let mut decoder = BinaryDecoder::new(Buffer::new(buffer));
        
        // Read layer ID
        let layer_id: u32 = decoder.decode()?;
        
        // Read last update time
        let last_update: f32 = decoder.decode()?;
        
        // Read potentials
        let potentials_slice: &[f32] = decoder.decode_slice()?;
        let potentials = potentials_slice.to_vec();
        
        // Read activations
        let activations_slice: &[f32] = decoder.decode_slice()?;
        let activations = activations_slice.to_vec();
        
        // Read spike history
        let spike_history_slice: &[SpikeEvent] = decoder.decode_slice()?;
        let spike_history = spike_history_slice.to_vec();
        
        Ok(Self {
            layer_id,
            potentials,
            activations,
            spike_history,
            last_update,
        })
    }
}

/// Neural network serializer with compression and optimization
#[derive(Debug)]
pub struct NeuralSerializer {
    /// Enable compression for large weight matrices
    pub compress_weights: bool,
    /// Precision for weight quantization (0 = no quantization)
    pub weight_precision: u8,
    /// Maximum spike history length to serialize
    pub max_spike_history: usize,
}

impl NeuralSerializer {
    /// Create new neural serializer with default settings
    pub fn new() -> Self {
        Self {
            compress_weights: false,
            weight_precision: 0,
            max_spike_history: 1000,
        }
    }

    /// Enable weight compression
    pub fn with_compression(mut self) -> Self {
        self.compress_weights = true;
        self
    }

    /// Set weight quantization precision (bits)
    pub fn with_precision(mut self, bits: u8) -> Self {
        self.weight_precision = bits;
        self
    }

    /// Set maximum spike history length
    pub fn with_max_history(mut self, max_len: usize) -> Self {
        self.max_spike_history = max_len;
        self
    }

    /// Serialize neural network state
    pub fn serialize_network_state(
        &self,
        layers: &[LayerState],
        weights: &[WeightMatrix],
        buffer: &mut [u8],
    ) -> Result<usize> {
        let mut encoder = BinaryEncoder::new(BufferMut::new(buffer));
        
        // Write network metadata
        encoder.encode(&(layers.len() as u32))?;  // number of layers
        encoder.encode(&(weights.len() as u32))?; // number of weight matrices
        
        // Serialize layers
        for layer in layers {
            // Limit spike history if configured
            let mut layer_copy = layer.clone();
            if layer_copy.spike_history.len() > self.max_spike_history {
                layer_copy.spike_history.truncate(self.max_spike_history);
            }
            encoder.encode(&layer_copy)?;
        }
        
        // Serialize weight matrices
        for weight_matrix in weights {
            if self.compress_weights {
                // Simple compression: store as quantized integers
                self.serialize_compressed_weights(weight_matrix, &mut encoder)?;
            } else {
                encoder.encode(weight_matrix)?;
            }
        }
        
        Ok(encoder.position())
    }

    /// Deserialize neural network state
    pub fn deserialize_network_state(
        &self,
        buffer: &[u8],
    ) -> Result<(Vec<LayerState>, Vec<WeightMatrix>)> {
        let mut decoder = BinaryDecoder::new(Buffer::new(buffer));
        
        // Read network metadata
        let num_layers: u32 = decoder.decode()?;
        let num_weights: u32 = decoder.decode()?;
        
        // Deserialize layers
        let mut layers = Vec::with_capacity(num_layers as usize);
        for _ in 0..num_layers {
            layers.push(decoder.decode()?);
        }
        
        // Deserialize weight matrices
        let mut weights = Vec::with_capacity(num_weights as usize);
        for _ in 0..num_weights {
            if self.compress_weights {
                weights.push(self.deserialize_compressed_weights(&mut decoder)?);
            } else {
                weights.push(decoder.decode()?);
            }
        }
        
        Ok((layers, weights))
    }

    /// Serialize compressed weight matrix
    fn serialize_compressed_weights(
        &self,
        weights: &WeightMatrix,
        encoder: &mut BinaryEncoder,
    ) -> Result<()> {
        // For now, just use the regular serialization
        // In a full implementation, this would apply quantization/compression
        encoder.encode(weights)
    }

    /// Deserialize compressed weight matrix
    fn deserialize_compressed_weights(
        &self,
        decoder: &mut BinaryDecoder,
    ) -> Result<WeightMatrix> {
        // For now, just use the regular deserialization
        decoder.decode()
    }
}

impl Default for NeuralSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Codec for efficient spike event streaming
#[derive(Debug)]
pub struct SpikeEventCodec {
    /// Buffer for batching spike events
    buffer: Vec<SpikeEvent>,
    /// Maximum batch size
    batch_size: usize,
}

impl SpikeEventCodec {
    /// Create new spike event codec
    pub fn new(batch_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(batch_size),
            batch_size,
        }
    }

    /// Add spike to batch
    pub fn add_spike(&mut self, spike: SpikeEvent) {
        self.buffer.push(spike);
    }

    /// Check if batch is full
    pub fn is_batch_ready(&self) -> bool {
        self.buffer.len() >= self.batch_size
    }

    /// Serialize current batch
    pub fn serialize_batch(&mut self, output: &mut [u8]) -> Result<usize> {
        if self.buffer.is_empty() {
            return Ok(0);
        }

        let mut encoder = BinaryEncoder::new(BufferMut::new(output));
        encoder.encode_slice(&self.buffer)?;
        
        let size = encoder.position();
        self.buffer.clear();
        Ok(size)
    }

    /// Deserialize spike batch
    pub fn deserialize_batch(&self, input: &[u8]) -> Result<Vec<SpikeEvent>> {
        let mut decoder = BinaryDecoder::new(Buffer::new(input));
        let spikes: &[SpikeEvent] = decoder.decode_slice()?;
        Ok(spikes.to_vec())
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Codec for weight matrix streaming and updates
#[derive(Debug)]
pub struct WeightMatrixCodec {
    /// Delta compression threshold
    pub delta_threshold: f32,
}

impl WeightMatrixCodec {
    /// Create new weight matrix codec
    pub fn new() -> Self {
        Self {
            delta_threshold: 1e-6,
        }
    }

    /// Serialize weight matrix delta (changes only)
    pub fn serialize_delta(
        &self,
        old_weights: &WeightMatrix,
        new_weights: &WeightMatrix,
        output: &mut [u8],
    ) -> Result<usize> {
        if old_weights.shape != new_weights.shape {
            return Err(SerializeError::ShapeMismatch);
        }

        let mut encoder = BinaryEncoder::new(BufferMut::new(output));
        
        // Write shape for validation
        encoder.encode(&new_weights.shape.0)?;
        encoder.encode(&new_weights.shape.1)?;
        
        // Collect changed indices and values
        let mut changes = Vec::new();
        for (i, (&old_val, &new_val)) in old_weights.weights.iter()
            .zip(new_weights.weights.iter()).enumerate() {
            if (new_val - old_val).abs() > self.delta_threshold {
                changes.push((i as u32, new_val));
            }
        }
        
        // Write number of changes
        encoder.encode(&(changes.len() as u32))?;
        
        // Write changes
        for (index, value) in changes {
            encoder.encode(&index)?;
            encoder.encode(&value)?;
        }
        
        Ok(encoder.position())
    }

    /// Apply weight matrix delta
    pub fn apply_delta(
        &self,
        weights: &mut WeightMatrix,
        delta: &[u8],
    ) -> Result<()> {
        let mut decoder = BinaryDecoder::new(Buffer::new(delta));
        
        // Validate shape
        let rows: u32 = decoder.decode()?;
        let cols: u32 = decoder.decode()?;
        if weights.shape != (rows, cols) {
            return Err(SerializeError::ShapeMismatch);
        }
        
        // Read number of changes
        let num_changes: u32 = decoder.decode()?;
        
        // Apply changes
        for _ in 0..num_changes {
            let index: u32 = decoder.decode()?;
            let value: f32 = decoder.decode()?;
            
            if (index as usize) < weights.weights.len() {
                weights.weights[index as usize] = value;
            } else {
                return Err(SerializeError::IndexOutOfBounds);
            }
        }
        
        Ok(())
    }
}

impl Default for WeightMatrixCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_event_serialization() {
        let spike = SpikeEvent {
            neuron_id: 42,
            timestamp: 1.5,
            amplitude: 0.8,
            layer: 2,
            reserved: 0,
        };

        let mut buffer = [0u8; 64];
        let size = spike.serialize(&mut buffer).unwrap();
        
        let deserialized = SpikeEvent::deserialize(&buffer[..size]).unwrap();
        assert_eq!(spike, deserialized);
    }

    #[test]
    fn test_weight_matrix_serialization() {
        let mut matrix = WeightMatrix::new(3, 4);
        matrix.set(1, 2, 0.5).unwrap();
        matrix.learning_rate = 0.02;

        let mut buffer = vec![0u8; matrix.serialized_size()];
        let size = matrix.serialize(&mut buffer).unwrap();
        
        let deserialized = WeightMatrix::deserialize(&buffer[..size]).unwrap();
        assert_eq!(matrix.shape, deserialized.shape);
        assert_eq!(matrix.learning_rate, deserialized.learning_rate);
        assert_eq!(matrix.get(1, 2).unwrap(), deserialized.get(1, 2).unwrap());
    }

    #[test]
    fn test_layer_state_serialization() {
        let mut layer = LayerState::new(1, 100);
        layer.potentials[50] = 0.7;
        layer.add_spike(SpikeEvent {
            neuron_id: 50,
            timestamp: 2.0,
            amplitude: 1.0,
            layer: 1,
            reserved: 0,
        });

        let mut buffer = vec![0u8; layer.serialized_size()];
        let size = layer.serialize(&mut buffer).unwrap();
        
        let deserialized = LayerState::deserialize(&buffer[..size]).unwrap();
        assert_eq!(layer.layer_id, deserialized.layer_id);
        assert_eq!(layer.potentials[50], deserialized.potentials[50]);
        assert_eq!(layer.spike_history.len(), deserialized.spike_history.len());
    }

    #[test]
    fn test_spike_codec() {
        let mut codec = SpikeEventCodec::new(3);
        
        codec.add_spike(SpikeEvent {
            neuron_id: 1,
            timestamp: 1.0,
            amplitude: 0.5,
            layer: 0,
            reserved: 0,
        });
        
        assert!(!codec.is_batch_ready());
        
        codec.add_spike(SpikeEvent {
            neuron_id: 2,
            timestamp: 2.0,
            amplitude: 0.6,
            layer: 0,
            reserved: 0,
        });
        
        codec.add_spike(SpikeEvent {
            neuron_id: 3,
            timestamp: 3.0,
            amplitude: 0.7,
            layer: 0,
            reserved: 0,
        });
        
        assert!(codec.is_batch_ready());
        
        let mut buffer = vec![0u8; 1024];
        let size = codec.serialize_batch(&mut buffer).unwrap();
        
        let deserialized = codec.deserialize_batch(&buffer[..size]).unwrap();
        assert_eq!(deserialized.len(), 3);
        assert_eq!(deserialized[0].neuron_id, 1);
        assert_eq!(deserialized[2].neuron_id, 3);
    }
}