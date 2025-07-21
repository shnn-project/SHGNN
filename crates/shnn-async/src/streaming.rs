//! Real-time spike streaming and processing
//!
//! This module provides streaming interfaces for continuous spike processing.

use crate::error::{AsyncError, AsyncResult};
use shnn_core::{spike::{Spike, SpikeTrain}, time::{Time, Duration}};

use std::collections::VecDeque;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::task::{Context, Poll};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Stream configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StreamConfig {
    /// Buffer size for the stream
    pub buffer_size: usize,
    /// Maximum processing latency
    pub max_latency: Duration,
    /// Enable backpressure handling
    pub enable_backpressure: bool,
    /// Stream processing mode
    pub mode: StreamMode,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            max_latency: Duration::from_millis(10),
            enable_backpressure: true,
            mode: StreamMode::Realtime,
        }
    }
}

/// Stream processing modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StreamMode {
    /// Real-time processing
    Realtime,
    /// Batch processing
    Batch,
    /// Buffered processing
    Buffered,
}

/// Spike stream for continuous processing
pub struct SpikeStream {
    /// Internal buffer
    buffer: VecDeque<Spike>,
    /// Configuration
    config: StreamConfig,
    /// Stream state
    state: StreamState,
}

/// Stream state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    Active,
    Paused,
    Closed,
}

impl SpikeStream {
    /// Create a new spike stream
    pub fn new(config: StreamConfig) -> Self {
        Self {
            buffer: VecDeque::with_capacity(config.buffer_size),
            config,
            state: StreamState::Active,
        }
    }
    
    /// Push a spike to the stream
    pub async fn push(&mut self, spike: Spike) -> AsyncResult<()> {
        if self.state != StreamState::Active {
            return Err(AsyncError::streaming("Stream is not active"));
        }
        
        if self.buffer.len() >= self.config.buffer_size {
            if self.config.enable_backpressure {
                return Err(AsyncError::streaming("Stream buffer full"));
            } else {
                // Drop oldest spike
                self.buffer.pop_front();
            }
        }
        
        self.buffer.push_back(spike);
        Ok(())
    }
    
    /// Get next spike from stream
    pub async fn next(&mut self) -> Option<Spike> {
        self.buffer.pop_front()
    }
    
    /// Close the stream
    pub fn close(&mut self) {
        self.state = StreamState::Closed;
    }
    
    /// Check if stream is active
    pub fn is_active(&self) -> bool {
        self.state == StreamState::Active
    }
}

impl Default for SpikeStream {
    fn default() -> Self {
        Self::new(StreamConfig::default())
    }
}

/// Stream processor for transforming spike streams
pub struct StreamProcessor<F> {
    /// Processing function
    processor: F,
    /// Configuration
    config: StreamConfig,
}

impl<F> StreamProcessor<F>
where
    F: Fn(Spike) -> AsyncResult<Vec<Spike>>,
{
    /// Create a new stream processor
    pub fn new(processor: F, config: StreamConfig) -> Self {
        Self { processor, config }
    }
    
    /// Process a spike stream
    pub async fn process_stream(
        &self,
        mut input: SpikeStream,
    ) -> AsyncResult<SpikeStream> {
        let mut output = SpikeStream::new(self.config.clone());
        
        while input.is_active() {
            if let Some(spike) = input.next().await {
                let processed = (self.processor)(spike)?;
                for out_spike in processed {
                    output.push(out_spike).await?;
                }
            }
        }
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shnn_core::spike::NeuronId;
    
    #[tokio::test]
    async fn test_spike_stream() {
        let mut stream = SpikeStream::new(StreamConfig::default());
        
        let spike = Spike::binary(NeuronId::new(1), Time::from_millis(1)).unwrap();
        assert!(stream.push(spike).await.is_ok());
        
        let received = stream.next().await;
        assert!(received.is_some());
    }
}