//! Async communication channels for spike processing
//!
//! This module provides high-performance async channels optimized for
//! neuromorphic spike communication patterns.

use crate::error::{AsyncError, AsyncResult};
use shnn_core::{spike::Spike, time::Time};

use std::collections::VecDeque;
use futures::{Stream, Sink};
use std::pin::Pin;
use std::task::{Context, Poll};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Channel configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChannelConfig {
    /// Channel capacity
    pub capacity: usize,
    /// Enable buffering
    pub buffered: bool,
    /// Channel type
    pub channel_type: ChannelType,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            buffered: true,
            channel_type: ChannelType::MPMC,
        }
    }
}

/// Channel type variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ChannelType {
    /// Single producer, single consumer
    SPSC,
    /// Multiple producer, single consumer
    MPSC,
    /// Single producer, multiple consumer
    SPMC,
    /// Multiple producer, multiple consumer
    MPMC,
}

/// Message types for channels
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MessageType {
    /// Spike message
    Spike(Spike),
    /// Control message
    Control(ControlMessage),
    /// Status update
    Status(StatusMessage),
}

/// Control messages
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ControlMessage {
    /// Start processing
    Start,
    /// Stop processing
    Stop,
    /// Pause processing
    Pause,
    /// Resume processing
    Resume,
    /// Shutdown channel
    Shutdown,
}

/// Status messages
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StatusMessage {
    /// Timestamp
    pub timestamp: Time,
    /// Message content
    pub content: String,
    /// Status level
    pub level: StatusLevel,
}

/// Status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StatusLevel {
    /// Debug information
    Debug,
    /// General information
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
}

/// Async spike channel
pub struct SpikeChannel {
    /// Internal message queue
    queue: VecDeque<MessageType>,
    /// Channel configuration
    config: ChannelConfig,
    /// Channel state
    state: ChannelState,
}

/// Channel state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelState {
    Open,
    Closed,
    Error,
}

impl SpikeChannel {
    /// Create a new spike channel
    pub fn new(config: ChannelConfig) -> Self {
        Self {
            queue: VecDeque::with_capacity(config.capacity),
            config,
            state: ChannelState::Open,
        }
    }
    
    /// Send a message
    pub async fn send(&mut self, message: MessageType) -> AsyncResult<()> {
        if self.state != ChannelState::Open {
            return Err(AsyncError::channel("SpikeChannel", "Channel is closed"));
        }
        
        if self.queue.len() >= self.config.capacity {
            return Err(AsyncError::channel("SpikeChannel", "Channel is full"));
        }
        
        self.queue.push_back(message);
        Ok(())
    }
    
    /// Receive a message
    pub async fn recv(&mut self) -> AsyncResult<Option<MessageType>> {
        if self.state == ChannelState::Error {
            return Err(AsyncError::channel("SpikeChannel", "Channel is in error state"));
        }
        
        Ok(self.queue.pop_front())
    }
    
    /// Close the channel
    pub fn close(&mut self) {
        self.state = ChannelState::Closed;
    }
    
    /// Check if channel is open
    pub fn is_open(&self) -> bool {
        self.state == ChannelState::Open
    }
    
    /// Get channel utilization
    pub fn utilization(&self) -> f32 {
        self.queue.len() as f32 / self.config.capacity as f32
    }
}

impl Default for SpikeChannel {
    fn default() -> Self {
        Self::new(ChannelConfig::default())
    }
}

/// Channel statistics
#[derive(Debug, Clone, Default)]
pub struct ChannelStats {
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
    /// Messages dropped
    pub messages_dropped: u64,
    /// Peak utilization
    pub peak_utilization: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use shnn_core::spike::{NeuronId, Spike};
    
    #[tokio::test]
    async fn test_spike_channel() {
        let mut channel = SpikeChannel::new(ChannelConfig::default());
        
        let spike = Spike::binary(NeuronId::new(1), Time::from_millis(1)).unwrap();
        let message = MessageType::Spike(spike);
        
        assert!(channel.send(message).await.is_ok());
        
        let received = channel.recv().await.unwrap();
        assert!(received.is_some());
    }
}