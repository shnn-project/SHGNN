//! # SHNN Zero-Copy Serialization Library
//! 
//! Zero-dependency, zero-copy serialization optimized for neuromorphic data structures.
//! Provides high-performance binary serialization with deterministic memory layout.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

/// Binary encoding and decoding
pub mod binary;

/// Zero-copy buffer management
pub mod buffer;

/// Serialization traits and implementations
pub mod traits;

/// Neural network specific serialization
pub mod neural;

/// Endianness handling
pub mod endian;

/// Protocol buffer implementation
pub mod proto;

// Re-exports for convenience
pub use binary::{BinaryEncoder, BinaryDecoder};
pub use buffer::{Buffer, BufferMut, ZeroCopyBuffer};
pub use traits::{Serialize, Deserialize, SerializeSize, ZeroCopySerialize};
pub use neural::{
    SpikeEvent, WeightMatrix, LayerState, NeuralSerializer,
    SpikeEventCodec, WeightMatrixCodec
};
pub use endian::{Endianness, EndianConverter};
pub use proto::{ProtoEncoder, ProtoDecoder, ProtoMessage, NeuralMessage};

/// Common error types for serialization operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializeError {
    /// Buffer too small for data
    BufferTooSmall,
    /// Invalid data format
    InvalidFormat,
    /// Unexpected end of data
    UnexpectedEof,
    /// Version mismatch
    VersionMismatch,
    /// Checksum validation failed
    ChecksumFailed,
    /// Alignment error
    AlignmentError,
    /// Invalid magic number
    InvalidMagic,
    /// Unsupported version
    UnsupportedVersion,
    /// Invalid UTF-8 sequence
    InvalidUtf8,
    /// Invalid position
    InvalidPosition,
    /// Header already written
    HeaderAlreadyWritten,
    /// Wire type mismatch in protobuf
    WireTypeMismatch,
    /// Invalid wire type
    InvalidWireType,
    /// Invalid field number
    InvalidFieldNumber,
    /// Varint too long
    VarintTooLong,
    /// Shape mismatch in matrices
    ShapeMismatch,
    /// Index out of bounds
    IndexOutOfBounds,
    /// Custom error with message
    Custom(&'static str),
}
impl core::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BufferTooSmall => write!(f, "Buffer too small for serialization"),
            Self::InvalidFormat => write!(f, "Invalid data format"),
            Self::UnexpectedEof => write!(f, "Unexpected end of data"),
            Self::VersionMismatch => write!(f, "Version mismatch"),
            Self::ChecksumFailed => write!(f, "Checksum validation failed"),
            Self::AlignmentError => write!(f, "Memory alignment error"),
            Self::InvalidMagic => write!(f, "Invalid magic number"),
            Self::UnsupportedVersion => write!(f, "Unsupported version"),
            Self::InvalidUtf8 => write!(f, "Invalid UTF-8 sequence"),
            Self::InvalidPosition => write!(f, "Invalid buffer position"),
            Self::HeaderAlreadyWritten => write!(f, "Header already written"),
            Self::WireTypeMismatch => write!(f, "Wire type mismatch in protobuf"),
            Self::InvalidWireType => write!(f, "Invalid wire type"),
            Self::InvalidFieldNumber => write!(f, "Invalid field number"),
            Self::VarintTooLong => write!(f, "Varint too long"),
            Self::ShapeMismatch => write!(f, "Shape mismatch in matrices"),
            Self::IndexOutOfBounds => write!(f, "Index out of bounds"),
            Self::Custom(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SerializeError {}


/// Result type for serialization operations
pub type Result<T> = core::result::Result<T, SerializeError>;

/// Magic number for SHNN binary format
pub const SHNN_MAGIC: u32 = 0x534E4E48; // "SHNN" in ASCII

/// Binary format constants
pub const MAGIC_HEADER: [u8; 4] = [b'S', b'H', b'N', b'N'];
pub const VERSION_1: u8 = 1;

/// Current serialization format version
pub const FORMAT_VERSION: u16 = 1;

/// Constants for neuromorphic serialization
pub mod constants {
    /// Maximum spike event batch size for serialization
    pub const MAX_SPIKE_BATCH_SIZE: usize = 1024;
    
    /// Default buffer alignment for neural data
    pub const NEURAL_ALIGNMENT: usize = 16;
    
    /// Maximum weight matrix dimension for serialization
    pub const MAX_MATRIX_DIM: usize = 65536;
    
    /// Default compression threshold in bytes
    pub const COMPRESSION_THRESHOLD: usize = 1024;
    
    /// Magic number for spike event data
    pub const SPIKE_MAGIC: u32 = 0x53504B45; // "SPKE"
    
    /// Magic number for weight matrix data
    pub const WEIGHT_MAGIC: u32 = 0x57474854; // "WGHT"
}

/// Utility functions for serialization
pub mod utils {
    use super::*;
    
    /// Calculate aligned size for given size and alignment
    pub const fn align_size(size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
    }
    
    /// Check if pointer is aligned
    pub fn is_aligned(ptr: *const u8, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }
    
    /// Calculate checksum for data
    pub fn calculate_checksum(data: &[u8]) -> u32 {
        let mut checksum = 0u32;
        for &byte in data {
            checksum = checksum.wrapping_add(byte as u32);
        }
        checksum
    }
    
    /// Validate magic number
    pub fn validate_magic(magic: u32, expected: u32) -> Result<()> {
        if magic == expected {
            Ok(())
        } else {
            Err(SerializeError::InvalidMagic)
        }
    }
    
    /// Validate format version
    pub fn validate_version(version: u16, expected: u16) -> Result<()> {
        if version == expected {
            Ok(())
        } else {
            Err(SerializeError::VersionMismatch)
        }
    }
}

/// Header for serialized data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SerializationHeader {
    /// Magic number
    pub magic: u32,
    /// Format version
    pub version: u16,
    /// Data type identifier
    pub data_type: u16,
    /// Total data size
    pub data_size: u32,
    /// Checksum of data
    pub checksum: u32,
}

impl SerializationHeader {
    /// Create new header
    pub const fn new(data_type: u16, data_size: u32, checksum: u32) -> Self {
        Self {
            magic: SHNN_MAGIC,
            version: FORMAT_VERSION,
            data_type,
            data_size,
            checksum,
        }
    }
    
    /// Validate header
    pub fn validate(&self) -> Result<()> {
        utils::validate_magic(self.magic, SHNN_MAGIC)?;
        utils::validate_version(self.version, FORMAT_VERSION)?;
        Ok(())
    }
    
    /// Get header size
    pub const fn size() -> usize {
        core::mem::size_of::<Self>()
    }
}

/// Data type identifiers
pub mod data_types {
    /// Spike event data
    pub const SPIKE_EVENT: u16 = 1;
    /// Weight matrix data
    pub const WEIGHT_MATRIX: u16 = 2;
    /// Neural state data
    pub const NEURAL_STATE: u16 = 3;
    /// Network topology data
    pub const NETWORK_TOPOLOGY: u16 = 4;
    /// Training data
    pub const TRAINING_DATA: u16 = 5;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_size() {
        assert_eq!(utils::align_size(15, 16), 16);
        assert_eq!(utils::align_size(16, 16), 16);
        assert_eq!(utils::align_size(17, 16), 32);
    }

    #[test]
    fn test_checksum() {
        let data = b"test data";
        let checksum = utils::calculate_checksum(data);
        assert!(checksum > 0);
    }

    #[test]
    fn test_header() {
        let header = SerializationHeader::new(data_types::SPIKE_EVENT, 1024, 0x12345678);
        assert_eq!(header.magic, SHNN_MAGIC);
        assert_eq!(header.version, FORMAT_VERSION);
        assert_eq!(header.data_type, data_types::SPIKE_EVENT);
        assert_eq!(header.data_size, 1024);
        
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_constants() {
        assert!(constants::MAX_SPIKE_BATCH_SIZE > 0);
        assert!(constants::NEURAL_ALIGNMENT.is_power_of_two());
        assert!(constants::MAX_MATRIX_DIM > 0);
    }
}