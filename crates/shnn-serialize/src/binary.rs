//! Binary serialization encoder and decoder
//! 
//! Provides efficient binary encoding/decoding with deterministic memory layout
//! and zero-copy optimizations for neuromorphic data structures.

use crate::{
    Buffer, BufferMut, Result, SerializeError, Serialize, Deserialize, MAGIC_HEADER, VERSION_1,
};
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Binary format encoder for serializing data structures
#[derive(Debug)]
pub struct BinaryEncoder<'a> {
    buffer: BufferMut<'a>,
    written_header: bool,
}

impl<'a> BinaryEncoder<'a> {
    /// Create a new binary encoder with the provided buffer
    pub fn new(buffer: BufferMut<'a>) -> Self {
        Self {
            buffer,
            written_header: false,
        }
    }

    /// Write the binary format header
    pub fn write_header(&mut self) -> Result<()> {
        if self.written_header {
            return Err(SerializeError::HeaderAlreadyWritten);
        }

        // Write magic header (4 bytes)
        self.buffer.write_bytes(&MAGIC_HEADER)?;
        
        // Write version (1 byte)
        self.buffer.write_u8(VERSION_1)?;
        
        // Write reserved bytes for future use (3 bytes)
        self.buffer.write_bytes(&[0, 0, 0])?;
        
        self.written_header = true;
        Ok(())
    }

    /// Encode a value using its Serialize implementation
    pub fn encode<T: Serialize>(&mut self, value: &T) -> Result<()> {
        if !self.written_header {
            self.write_header()?;
        }

        // Write the serialized size first
        let size = value.serialized_size();
        self.buffer.write_u32(size as u32)?;
        
        // Write the actual data
        let remaining = self.buffer.remaining_mut();
        let written = value.serialize(remaining)?;
        
        // Advance the buffer position
        let new_pos = self.buffer.position() + written;
        self.buffer.seek(new_pos)?;
        
        Ok(())
    }

    /// Encode raw bytes with length prefix
    pub fn encode_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if !self.written_header {
            self.write_header()?;
        }

        // Write length prefix
        self.buffer.write_u32(bytes.len() as u32)?;
        
        // Write data
        self.buffer.write_bytes(bytes)?;
        
        Ok(())
    }

    /// Encode a string with UTF-8 encoding
    pub fn encode_string(&mut self, s: &str) -> Result<()> {
        self.encode_bytes(s.as_bytes())
    }

    /// Encode an array of values
    pub fn encode_array<T: Serialize>(&mut self, array: &[T]) -> Result<()> {
        if !self.written_header {
            self.write_header()?;
        }

        // Write array length
        self.buffer.write_u32(array.len() as u32)?;
        
        // Write each element
        for item in array {
            self.encode(item)?;
        }
        
        Ok(())
    }

    /// Encode a slice with zero-copy for POD types
    pub fn encode_slice<T: Copy>(&mut self, slice: &[T]) -> Result<()> {
        if !self.written_header {
            self.write_header()?;
        }

        // Write slice length
        self.buffer.write_u32(slice.len() as u32)?;
        
        // Write slice data with zero-copy
        self.buffer.write_slice(slice)?;
        
        Ok(())
    }

    /// Get the current position in the buffer
    pub fn position(&self) -> usize {
        self.buffer.position()
    }

    /// Get the written data
    pub fn written(&self) -> &[u8] {
        self.buffer.written()
    }

    /// Check if header has been written
    pub fn has_header(&self) -> bool {
        self.written_header
    }
}

/// Binary format decoder for deserializing data structures
#[derive(Debug)]
pub struct BinaryDecoder<'a> {
    buffer: Buffer<'a>,
    validated_header: bool,
}

impl<'a> BinaryDecoder<'a> {
    /// Create a new binary decoder with the provided buffer
    pub fn new(buffer: Buffer<'a>) -> Self {
        Self {
            buffer,
            validated_header: false,
        }
    }

    /// Validate the binary format header
    pub fn validate_header(&mut self) -> Result<()> {
        if self.validated_header {
            return Ok(());
        }

        // Check magic header
        let magic = self.buffer.read_bytes(4)?;
        if magic != MAGIC_HEADER {
            return Err(SerializeError::InvalidMagic);
        }

        // Check version
        let version = self.buffer.read_u8()?;
        if version != VERSION_1 {
            return Err(SerializeError::UnsupportedVersion);
        }

        // Skip reserved bytes
        self.buffer.read_bytes(3)?;

        self.validated_header = true;
        Ok(())
    }

    /// Decode a value using its Deserialize implementation
    pub fn decode<T: Deserialize<'a>>(&mut self) -> Result<T> {
        if !self.validated_header {
            self.validate_header()?;
        }

        // Read the serialized size
        let _size = self.buffer.read_u32()?;
        
        // Deserialize the data
        T::deserialize(self.buffer.remaining_slice())
    }

    /// Decode raw bytes with length prefix
    pub fn decode_bytes(&mut self) -> Result<&'a [u8]> {
        if !self.validated_header {
            self.validate_header()?;
        }

        // Read length prefix
        let length = self.buffer.read_u32()? as usize;
        
        // Read data
        self.buffer.read_bytes(length)
    }

    /// Decode a string with UTF-8 validation
    pub fn decode_string(&mut self) -> Result<&'a str> {
        let bytes = self.decode_bytes()?;
        core::str::from_utf8(bytes)
            .map_err(|_| SerializeError::InvalidUtf8)
    }

    /// Decode an array of values
    pub fn decode_array<T: Deserialize<'a>>(&mut self) -> Result<Vec<T>> {
        if !self.validated_header {
            self.validate_header()?;
        }

        // Read array length
        let length = self.buffer.read_u32()? as usize;
        
        // Read each element
        let mut result = Vec::with_capacity(length);
        for _ in 0..length {
            result.push(self.decode()?);
        }
        
        Ok(result)
    }

    /// Decode a slice with zero-copy for POD types
    pub fn decode_slice<T: Copy>(&mut self) -> Result<&'a [T]> {
        if !self.validated_header {
            self.validate_header()?;
        }

        // Read slice length
        let length = self.buffer.read_u32()? as usize;
        
        // Read slice data with zero-copy
        self.buffer.read_slice(length)
    }

    /// Get the current position in the buffer
    pub fn position(&self) -> usize {
        self.buffer.position()
    }

    /// Get remaining bytes
    pub fn remaining(&self) -> usize {
        self.buffer.remaining()
    }

    /// Check if header has been validated
    pub fn has_validated_header(&self) -> bool {
        self.validated_header
    }

    /// Skip bytes in the buffer
    pub fn skip(&mut self, count: usize) -> Result<()> {
        self.buffer.read_bytes(count)?;
        Ok(())
    }

    /// Peek at the next bytes without consuming them
    pub fn peek(&self, count: usize) -> Result<&'a [u8]> {
        self.buffer.peek_bytes(count)
    }
}

/// Utility functions for binary serialization
pub mod utils {
    use super::*;

    /// Serialize a value to a vector
    pub fn serialize_to_vec<T: Serialize>(value: &T) -> Result<Vec<u8>> {
        let size = value.serialized_size() + 8; // Header + size prefix
        let mut buffer = vec![0u8; size];
        
        let position = {
            let mut encoder = BinaryEncoder::new(BufferMut::new(&mut buffer));
            encoder.encode(value)?;
            encoder.position()
        };
        
        Ok(buffer[..position].to_vec())
    }

    /// Deserialize a value from bytes
    pub fn deserialize_from_bytes<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T> {
        let mut decoder = BinaryDecoder::new(Buffer::new(bytes));
        decoder.decode()
    }

    /// Calculate the total serialized size including header
    pub fn calculate_serialized_size<T: Serialize>(value: &T) -> usize {
        8 + 4 + value.serialized_size() // Header + size prefix + data
    }

    /// Validate binary format without full deserialization
    pub fn validate_format(bytes: &[u8]) -> Result<()> {
        let mut decoder = BinaryDecoder::new(Buffer::new(bytes));
        decoder.validate_header()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_encoder_decoder() {
        let mut buffer = vec![0u8; 1024];
        let mut encoder = BinaryEncoder::new(BufferMut::new(&mut buffer));
        
        // Encode some data
        encoder.encode(&42u32).unwrap();
        encoder.encode(&3.14f32).unwrap();
        encoder.encode_string("hello").unwrap();
        
        let written = encoder.written().to_vec();
        
        // Decode the data
        let mut decoder = BinaryDecoder::new(Buffer::new(&written));
        
        let value1: u32 = decoder.decode().unwrap();
        let value2: f32 = decoder.decode().unwrap();
        let string = decoder.decode_string().unwrap();
        
        assert_eq!(value1, 42);
        assert!((value2 - 3.14).abs() < 1e-6);
        assert_eq!(string, "hello");
    }

    #[test]
    fn test_array_encoding() {
        let data = [1u32, 2, 3, 4, 5];
        let mut buffer = vec![0u8; 1024];
        let mut encoder = BinaryEncoder::new(BufferMut::new(&mut buffer));
        
        encoder.encode_slice(&data).unwrap();
        let written = encoder.written().to_vec();
        
        let mut decoder = BinaryDecoder::new(Buffer::new(&written));
        let decoded: &[u32] = decoder.decode_slice().unwrap();
        
        assert_eq!(decoded, &data);
    }

    #[test]
    fn test_header_validation() {
        let mut buffer = vec![0u8; 100];
        let mut encoder = BinaryEncoder::new(BufferMut::new(&mut buffer));
        encoder.write_header().unwrap();
        
        let written = encoder.written().to_vec();
        
        let mut decoder = BinaryDecoder::new(Buffer::new(&written));
        decoder.validate_header().unwrap();
        
        assert!(decoder.has_validated_header());
    }

    #[test]
    fn test_invalid_magic() {
        let invalid_data = [0xFF, 0xFF, 0xFF, 0xFF, 1, 0, 0, 0];
        let mut decoder = BinaryDecoder::new(Buffer::new(&invalid_data));
        
        assert!(decoder.validate_header().is_err());
    }

    #[test]
    fn test_utils() {
        let value = 12345u32;
        let serialized = utils::serialize_to_vec(&value).unwrap();
        let deserialized: u32 = utils::deserialize_from_bytes(&serialized).unwrap();
        
        assert_eq!(value, deserialized);
    }
}