//! Zero-copy buffer management for serialization
//! 
//! Provides efficient buffer handling with zero-copy operations and memory-safe
//! access patterns optimized for neuromorphic data structures.

use crate::{Result, SerializeError};
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Read-only buffer for zero-copy deserialization
#[derive(Debug, Clone)]
pub struct Buffer<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Buffer<'a> {
    /// Create a new buffer from a byte slice
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// Get the current position in the buffer
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get the total length of the buffer
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get remaining bytes in the buffer
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    /// Check if we have at least n bytes remaining
    pub fn has_remaining(&self, n: usize) -> bool {
        self.remaining() >= n
    }

    /// Read bytes from the buffer, advancing position
    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if !self.has_remaining(n) {
            return Err(SerializeError::UnexpectedEof);
        }
        
        let bytes = &self.data[self.position..self.position + n];
        self.position += n;
        Ok(bytes)
    }

    /// Peek at bytes without advancing position
    pub fn peek_bytes(&self, n: usize) -> Result<&'a [u8]> {
        if !self.has_remaining(n) {
            return Err(SerializeError::UnexpectedEof);
        }
        
        Ok(&self.data[self.position..self.position + n])
    }

    /// Read a single byte
    pub fn read_u8(&mut self) -> Result<u8> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0])
    }

    /// Read a 32-bit unsigned integer (little-endian)
    pub fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        let mut array = [0u8; 4];
        array.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(array))
    }

    /// Read a 32-bit float (little-endian)
    pub fn read_f32(&mut self) -> Result<f32> {
        let bits = self.read_u32()?;
        Ok(f32::from_bits(bits))
    }

    /// Read a slice of the specified type with zero-copy
    pub fn read_slice<T>(&mut self, count: usize) -> Result<&'a [T]> 
    where
        T: Copy,
    {
        let bytes_needed = count * core::mem::size_of::<T>();
        let bytes = self.read_bytes(bytes_needed)?;
        
        // Check alignment
        let ptr = bytes.as_ptr();
        if !crate::utils::is_aligned(ptr, core::mem::align_of::<T>()) {
            return Err(SerializeError::AlignmentError);
        }
        
        // Safety: We've checked alignment and length
        let slice = unsafe {
            core::slice::from_raw_parts(ptr as *const T, count)
        };
        
        Ok(slice)
    }

    /// Reset position to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Seek to a specific position
    pub fn seek(&mut self, position: usize) -> Result<()> {
        if position > self.data.len() {
            return Err(SerializeError::InvalidPosition);
        }
        self.position = position;
        Ok(())
    }

    /// Get the underlying data slice
    pub fn as_slice(&self) -> &'a [u8] {
        self.data
    }

    /// Get remaining data from current position
    pub fn remaining_slice(&self) -> &'a [u8] {
        &self.data[self.position..]
    }
}

/// Mutable buffer for serialization
#[derive(Debug)]
pub struct BufferMut<'a> {
    data: &'a mut [u8],
    position: usize,
}

impl<'a> BufferMut<'a> {
    /// Create a new mutable buffer
    pub fn new(data: &'a mut [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// Get the current position
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get the total capacity
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Get remaining space
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    /// Check if we have space for n bytes
    pub fn has_remaining(&self, n: usize) -> bool {
        self.remaining() >= n
    }

    /// Write bytes to the buffer
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if !self.has_remaining(bytes.len()) {
            return Err(SerializeError::BufferTooSmall);
        }
        
        let end = self.position + bytes.len();
        self.data[self.position..end].copy_from_slice(bytes);
        self.position = end;
        Ok(())
    }

    /// Write a single byte
    pub fn write_u8(&mut self, value: u8) -> Result<()> {
        self.write_bytes(&[value])
    }

    /// Write a 32-bit unsigned integer (little-endian)
    pub fn write_u32(&mut self, value: u32) -> Result<()> {
        let bytes = value.to_le_bytes();
        self.write_bytes(&bytes)
    }

    /// Write a 32-bit float (little-endian)
    pub fn write_f32(&mut self, value: f32) -> Result<()> {
        let bits = value.to_bits();
        self.write_u32(bits)
    }

    /// Write a slice with zero-copy
    pub fn write_slice<T>(&mut self, slice: &[T]) -> Result<()> 
    where
        T: Copy,
    {
        let bytes_needed = slice.len() * core::mem::size_of::<T>();
        if !self.has_remaining(bytes_needed) {
            return Err(SerializeError::BufferTooSmall);
        }
        
        // Safety: T is Copy, so we can safely cast to bytes
        let bytes = unsafe {
            core::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                bytes_needed,
            )
        };
        
        self.write_bytes(bytes)
    }

    /// Reset position to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Seek to a specific position
    pub fn seek(&mut self, position: usize) -> Result<()> {
        if position > self.data.len() {
            return Err(SerializeError::InvalidPosition);
        }
        self.position = position;
        Ok(())
    }

    /// Get the written data as a slice
    pub fn written(&self) -> &[u8] {
        &self.data[..self.position]
    }

    /// Get remaining space as a mutable slice
    pub fn remaining_mut(&mut self) -> &mut [u8] {
        &mut self.data[self.position..]
    }
}

/// Zero-copy buffer for efficient memory operations
#[derive(Debug)]
pub struct ZeroCopyBuffer {
    data: Vec<u8>,
    position: usize,
}

impl ZeroCopyBuffer {
    /// Create a new zero-copy buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            position: 0,
        }
    }

    /// Create from existing data
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self { data, position: 0 }
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Get current length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reserve additional capacity
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Extend buffer with zeros
    pub fn extend_zeros(&mut self, count: usize) {
        self.data.resize(self.data.len() + count, 0);
    }

    /// Append data
    pub fn append(&mut self, data: &[u8]) {
        self.data.extend_from_slice(data);
    }

    /// Get as immutable buffer
    pub fn as_buffer(&self) -> Buffer<'_> {
        Buffer::new(&self.data)
    }

    /// Get as mutable buffer
    pub fn as_buffer_mut(&mut self) -> BufferMut<'_> {
        BufferMut::new(&mut self.data)
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.data.clear();
        self.position = 0;
    }

    /// Shrink to fit actual data
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Get the underlying vector
    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }

    /// Get a reference to the underlying data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_read() {
        let data = [1, 2, 3, 4, 5];
        let mut buffer = Buffer::new(&data);
        
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.remaining(), 5);
        
        let byte = buffer.read_u8().unwrap();
        assert_eq!(byte, 1);
        assert_eq!(buffer.remaining(), 4);
        
        let bytes = buffer.read_bytes(2).unwrap();
        assert_eq!(bytes, &[2, 3]);
        assert_eq!(buffer.remaining(), 2);
    }

    #[test]
    fn test_buffer_mut_write() {
        let mut data = [0u8; 10];
        let mut buffer = BufferMut::new(&mut data);
        
        buffer.write_u8(42).unwrap();
        buffer.write_u32(0x12345678).unwrap();
        
        assert_eq!(buffer.position(), 5);
        assert_eq!(data[0], 42);
        assert_eq!(&data[1..5], &[0x78, 0x56, 0x34, 0x12]); // little-endian
    }

    #[test]
    fn test_zero_copy_buffer() {
        let mut buffer = ZeroCopyBuffer::new(100);
        
        buffer.append(&[1, 2, 3, 4]);
        assert_eq!(buffer.len(), 4);
        
        let read_buffer = buffer.as_buffer();
        assert_eq!(read_buffer.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_buffer_bounds_checking() {
        let data = [1, 2];
        let mut buffer = Buffer::new(&data);
        
        // This should succeed
        buffer.read_u8().unwrap();
        buffer.read_u8().unwrap();
        
        // This should fail
        assert!(buffer.read_u8().is_err());
    }
}