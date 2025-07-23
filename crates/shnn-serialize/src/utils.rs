//! Utility functions for serialization
//! 
//! Provides helper functions for memory alignment, endianness handling,
//! and other common serialization operations.

use crate::{Result, SerializeError};

/// Check if a pointer is properly aligned for a given type
pub fn is_aligned<T>(ptr: *const u8, align: usize) -> bool {
    (ptr as usize) % align == 0
}

/// Align a size to the next boundary
pub fn align_size(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

/// Calculate padding needed to align to boundary
pub fn padding_for_align(size: usize, align: usize) -> usize {
    align_size(size, align) - size
}

/// Check if the system is little-endian
pub fn is_little_endian() -> bool {
    cfg!(target_endian = "little")
}

/// Check if the system is big-endian  
pub fn is_big_endian() -> bool {
    cfg!(target_endian = "big")
}

/// Convert bytes to little-endian u16
pub fn bytes_to_u16_le(bytes: &[u8]) -> Result<u16> {
    if bytes.len() < 2 {
        return Err(SerializeError::UnexpectedEof);
    }
    let mut array = [0u8; 2];
    array.copy_from_slice(&bytes[..2]);
    Ok(u16::from_le_bytes(array))
}

/// Convert bytes to little-endian u32
pub fn bytes_to_u32_le(bytes: &[u8]) -> Result<u32> {
    if bytes.len() < 4 {
        return Err(SerializeError::UnexpectedEof);
    }
    let mut array = [0u8; 4];
    array.copy_from_slice(&bytes[..4]);
    Ok(u32::from_le_bytes(array))
}

/// Convert bytes to little-endian u64
pub fn bytes_to_u64_le(bytes: &[u8]) -> Result<u64> {
    if bytes.len() < 8 {
        return Err(SerializeError::UnexpectedEof);
    }
    let mut array = [0u8; 8];
    array.copy_from_slice(&bytes[..8]);
    Ok(u64::from_le_bytes(array))
}

/// Convert bytes to little-endian f32
pub fn bytes_to_f32_le(bytes: &[u8]) -> Result<f32> {
    let bits = bytes_to_u32_le(bytes)?;
    Ok(f32::from_bits(bits))
}

/// Convert bytes to little-endian f64
pub fn bytes_to_f64_le(bytes: &[u8]) -> Result<f64> {
    let bits = bytes_to_u64_le(bytes)?;
    Ok(f64::from_bits(bits))
}

/// Convert u16 to little-endian bytes
pub fn u16_to_bytes_le(value: u16) -> [u8; 2] {
    value.to_le_bytes()
}

/// Convert u32 to little-endian bytes
pub fn u32_to_bytes_le(value: u32) -> [u8; 4] {
    value.to_le_bytes()
}

/// Convert u64 to little-endian bytes
pub fn u64_to_bytes_le(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

/// Convert f32 to little-endian bytes
pub fn f32_to_bytes_le(value: f32) -> [u8; 4] {
    value.to_bits().to_le_bytes()
}

/// Convert f64 to little-endian bytes
pub fn f64_to_bytes_le(value: f64) -> [u8; 8] {
    value.to_bits().to_le_bytes()
}

/// Validate UTF-8 string
pub fn validate_utf8(bytes: &[u8]) -> Result<()> {
    core::str::from_utf8(bytes)
        .map_err(|_| SerializeError::InvalidUtf8)?;
    Ok(())
}

/// Calculate CRC32 checksum (simple implementation)
pub fn calculate_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    
    !crc
}

/// Verify CRC32 checksum
pub fn verify_crc32(data: &[u8], expected: u32) -> bool {
    calculate_crc32(data) == expected
}

/// Round up to next power of 2
pub fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}

/// Check if a number is a power of 2
pub fn is_power_of_2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Safe byte slice operations
pub struct SafeSlice<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> SafeSlice<'a> {
    /// Create a new safe slice
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// Check if we can read n bytes
    pub fn can_read(&self, n: usize) -> bool {
        self.position + n <= self.data.len()
    }

    /// Read n bytes safely
    pub fn read(&mut self, n: usize) -> Result<&'a [u8]> {
        if !self.can_read(n) {
            return Err(SerializeError::UnexpectedEof);
        }
        
        let start = self.position;
        self.position += n;
        Ok(&self.data[start..self.position])
    }

    /// Peek at n bytes without consuming
    pub fn peek(&self, n: usize) -> Result<&'a [u8]> {
        if !self.can_read(n) {
            return Err(SerializeError::UnexpectedEof);
        }
        
        Ok(&self.data[self.position..self.position + n])
    }

    /// Get remaining bytes
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.position
    }

    /// Reset position
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

/// Memory pool for reducing allocations during serialization
#[derive(Debug)]
pub struct MemoryPool {
    buffers: Vec<Vec<u8>>,
    current_size: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            current_size: 1024,
        }
    }

    /// Get a buffer of at least the specified size
    pub fn get_buffer(&mut self, min_size: usize) -> Vec<u8> {
        // Try to reuse existing buffer
        for i in 0..self.buffers.len() {
            if self.buffers[i].capacity() >= min_size {
                let mut buffer = self.buffers.swap_remove(i);
                buffer.clear();
                return buffer;
            }
        }

        // Create new buffer
        let size = core::cmp::max(min_size, self.current_size);
        self.current_size = next_power_of_2(size * 2);
        Vec::with_capacity(size)
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, buffer: Vec<u8>) {
        if buffer.capacity() > 0 {
            self.buffers.push(buffer);
        }
    }

    /// Clear all cached buffers
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.current_size = 1024;
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        let ptr = 8 as *const u8;
        assert!(is_aligned(ptr, 4));
        assert!(is_aligned(ptr, 8));
        
        let ptr = 9 as *const u8;
        assert!(!is_aligned(ptr, 4));
        assert!(!is_aligned(ptr, 8));
    }

    #[test]
    fn test_align_size() {
        assert_eq!(align_size(0, 4), 0);
        assert_eq!(align_size(1, 4), 4);
        assert_eq!(align_size(4, 4), 4);
        assert_eq!(align_size(5, 4), 8);
    }

    #[test]
    fn test_endian_conversion() {
        let value = 0x12345678u32;
        let bytes = u32_to_bytes_le(value);
        let converted = bytes_to_u32_le(&bytes).unwrap();
        assert_eq!(value, converted);
    }

    #[test]
    fn test_crc32() {
        let data = b"hello world";
        let crc = calculate_crc32(data);
        assert!(verify_crc32(data, crc));
        assert!(!verify_crc32(data, crc + 1));
    }

    #[test]
    fn test_power_of_2() {
        assert!(is_power_of_2(1));
        assert!(is_power_of_2(2));
        assert!(is_power_of_2(4));
        assert!(is_power_of_2(8));
        assert!(!is_power_of_2(3));
        assert!(!is_power_of_2(5));
        
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(8), 8);
        assert_eq!(next_power_of_2(9), 16);
    }

    #[test]
    fn test_safe_slice() {
        let data = [1, 2, 3, 4, 5];
        let mut slice = SafeSlice::new(&data);
        
        let bytes = slice.read(2).unwrap();
        assert_eq!(bytes, &[1, 2]);
        
        let bytes = slice.read(2).unwrap();
        assert_eq!(bytes, &[3, 4]);
        
        assert_eq!(slice.remaining(), 1);
        
        // Should fail to read more than remaining
        assert!(slice.read(2).is_err());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new();
        
        let buffer1 = pool.get_buffer(100);
        assert!(buffer1.capacity() >= 100);
        
        let buffer2 = pool.get_buffer(200);
        assert!(buffer2.capacity() >= 200);
        
        pool.return_buffer(buffer1);
        pool.return_buffer(buffer2);
        
        // Should reuse existing buffer
        let buffer3 = pool.get_buffer(150);
        assert!(buffer3.capacity() >= 150);
    }
}