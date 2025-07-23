//! Endianness handling utilities
//! 
//! Provides cross-platform endianness conversion and handling for consistent
//! binary serialization across different architectures.

use crate::{Result, SerializeError};

/// Endianness types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    /// Little-endian byte order
    Little,
    /// Big-endian byte order
    Big,
}

/// Get the native endianness of the current platform
pub fn native_endianness() -> Endianness {
    if cfg!(target_endian = "little") {
        Endianness::Little
    } else {
        Endianness::Big
    }
}

/// Check if the current platform is little-endian
pub fn is_little_endian() -> bool {
    native_endianness() == Endianness::Little
}

/// Check if the current platform is big-endian
pub fn is_big_endian() -> bool {
    native_endianness() == Endianness::Big
}

/// Endian-aware number conversion utilities
pub struct EndianConverter {
    target_endian: Endianness,
}

impl EndianConverter {
    /// Create converter for target endianness
    pub fn new(target_endian: Endianness) -> Self {
        Self { target_endian }
    }

    /// Create converter for little-endian
    pub fn little_endian() -> Self {
        Self::new(Endianness::Little)
    }

    /// Create converter for big-endian
    pub fn big_endian() -> Self {
        Self::new(Endianness::Big)
    }

    /// Create converter for native endianness
    pub fn native() -> Self {
        Self::new(native_endianness())
    }

    /// Check if conversion is needed
    pub fn needs_conversion(&self) -> bool {
        self.target_endian != native_endianness()
    }

    /// Convert u16 to target endianness
    pub fn to_u16(&self, value: u16) -> u16 {
        match self.target_endian {
            Endianness::Little => value.to_le(),
            Endianness::Big => value.to_be(),
        }
    }

    /// Convert u32 to target endianness
    pub fn to_u32(&self, value: u32) -> u32 {
        match self.target_endian {
            Endianness::Little => value.to_le(),
            Endianness::Big => value.to_be(),
        }
    }

    /// Convert u64 to target endianness
    pub fn to_u64(&self, value: u64) -> u64 {
        match self.target_endian {
            Endianness::Little => value.to_le(),
            Endianness::Big => value.to_be(),
        }
    }

    /// Convert f32 to target endianness
    pub fn to_f32(&self, value: f32) -> u32 {
        let bits = value.to_bits();
        self.to_u32(bits)
    }

    /// Convert f64 to target endianness
    pub fn to_f64(&self, value: f64) -> u64 {
        let bits = value.to_bits();
        self.to_u64(bits)
    }

    /// Convert from u16 in target endianness
    pub fn from_u16(&self, value: u16) -> u16 {
        match self.target_endian {
            Endianness::Little => u16::from_le(value),
            Endianness::Big => u16::from_be(value),
        }
    }

    /// Convert from u32 in target endianness
    pub fn from_u32(&self, value: u32) -> u32 {
        match self.target_endian {
            Endianness::Little => u32::from_le(value),
            Endianness::Big => u32::from_be(value),
        }
    }

    /// Convert from u64 in target endianness
    pub fn from_u64(&self, value: u64) -> u64 {
        match self.target_endian {
            Endianness::Little => u64::from_le(value),
            Endianness::Big => u64::from_be(value),
        }
    }

    /// Convert from f32 bits in target endianness
    pub fn from_f32_bits(&self, bits: u32) -> f32 {
        let native_bits = self.from_u32(bits);
        f32::from_bits(native_bits)
    }

    /// Convert from f64 bits in target endianness
    pub fn from_f64_bits(&self, bits: u64) -> f64 {
        let native_bits = self.from_u64(bits);
        f64::from_bits(native_bits)
    }

    /// Write u16 to bytes in target endianness
    pub fn write_u16(&self, value: u16, buffer: &mut [u8]) -> Result<()> {
        if buffer.len() < 2 {
            return Err(SerializeError::BufferTooSmall);
        }
        
        let converted = self.to_u16(value);
        match self.target_endian {
            Endianness::Little => buffer[..2].copy_from_slice(&converted.to_le_bytes()),
            Endianness::Big => buffer[..2].copy_from_slice(&converted.to_be_bytes()),
        }
        Ok(())
    }

    /// Write u32 to bytes in target endianness
    pub fn write_u32(&self, value: u32, buffer: &mut [u8]) -> Result<()> {
        if buffer.len() < 4 {
            return Err(SerializeError::BufferTooSmall);
        }
        
        let converted = self.to_u32(value);
        match self.target_endian {
            Endianness::Little => buffer[..4].copy_from_slice(&converted.to_le_bytes()),
            Endianness::Big => buffer[..4].copy_from_slice(&converted.to_be_bytes()),
        }
        Ok(())
    }

    /// Write u64 to bytes in target endianness
    pub fn write_u64(&self, value: u64, buffer: &mut [u8]) -> Result<()> {
        if buffer.len() < 8 {
            return Err(SerializeError::BufferTooSmall);
        }
        
        let converted = self.to_u64(value);
        match self.target_endian {
            Endianness::Little => buffer[..8].copy_from_slice(&converted.to_le_bytes()),
            Endianness::Big => buffer[..8].copy_from_slice(&converted.to_be_bytes()),
        }
        Ok(())
    }

    /// Write f32 to bytes in target endianness
    pub fn write_f32(&self, value: f32, buffer: &mut [u8]) -> Result<()> {
        let bits = self.to_f32(value);
        self.write_u32(bits, buffer)
    }

    /// Write f64 to bytes in target endianness
    pub fn write_f64(&self, value: f64, buffer: &mut [u8]) -> Result<()> {
        let bits = self.to_f64(value);
        self.write_u64(bits, buffer)
    }

    /// Read u16 from bytes in target endianness
    pub fn read_u16(&self, buffer: &[u8]) -> Result<u16> {
        if buffer.len() < 2 {
            return Err(SerializeError::UnexpectedEof);
        }
        
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(&buffer[..2]);
        
        let value = match self.target_endian {
            Endianness::Little => u16::from_le_bytes(bytes),
            Endianness::Big => u16::from_be_bytes(bytes),
        };
        
        Ok(self.from_u16(value))
    }

    /// Read u32 from bytes in target endianness
    pub fn read_u32(&self, buffer: &[u8]) -> Result<u32> {
        if buffer.len() < 4 {
            return Err(SerializeError::UnexpectedEof);
        }
        
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&buffer[..4]);
        
        let value = match self.target_endian {
            Endianness::Little => u32::from_le_bytes(bytes),
            Endianness::Big => u32::from_be_bytes(bytes),
        };
        
        Ok(self.from_u32(value))
    }

    /// Read u64 from bytes in target endianness
    pub fn read_u64(&self, buffer: &[u8]) -> Result<u64> {
        if buffer.len() < 8 {
            return Err(SerializeError::UnexpectedEof);
        }
        
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buffer[..8]);
        
        let value = match self.target_endian {
            Endianness::Little => u64::from_le_bytes(bytes),
            Endianness::Big => u64::from_be_bytes(bytes),
        };
        
        Ok(self.from_u64(value))
    }

    /// Read f32 from bytes in target endianness
    pub fn read_f32(&self, buffer: &[u8]) -> Result<f32> {
        let bits = self.read_u32(buffer)?;
        Ok(self.from_f32_bits(bits))
    }

    /// Read f64 from bytes in target endianness
    pub fn read_f64(&self, buffer: &[u8]) -> Result<f64> {
        let bits = self.read_u64(buffer)?;
        Ok(self.from_f64_bits(bits))
    }
}

/// Swap bytes for endianness conversion
pub mod swap {
    /// Swap bytes in u16
    pub fn swap_u16(value: u16) -> u16 {
        value.swap_bytes()
    }

    /// Swap bytes in u32
    pub fn swap_u32(value: u32) -> u32 {
        value.swap_bytes()
    }

    /// Swap bytes in u64
    pub fn swap_u64(value: u64) -> u64 {
        value.swap_bytes()
    }

    /// Swap bytes in slice of u16s
    pub fn swap_u16_slice(slice: &mut [u16]) {
        for value in slice {
            *value = value.swap_bytes();
        }
    }

    /// Swap bytes in slice of u32s
    pub fn swap_u32_slice(slice: &mut [u32]) {
        for value in slice {
            *value = value.swap_bytes();
        }
    }

    /// Swap bytes in slice of u64s
    pub fn swap_u64_slice(slice: &mut [u64]) {
        for value in slice {
            *value = value.swap_bytes();
        }
    }

    /// Swap bytes in slice of f32s
    pub fn swap_f32_slice(slice: &mut [f32]) {
        for value in slice {
            let bits = value.to_bits().swap_bytes();
            *value = f32::from_bits(bits);
        }
    }

    /// Swap bytes in slice of f64s
    pub fn swap_f64_slice(slice: &mut [f64]) {
        for value in slice {
            let bits = value.to_bits().swap_bytes();
            *value = f64::from_bits(bits);
        }
    }
}

/// Network byte order utilities (big-endian)
pub mod network {
    use super::*;

    /// Convert u16 to network byte order
    pub fn htons(value: u16) -> u16 {
        value.to_be()
    }

    /// Convert u32 to network byte order
    pub fn htonl(value: u32) -> u32 {
        value.to_be()
    }

    /// Convert u16 from network byte order
    pub fn ntohs(value: u16) -> u16 {
        u16::from_be(value)
    }

    /// Convert u32 from network byte order
    pub fn ntohl(value: u32) -> u32 {
        u32::from_be(value)
    }

    /// Write u16 in network byte order
    pub fn write_u16(value: u16, buffer: &mut [u8]) -> Result<()> {
        if buffer.len() < 2 {
            return Err(SerializeError::BufferTooSmall);
        }
        buffer[..2].copy_from_slice(&htons(value).to_be_bytes());
        Ok(())
    }

    /// Write u32 in network byte order
    pub fn write_u32(value: u32, buffer: &mut [u8]) -> Result<()> {
        if buffer.len() < 4 {
            return Err(SerializeError::BufferTooSmall);
        }
        buffer[..4].copy_from_slice(&htonl(value).to_be_bytes());
        Ok(())
    }

    /// Read u16 from network byte order
    pub fn read_u16(buffer: &[u8]) -> Result<u16> {
        if buffer.len() < 2 {
            return Err(SerializeError::UnexpectedEof);
        }
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(&buffer[..2]);
        Ok(ntohs(u16::from_be_bytes(bytes)))
    }

    /// Read u32 from network byte order
    pub fn read_u32(buffer: &[u8]) -> Result<u32> {
        if buffer.len() < 4 {
            return Err(SerializeError::UnexpectedEof);
        }
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&buffer[..4]);
        Ok(ntohl(u32::from_be_bytes(bytes)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endianness_detection() {
        let native = native_endianness();
        assert!(native == Endianness::Little || native == Endianness::Big);
        
        if cfg!(target_endian = "little") {
            assert_eq!(native, Endianness::Little);
            assert!(is_little_endian());
            assert!(!is_big_endian());
        } else {
            assert_eq!(native, Endianness::Big);
            assert!(!is_little_endian());
            assert!(is_big_endian());
        }
    }

    #[test]
    fn test_endian_converter() {
        let converter = EndianConverter::little_endian();
        
        let value = 0x12345678u32;
        let converted = converter.to_u32(value);
        let back = converter.from_u32(converted);
        
        assert_eq!(value, back);
    }

    #[test]
    fn test_buffer_operations() {
        let converter = EndianConverter::little_endian();
        let mut buffer = [0u8; 4];
        
        let value = 0x12345678u32;
        converter.write_u32(value, &mut buffer).unwrap();
        
        let read_value = converter.read_u32(&buffer).unwrap();
        assert_eq!(value, read_value);
    }

    #[test]
    fn test_float_conversion() {
        let converter = EndianConverter::little_endian();
        let mut buffer = [0u8; 4];
        
        let value = 3.14159f32;
        converter.write_f32(value, &mut buffer).unwrap();
        
        let read_value = converter.read_f32(&buffer).unwrap();
        assert!((value - read_value).abs() < 1e-6);
    }

    #[test]
    fn test_byte_swapping() {
        let value = 0x1234u16;
        let swapped = swap::swap_u16(value);
        assert_eq!(swapped, 0x3412u16);
        
        let double_swapped = swap::swap_u16(swapped);
        assert_eq!(double_swapped, value);
    }

    #[test]
    fn test_network_byte_order() {
        let value = 0x12345678u32;
        let network = network::htonl(value);
        let back = network::ntohl(network);
        assert_eq!(value, back);
        
        let mut buffer = [0u8; 4];
        network::write_u32(value, &mut buffer).unwrap();
        let read_value = network::read_u32(&buffer).unwrap();
        assert_eq!(value, read_value);
    }

    #[test]
    fn test_slice_swapping() {
        let mut values = [0x1234u16, 0x5678u16, 0x9ABCu16];
        let original = values;
        
        swap::swap_u16_slice(&mut values);
        assert_ne!(values, original);
        
        swap::swap_u16_slice(&mut values);
        assert_eq!(values, original);
    }
}