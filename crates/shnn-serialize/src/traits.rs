//! Serialization traits for zero-copy operations
//! 
//! Provides core traits for serializing and deserializing neuromorphic data
//! with zero-copy optimizations and deterministic memory layout.

use crate::{Result, SerializeError};

/// Trait for types that can be serialized
pub trait Serialize {
    /// Serialize data into the provided buffer
    fn serialize(&self, buffer: &mut [u8]) -> Result<usize>;
    
    /// Get the serialized size of this data
    fn serialized_size(&self) -> usize;
}

/// Trait for types that can be deserialized
pub trait Deserialize<'a>: Sized {
    /// Deserialize data from the provided buffer
    fn deserialize(buffer: &'a [u8]) -> Result<Self>;
}

/// Trait for getting serialized size without serializing
pub trait SerializeSize {
    /// Get the size needed to serialize this data
    fn serialize_size(&self) -> usize;
}

/// Zero-copy serialization trait for types with fixed layout
pub trait ZeroCopySerialize: Copy {
    /// Serialize with zero-copy by casting to bytes
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const u8,
                core::mem::size_of::<Self>(),
            )
        }
    }
    
    /// Deserialize with zero-copy by casting from bytes
    fn from_bytes(bytes: &[u8]) -> Result<&Self> {
        if bytes.len() < core::mem::size_of::<Self>() {
            return Err(SerializeError::UnexpectedEof);
        }
        
        let ptr = bytes.as_ptr() as *const Self;
        if !crate::utils::is_aligned(ptr as *const u8, core::mem::align_of::<Self>()) {
            return Err(SerializeError::AlignmentError);
        }
        
        Ok(unsafe { &*ptr })
    }
}

// Implement basic serialization for primitive types
impl Serialize for u8 {
    fn serialize(&self, buffer: &mut [u8]) -> Result<usize> {
        if buffer.is_empty() {
            return Err(SerializeError::BufferTooSmall);
        }
        buffer[0] = *self;
        Ok(1)
    }
    
    fn serialized_size(&self) -> usize {
        1
    }
}

impl<'a> Deserialize<'a> for u8 {
    fn deserialize(buffer: &'a [u8]) -> Result<Self> {
        if buffer.is_empty() {
            return Err(SerializeError::UnexpectedEof);
        }
        Ok(buffer[0])
    }
}

impl Serialize for u32 {
    fn serialize(&self, buffer: &mut [u8]) -> Result<usize> {
        if buffer.len() < 4 {
            return Err(SerializeError::BufferTooSmall);
        }
        let bytes = self.to_le_bytes();
        buffer[..4].copy_from_slice(&bytes);
        Ok(4)
    }
    
    fn serialized_size(&self) -> usize {
        4
    }
}

impl<'a> Deserialize<'a> for u32 {
    fn deserialize(buffer: &'a [u8]) -> Result<Self> {
        if buffer.len() < 4 {
            return Err(SerializeError::UnexpectedEof);
        }
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&buffer[..4]);
        Ok(u32::from_le_bytes(bytes))
    }
}

impl Serialize for f32 {
    fn serialize(&self, buffer: &mut [u8]) -> Result<usize> {
        let bits = self.to_bits();
        bits.serialize(buffer)
    }
    
    fn serialized_size(&self) -> usize {
        4
    }
}

impl<'a> Deserialize<'a> for f32 {
    fn deserialize(buffer: &'a [u8]) -> Result<Self> {
        let bits = u32::deserialize(buffer)?;
        Ok(f32::from_bits(bits))
    }
}

// Implement for arrays
impl<T: Serialize, const N: usize> Serialize for [T; N] {
    fn serialize(&self, buffer: &mut [u8]) -> Result<usize> {
        let mut offset = 0;
        for item in self.iter() {
            let size = item.serialize(&mut buffer[offset..])?;
            offset += size;
        }
        Ok(offset)
    }
    
    fn serialized_size(&self) -> usize {
        self.iter().map(|item| item.serialized_size()).sum()
    }
}

// Implement zero-copy for POD types
impl ZeroCopySerialize for u8 {}
impl ZeroCopySerialize for u16 {}
impl ZeroCopySerialize for u32 {}
impl ZeroCopySerialize for u64 {}
impl ZeroCopySerialize for f32 {}
impl ZeroCopySerialize for f64 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u8_serialize() {
        let value = 42u8;
        let mut buffer = [0u8; 1];
        
        let size = value.serialize(&mut buffer).unwrap();
        assert_eq!(size, 1);
        assert_eq!(buffer[0], 42);
        
        let deserialized = u8::deserialize(&buffer).unwrap();
        assert_eq!(deserialized, 42);
    }

    #[test]
    fn test_u32_serialize() {
        let value = 0x12345678u32;
        let mut buffer = [0u8; 4];
        
        let size = value.serialize(&mut buffer).unwrap();
        assert_eq!(size, 4);
        
        let deserialized = u32::deserialize(&buffer).unwrap();
        assert_eq!(deserialized, 0x12345678);
    }

    #[test]
    fn test_f32_serialize() {
        let value = 3.14159f32;
        let mut buffer = [0u8; 4];
        
        let size = value.serialize(&mut buffer).unwrap();
        assert_eq!(size, 4);
        
        let deserialized = f32::deserialize(&buffer).unwrap();
        assert!((deserialized - 3.14159).abs() < 1e-6);
    }

    #[test]
    fn test_zero_copy() {
        let value = 0x12345678u32;
        let bytes = value.as_bytes();
        assert_eq!(bytes.len(), 4);
        
        let deserialized = u32::from_bytes(bytes).unwrap();
        assert_eq!(*deserialized, 0x12345678);
    }
}