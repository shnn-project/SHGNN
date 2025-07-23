//! SIMD-optimized operations for neuromorphic computations
//! 
//! Provides portable SIMD implementations with fallbacks for platforms
//! without SIMD support. Zero external dependencies.

#![cfg(feature = "simd")]

use core::simd::{f32x8, f32x16, SimdFloat, SimdPartialOrd};
use crate::{Float, Vector, Matrix};

/// SIMD vector width for f32 operations
pub const SIMD_WIDTH_F32: usize = 8;

/// SIMD operations trait for vectorized computations
pub trait SimdOps {
    /// Apply function using SIMD where possible
    fn simd_apply<F>(&mut self, f: F)
    where
        F: Fn(f32x8) -> f32x8;

    /// SIMD reduction operation
    fn simd_reduce<F>(&self, f: F) -> Float
    where
        F: Fn(f32x8) -> Float;

    /// SIMD binary operation
    fn simd_binary<F>(&mut self, other: &Self, f: F)
    where
        F: Fn(f32x8, f32x8) -> f32x8;
}

impl SimdOps for Vector {
    fn simd_apply<F>(&mut self, f: F)
    where
        F: Fn(f32x8) -> f32x8,
    {
        let data = self.as_mut_slice();
        let chunks = data.len() / SIMD_WIDTH_F32;

        // Process SIMD chunks
        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            let chunk = f32x8::from_slice(&data[offset..offset + SIMD_WIDTH_F32]);
            let result = f(chunk);
            result.copy_to_slice(&mut data[offset..offset + SIMD_WIDTH_F32]);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..data.len() {
            let scalar_chunk = f32x8::splat(data[i]);
            let result = f(scalar_chunk);
            data[i] = result.to_array()[0];
        }
    }

    fn simd_reduce<F>(&self, f: F) -> Float
    where
        F: Fn(f32x8) -> Float,
    {
        let data = self.as_slice();
        let chunks = data.len() / SIMD_WIDTH_F32;
        let mut accumulator = 0.0;

        // Process SIMD chunks
        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            let chunk = f32x8::from_slice(&data[offset..offset + SIMD_WIDTH_F32]);
            accumulator += f(chunk);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..data.len() {
            let scalar_chunk = f32x8::splat(data[i]);
            accumulator += f(scalar_chunk);
        }

        accumulator
    }

    fn simd_binary<F>(&mut self, other: &Self, f: F)
    where
        F: Fn(f32x8, f32x8) -> f32x8,
    {
        assert_eq!(self.len(), other.len());

        let self_data = self.as_mut_slice();
        let other_data = other.as_slice();
        let chunks = self_data.len() / SIMD_WIDTH_F32;

        // Process SIMD chunks
        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            let a = f32x8::from_slice(&self_data[offset..offset + SIMD_WIDTH_F32]);
            let b = f32x8::from_slice(&other_data[offset..offset + SIMD_WIDTH_F32]);
            let result = f(a, b);
            result.copy_to_slice(&mut self_data[offset..offset + SIMD_WIDTH_F32]);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..self_data.len() {
            let a = f32x8::splat(self_data[i]);
            let b = f32x8::splat(other_data[i]);
            let result = f(a, b);
            self_data[i] = result.to_array()[0];
        }
    }
}

impl SimdOps for Matrix {
    fn simd_apply<F>(&mut self, f: F)
    where
        F: Fn(f32x8) -> f32x8,
    {
        let data = self.as_mut_slice();
        let chunks = data.len() / SIMD_WIDTH_F32;

        // Process SIMD chunks
        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            let chunk = f32x8::from_slice(&data[offset..offset + SIMD_WIDTH_F32]);
            let result = f(chunk);
            result.copy_to_slice(&mut data[offset..offset + SIMD_WIDTH_F32]);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..data.len() {
            let scalar_chunk = f32x8::splat(data[i]);
            let result = f(scalar_chunk);
            data[i] = result.to_array()[0];
        }
    }

    fn simd_reduce<F>(&self, f: F) -> Float
    where
        F: Fn(f32x8) -> Float,
    {
        let data = self.as_slice();
        let chunks = data.len() / SIMD_WIDTH_F32;
        let mut accumulator = 0.0;

        // Process SIMD chunks
        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            let chunk = f32x8::from_slice(&data[offset..offset + SIMD_WIDTH_F32]);
            accumulator += f(chunk);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..data.len() {
            let scalar_chunk = f32x8::splat(data[i]);
            accumulator += f(scalar_chunk);
        }

        accumulator
    }

    fn simd_binary<F>(&mut self, other: &Self, f: F)
    where
        F: Fn(f32x8, f32x8) -> f32x8,
    {
        assert_eq!(self.dims(), other.dims());

        let self_data = self.as_mut_slice();
        let other_data = other.as_slice();
        let chunks = self_data.len() / SIMD_WIDTH_F32;

        // Process SIMD chunks
        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            let a = f32x8::from_slice(&self_data[offset..offset + SIMD_WIDTH_F32]);
            let b = f32x8::from_slice(&other_data[offset..offset + SIMD_WIDTH_F32]);
            let result = f(a, b);
            result.copy_to_slice(&mut self_data[offset..offset + SIMD_WIDTH_F32]);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..self_data.len() {
            let a = f32x8::splat(self_data[i]);
            let b = f32x8::splat(other_data[i]);
            let result = f(a, b);
            self_data[i] = result.to_array()[0];
        }
    }
}

/// Neuromorphic-specific SIMD operations
pub struct NeuromorphicSimd;

impl NeuromorphicSimd {
    /// SIMD membrane potential integration
    pub fn integrate_membrane_potentials(
        membrane: &mut [Float],
        input_current: &[Float],
        decay_factor: Float,
    ) {
        assert_eq!(membrane.len(), input_current.len());
        
        let chunks = membrane.len() / SIMD_WIDTH_F32;
        let decay_vec = f32x8::splat(decay_factor);

        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            
            // Load membrane potentials and input currents
            let mut mem = f32x8::from_slice(&membrane[offset..offset + SIMD_WIDTH_F32]);
            let input = f32x8::from_slice(&input_current[offset..offset + SIMD_WIDTH_F32]);
            
            // Apply decay and add input: V = V * decay + I
            mem = mem * decay_vec + input;
            
            // Store result
            mem.copy_to_slice(&mut membrane[offset..offset + SIMD_WIDTH_F32]);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..membrane.len() {
            membrane[i] = membrane[i] * decay_factor + input_current[i];
        }
    }

    /// SIMD spike detection and reset
    pub fn detect_spikes_and_reset(
        membrane: &mut [Float],
        spikes: &mut [bool],
        threshold: Float,
        reset_potential: Float,
    ) {
        assert_eq!(membrane.len(), spikes.len());
        
        let chunks = membrane.len() / SIMD_WIDTH_F32;
        let threshold_vec = f32x8::splat(threshold);
        let reset_vec = f32x8::splat(reset_potential);

        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            
            // Load membrane potentials
            let mem = f32x8::from_slice(&membrane[offset..offset + SIMD_WIDTH_F32]);
            
            // Create spike mask (mem >= threshold)
            let spike_mask = mem.simd_ge(threshold_vec);
            
            // Apply reset where spikes occurred
            let reset_mem = spike_mask.select(reset_vec, mem);
            reset_mem.copy_to_slice(&mut membrane[offset..offset + SIMD_WIDTH_F32]);
            
            // Store spike indicators
            for j in 0..SIMD_WIDTH_F32 {
                if offset + j < spikes.len() {
                    spikes[offset + j] = spike_mask.to_array()[j];
                }
            }
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..membrane.len() {
            if membrane[i] >= threshold {
                spikes[i] = true;
                membrane[i] = reset_potential;
            } else {
                spikes[i] = false;
            }
        }
    }

    /// SIMD synaptic weight update (simplified STDP)
    pub fn update_synaptic_weights(
        weights: &mut [Float],
        pre_spikes: &[bool],
        post_spikes: &[bool],
        learning_rate: Float,
        depression_factor: Float,
    ) {
        assert_eq!(weights.len(), pre_spikes.len());
        assert_eq!(weights.len(), post_spikes.len());
        
        let chunks = weights.len() / SIMD_WIDTH_F32;
        let lr_vec = f32x8::splat(learning_rate);
        let dep_vec = f32x8::splat(depression_factor);
        let zero = f32x8::splat(0.0);

        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            
            // Load current weights
            let mut w = f32x8::from_slice(&weights[offset..offset + SIMD_WIDTH_F32]);
            
            // Create spike vectors (convert bool to float)
            let mut pre_vec = f32x8::splat(0.0);
            let mut post_vec = f32x8::splat(0.0);
            
            for j in 0..SIMD_WIDTH_F32 {
                if offset + j < pre_spikes.len() {
                    pre_vec = pre_vec.replace(j, if pre_spikes[offset + j] { 1.0 } else { 0.0 });
                    post_vec = post_vec.replace(j, if post_spikes[offset + j] { 1.0 } else { 0.0 });
                }
            }
            
            // STDP update rule: w += lr * (post * pre - depression * (post + pre))
            let potentiation = post_vec * pre_vec * lr_vec;
            let depression = (post_vec + pre_vec) * dep_vec * lr_vec;
            w = w + potentiation - depression;
            
            // Store updated weights
            w.copy_to_slice(&mut weights[offset..offset + SIMD_WIDTH_F32]);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..weights.len() {
            let pre = if pre_spikes[i] { 1.0 } else { 0.0 };
            let post = if post_spikes[i] { 1.0 } else { 0.0 };
            
            let potentiation = post * pre * learning_rate;
            let depression = (post + pre) * depression_factor * learning_rate;
            weights[i] += potentiation - depression;
        }
    }

    /// SIMD convolution for spike patterns
    pub fn convolve_spike_trains(
        input: &[Float],
        kernel: &[Float],
        output: &mut [Float],
    ) {
        assert_eq!(output.len(), input.len());
        
        let kernel_size = kernel.len();
        let half_kernel = kernel_size / 2;
        
        for i in 0..output.len() {
            let mut sum = 0.0;
            
            // Convolution with boundary handling
            for k in 0..kernel_size {
                let input_idx = if i + k >= half_kernel && i + k - half_kernel < input.len() {
                    i + k - half_kernel
                } else {
                    // Use zero-padding for boundaries
                    continue;
                };
                
                sum += input[input_idx] * kernel[k];
            }
            
            output[i] = sum;
        }
    }

    /// SIMD batch normalization for neural layers
    pub fn batch_normalize(
        data: &mut [Float],
        mean: Float,
        variance: Float,
        epsilon: Float,
    ) {
        let chunks = data.len() / SIMD_WIDTH_F32;
        let mean_vec = f32x8::splat(mean);
        let inv_std = f32x8::splat(1.0 / (variance + epsilon).sqrt());

        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            
            // Load data
            let x = f32x8::from_slice(&data[offset..offset + SIMD_WIDTH_F32]);
            
            // Normalize: (x - mean) / sqrt(variance + epsilon)
            let normalized = (x - mean_vec) * inv_std;
            
            // Store result
            normalized.copy_to_slice(&mut data[offset..offset + SIMD_WIDTH_F32]);
        }

        // Handle remaining elements
        let inv_std_scalar = 1.0 / (variance + epsilon).sqrt();
        for i in (chunks * SIMD_WIDTH_F32)..data.len() {
            data[i] = (data[i] - mean) * inv_std_scalar;
        }
    }

    /// SIMD element-wise activation functions
    pub fn apply_relu_simd(data: &mut [Float]) {
        let chunks = data.len() / SIMD_WIDTH_F32;
        let zero = f32x8::splat(0.0);

        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            let x = f32x8::from_slice(&data[offset..offset + SIMD_WIDTH_F32]);
            let result = x.simd_max(zero);
            result.copy_to_slice(&mut data[offset..offset + SIMD_WIDTH_F32]);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..data.len() {
            data[i] = data[i].max(0.0);
        }
    }

    /// SIMD Leaky ReLU activation
    pub fn apply_leaky_relu_simd(data: &mut [Float], alpha: Float) {
        let chunks = data.len() / SIMD_WIDTH_F32;
        let zero = f32x8::splat(0.0);
        let alpha_vec = f32x8::splat(alpha);

        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            let x = f32x8::from_slice(&data[offset..offset + SIMD_WIDTH_F32]);
            let mask = x.simd_gt(zero);
            let result = mask.select(x, alpha_vec * x);
            result.copy_to_slice(&mut data[offset..offset + SIMD_WIDTH_F32]);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..data.len() {
            if data[i] <= 0.0 {
                data[i] *= alpha;
            }
        }
    }

    /// SIMD fast approximate exponential (for softmax)
    pub fn fast_exp_simd(data: &mut [Float]) {
        let chunks = data.len() / SIMD_WIDTH_F32;

        for i in 0..chunks {
            let offset = i * SIMD_WIDTH_F32;
            let x = f32x8::from_slice(&data[offset..offset + SIMD_WIDTH_F32]);
            
            // Fast exp approximation using polynomial
            // exp(x) ≈ 1 + x + x²/2 + x³/6 (for |x| < 1)
            let x2 = x * x;
            let x3 = x2 * x;
            let one = f32x8::splat(1.0);
            let half = f32x8::splat(0.5);
            let sixth = f32x8::splat(1.0 / 6.0);
            
            let result = one + x + x2 * half + x3 * sixth;
            result.copy_to_slice(&mut data[offset..offset + SIMD_WIDTH_F32]);
        }

        // Handle remaining elements
        for i in (chunks * SIMD_WIDTH_F32)..data.len() {
            let x = data[i];
            let x2 = x * x;
            let x3 = x2 * x;
            data[i] = 1.0 + x + x2 * 0.5 + x3 / 6.0;
        }
    }
}

/// SIMD utilities for performance monitoring
pub struct SimdUtils;

impl SimdUtils {
    /// Check if SIMD is available and working
    pub fn is_simd_available() -> bool {
        // On platforms with portable_simd, this should always be true
        // when the feature is enabled
        cfg!(feature = "simd")
    }

    /// Get optimal vector size for current platform
    pub fn optimal_vector_size() -> usize {
        SIMD_WIDTH_F32
    }

    /// Align length to SIMD boundary
    pub fn align_to_simd(length: usize) -> usize {
        (length + SIMD_WIDTH_F32 - 1) & !(SIMD_WIDTH_F32 - 1)
    }

    /// Calculate SIMD efficiency for given array size
    pub fn simd_efficiency(length: usize) -> Float {
        let aligned = Self::align_to_simd(length);
        length as Float / aligned as Float
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_vector_operations() {
        let mut v1 = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        
        // Test SIMD apply (square operation)
        v1.simd_apply(|x| x * x);
        
        // Should be [1, 4, 9, 16, 25, 36, 49, 64]
        assert_eq!(v1[0], 1.0);
        assert_eq!(v1[1], 4.0);
        assert_eq!(v1[7], 64.0);
    }

    #[test]
    fn test_membrane_integration() {
        let mut membrane = [0.0; 8];
        let input = [1.0; 8];
        let decay = 0.9;

        NeuromorphicSimd::integrate_membrane_potentials(&mut membrane, &input, decay);
        
        // Should all be 1.0 (0 * 0.9 + 1.0)
        for &v in &membrane {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_spike_detection() {
        let mut membrane = [-60.0, -40.0, -30.0, -70.0, -45.0, -25.0, -80.0, -35.0];
        let mut spikes = [false; 8];
        let threshold = -50.0;
        let reset = -70.0;

        NeuromorphicSimd::detect_spikes_and_reset(&mut membrane, &mut spikes, threshold, reset);
        
        // Check spikes: only potentials >= -50.0 should spike
        assert!(!spikes[0]); // -60.0
        assert!(spikes[1]);  // -40.0
        assert!(spikes[2]);  // -30.0
        assert!(!spikes[3]); // -70.0
        assert!(spikes[4]);  // -45.0
        assert!(spikes[5]);  // -25.0
        assert!(!spikes[6]); // -80.0
        assert!(spikes[7]);  // -35.0
        
        // Check reset values
        assert_eq!(membrane[1], reset); // Should be reset
        assert_eq!(membrane[0], -60.0); // Should be unchanged
    }

    #[test]
    fn test_relu_activation() {
        let mut data = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        
        NeuromorphicSimd::apply_relu_simd(&mut data);
        
        let expected = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        for (actual, &expected) in data.iter().zip(expected.iter()) {
            assert_eq!(*actual, expected);
        }
    }

    #[test]
    fn test_simd_utils() {
        assert!(SimdUtils::is_simd_available());
        assert_eq!(SimdUtils::optimal_vector_size(), 8);
        assert_eq!(SimdUtils::align_to_simd(10), 16);
        assert!((SimdUtils::simd_efficiency(8) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_normalization() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mean = 4.5;
        let variance = 5.25; // Variance of [1,2,3,4,5,6,7,8]
        let epsilon = 1e-8;

        NeuromorphicSimd::batch_normalize(&mut data, mean, variance, epsilon);
        
        // Check that mean is approximately 0
        let new_mean: Float = data.iter().sum::<Float>() / data.len() as Float;
        assert!(new_mean.abs() < 1e-5);
    }
}

/// Standalone SIMD functions for compatibility
pub fn simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    // Basic implementation - add SIMD optimization later
    for ((ai, bi), ri) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
        *ri = ai + bi;
    }
}

pub fn simd_multiply_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    // Basic implementation - add SIMD optimization later
    for ((ai, bi), ri) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
        *ri = ai * bi;
    }
}

pub fn simd_activation(input: &[f32], output: &mut [f32], activation: fn(f32) -> f32) {
    // Basic implementation - add SIMD optimization later
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = activation(*inp);
    }
}