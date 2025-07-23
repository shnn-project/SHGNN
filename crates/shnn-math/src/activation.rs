//! Neural activation functions optimized for neuromorphic computations
//! 
//! Provides fast implementations of common activation functions with optional
//! SIMD acceleration and approximations for real-time applications.

use crate::{Float, constants::EPSILON, math::FloatMath};

/// Trait for activation functions
pub trait ActivationFunction {
    /// Apply activation function
    fn activate(&self, x: Float) -> Float;
    
    /// Compute derivative (for backpropagation)
    fn derivative(&self, x: Float) -> Float;
    
    /// Apply activation in-place to slice
    fn activate_slice(&self, data: &mut [Float]) {
        for x in data {
            *x = self.activate(*x);
        }
    }
    
    /// Apply activation with SIMD optimization
    #[cfg(feature = "simd")]
    fn activate_simd(&self, data: &mut [Float]);
}

/// Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    #[inline]
    fn activate(&self, x: Float) -> Float {
        1.0 / (1.0 + (-x).exp())
    }
    
    #[inline]
    fn derivative(&self, x: Float) -> Float {
        let s = self.activate(x);
        s * (1.0 - s)
    }
    
    #[cfg(feature = "simd")]
    fn activate_simd(&self, data: &mut [Float]) {
        use core::simd::{f32x8, SimdFloat};
        
        let chunks = data.len() / 8;
        let one = f32x8::splat(1.0);
        
        for i in 0..chunks {
            let offset = i * 8;
            let x = f32x8::from_slice(&data[offset..offset + 8]);
            let result = one / (one + (-x).exp());
            result.copy_to_slice(&mut data[offset..offset + 8]);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..data.len() {
            data[i] = self.activate(data[i]);
        }
    }
}

/// Tanh activation function: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl ActivationFunction for Tanh {
    #[inline]
    fn activate(&self, x: Float) -> Float {
        x.tanh()
    }
    
    #[inline]
    fn derivative(&self, x: Float) -> Float {
        let t = x.tanh();
        1.0 - t * t
    }
    
    #[cfg(feature = "simd")]
    fn activate_simd(&self, data: &mut [Float]) {
        use core::simd::{f32x8, SimdFloat};
        
        let chunks = data.len() / 8;
        
        for i in 0..chunks {
            let offset = i * 8;
            let x = f32x8::from_slice(&data[offset..offset + 8]);
            let result = x.tanh();
            result.copy_to_slice(&mut data[offset..offset + 8]);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..data.len() {
            data[i] = self.activate(data[i]);
        }
    }
}

/// ReLU activation function: ReLU(x) = max(0, x)
#[derive(Debug, Clone, Copy)]
pub struct ReLU;

impl ActivationFunction for ReLU {
    #[inline]
    fn activate(&self, x: Float) -> Float {
        x.max(0.0)
    }
    
    #[inline]
    fn derivative(&self, x: Float) -> Float {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
    
    #[cfg(feature = "simd")]
    fn activate_simd(&self, data: &mut [Float]) {
        use core::simd::{f32x8, SimdFloat};
        
        let chunks = data.len() / 8;
        let zero = f32x8::splat(0.0);
        
        for i in 0..chunks {
            let offset = i * 8;
            let x = f32x8::from_slice(&data[offset..offset + 8]);
            let result = x.simd_max(zero);
            result.copy_to_slice(&mut data[offset..offset + 8]);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..data.len() {
            data[i] = self.activate(data[i]);
        }
    }
}

/// Leaky ReLU activation function: LeakyReLU(x) = max(αx, x)
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    pub alpha: Float,
}

impl LeakyReLU {
    /// Create new Leaky ReLU with given slope for negative values
    pub fn new(alpha: Float) -> Self {
        Self { alpha }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl ActivationFunction for LeakyReLU {
    #[inline]
    fn activate(&self, x: Float) -> Float {
        if x > 0.0 { x } else { self.alpha * x }
    }
    
    #[inline]
    fn derivative(&self, x: Float) -> Float {
        if x > 0.0 { 1.0 } else { self.alpha }
    }
    
    #[cfg(feature = "simd")]
    fn activate_simd(&self, data: &mut [Float]) {
        use core::simd::{f32x8, SimdFloat};
        
        let chunks = data.len() / 8;
        let zero = f32x8::splat(0.0);
        let alpha = f32x8::splat(self.alpha);
        
        for i in 0..chunks {
            let offset = i * 8;
            let x = f32x8::from_slice(&data[offset..offset + 8]);
            let mask = x.simd_gt(zero);
            let result = mask.select(x, alpha * x);
            result.copy_to_slice(&mut data[offset..offset + 8]);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..data.len() {
            data[i] = self.activate(data[i]);
        }
    }
}

/// ELU activation function: ELU(x) = x if x > 0, α(e^x - 1) if x ≤ 0
#[derive(Debug, Clone, Copy)]
pub struct ELU {
    pub alpha: Float,
}

impl ELU {
    /// Create new ELU with given alpha parameter
    pub fn new(alpha: Float) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl ActivationFunction for ELU {
    #[inline]
    fn activate(&self, x: Float) -> Float {
        if x > 0.0 { 
            x 
        } else { 
            self.alpha * (x.exp() - 1.0) 
        }
    }
    
    #[inline]
    fn derivative(&self, x: Float) -> Float {
        if x > 0.0 { 
            1.0 
        } else { 
            self.alpha * x.exp() 
        }
    }
    
    #[cfg(feature = "simd")]
    fn activate_simd(&self, data: &mut [Float]) {
        use core::simd::{f32x8, SimdFloat};
        
        let chunks = data.len() / 8;
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);
        let alpha = f32x8::splat(self.alpha);
        
        for i in 0..chunks {
            let offset = i * 8;
            let x = f32x8::from_slice(&data[offset..offset + 8]);
            let mask = x.simd_gt(zero);
            let neg_result = alpha * (x.exp() - one);
            let result = mask.select(x, neg_result);
            result.copy_to_slice(&mut data[offset..offset + 8]);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..data.len() {
            data[i] = self.activate(data[i]);
        }
    }
}

/// Swish activation function: Swish(x) = x * σ(x)
#[derive(Debug, Clone, Copy)]
pub struct Swish;

impl ActivationFunction for Swish {
    #[inline]
    fn activate(&self, x: Float) -> Float {
        x * sigmoid(x)
    }
    
    #[inline]
    fn derivative(&self, x: Float) -> Float {
        let s = sigmoid(x);
        s + x * s * (1.0 - s)
    }
    
    #[cfg(feature = "simd")]
    fn activate_simd(&self, data: &mut [Float]) {
        use core::simd::{f32x8, SimdFloat};
        
        let chunks = data.len() / 8;
        let one = f32x8::splat(1.0);
        
        for i in 0..chunks {
            let offset = i * 8;
            let x = f32x8::from_slice(&data[offset..offset + 8]);
            let sigmoid_x = one / (one + (-x).exp());
            let result = x * sigmoid_x;
            result.copy_to_slice(&mut data[offset..offset + 8]);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..data.len() {
            data[i] = self.activate(data[i]);
        }
    }
}

/// Neuromorphic spike activation function
/// Integrates membrane potential and fires spikes when threshold is exceeded
#[derive(Debug, Clone, Copy)]
pub struct SpikeActivation {
    pub threshold: Float,
    pub reset: Float,
    pub decay: Float,
}

impl SpikeActivation {
    /// Create new spike activation with neuromorphic parameters
    pub fn new(threshold: Float, reset: Float, decay: Float) -> Self {
        Self { threshold, reset, decay }
    }
    
    /// Process membrane potentials and return spike indicators
    pub fn process_membrane_batch(&self, membrane: &mut [Float], spikes: &mut [bool]) {
        assert_eq!(membrane.len(), spikes.len());
        
        for (v, spike) in membrane.iter_mut().zip(spikes.iter_mut()) {
            // Apply membrane decay
            *v *= self.decay;
            
            // Check for spike
            if *v >= self.threshold {
                *spike = true;
                *v = self.reset; // Reset after spike
            } else {
                *spike = false;
            }
        }
    }
    
    /// Integrate synaptic input into membrane potential
    pub fn integrate_input(&self, membrane: &mut [Float], input: &[Float]) {
        assert_eq!(membrane.len(), input.len());
        
        for (v, i) in membrane.iter_mut().zip(input.iter()) {
            *v += i;
        }
    }
}

impl Default for SpikeActivation {
    fn default() -> Self {
        Self::new(
            crate::constants::THRESHOLD,
            crate::constants::RESET_POTENTIAL,
            (-1.0 / crate::constants::TAU_M).exp(), // Exponential decay factor
        )
    }
}

impl ActivationFunction for SpikeActivation {
    #[inline]
    fn activate(&self, x: Float) -> Float {
        if x >= self.threshold { 1.0 } else { 0.0 }
    }
    
    #[inline]
    fn derivative(&self, _x: Float) -> Float {
        // Derivative is technically a Dirac delta function
        // For practical purposes, return 0 (non-differentiable)
        0.0
    }
    
    #[cfg(feature = "simd")]
    fn activate_simd(&self, data: &mut [Float]) {
        use core::simd::{f32x8, SimdFloat};
        
        let chunks = data.len() / 8;
        let threshold = f32x8::splat(self.threshold);
        let one = f32x8::splat(1.0);
        let zero = f32x8::splat(0.0);
        
        for i in 0..chunks {
            let offset = i * 8;
            let x = f32x8::from_slice(&data[offset..offset + 8]);
            let mask = x.simd_ge(threshold);
            let result = mask.select(one, zero);
            result.copy_to_slice(&mut data[offset..offset + 8]);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..data.len() {
            data[i] = self.activate(data[i]);
        }
    }
}

/// Fast approximate sigmoid using polynomial approximation
#[inline]
pub fn sigmoid_approx(x: Float) -> Float {
    // Pade approximation: sigmoid(x) ≈ x / (1 + |x|)
    // Fast but less accurate than true sigmoid
    if x >= 0.0 {
        x / (1.0 + x)
    } else {
        x / (1.0 - x)
    }
}

/// Fast approximate tanh using polynomial approximation
#[inline]
pub fn tanh_approx(x: Float) -> Float {
    // Rational approximation for tanh
    let x2 = x * x;
    let a = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let b = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    a / b
}

/// Convenience functions for common activations
#[inline]
pub fn sigmoid(x: Float) -> Float {
    Sigmoid.activate(x)
}

#[inline]
pub fn tanh(x: Float) -> Float {
    Tanh.activate(x)
}

#[inline]
pub fn relu(x: Float) -> Float {
    ReLU.activate(x)
}

#[inline]
pub fn leaky_relu(x: Float) -> Float {
    LeakyReLU::default().activate(x)
}

#[inline]
pub fn elu(x: Float) -> Float {
    ELU::default().activate(x)
}

#[inline]
pub fn swish(x: Float) -> Float {
    Swish.activate(x)
}

/// Softmax activation for output layers
pub fn softmax(input: &[Float], output: &mut [Float]) {
    assert_eq!(input.len(), output.len());
    
    // Find maximum for numerical stability
    let max_val = input.iter().copied().fold(Float::NEG_INFINITY, Float::max);
    
    // Compute exponentials
    let mut sum = 0.0;
    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = (i - max_val).exp();
        sum += *o;
    }
    
    // Normalize
    if sum > EPSILON {
        for o in output.iter_mut() {
            *o /= sum;
        }
    }
}

/// Log-softmax activation (numerically stable)
pub fn log_softmax(input: &[Float], output: &mut [Float]) {
    assert_eq!(input.len(), output.len());
    
    // Find maximum for numerical stability
    let max_val = input.iter().copied().fold(Float::NEG_INFINITY, Float::max);
    
    // Compute log-sum-exp
    let mut sum_exp = 0.0;
    for &x in input.iter() {
        sum_exp += (x - max_val).exp();
    }
    let log_sum_exp = max_val + sum_exp.ln();
    
    // Compute log-softmax
    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = i - log_sum_exp;
    }
}

/// Gated Linear Unit (GLU) activation
pub fn glu(input: &[Float], output: &mut [Float]) {
    assert_eq!(input.len() % 2, 0);
    assert_eq!(output.len(), input.len() / 2);
    
    let half = input.len() / 2;
    for i in 0..half {
        let gate = sigmoid(input[i + half]);
        output[i] = input[i] * gate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid;
        assert!((sigmoid.activate(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid.activate(1000.0) > 0.99);
        assert!(sigmoid.activate(-1000.0) < 0.01);
    }

    #[test]
    fn test_relu() {
        let relu = ReLU;
        assert_eq!(relu.activate(-5.0), 0.0);
        assert_eq!(relu.activate(0.0), 0.0);
        assert_eq!(relu.activate(5.0), 5.0);
    }

    #[test]
    fn test_leaky_relu() {
        let leaky = LeakyReLU::new(0.1);
        assert_eq!(leaky.activate(-10.0), -1.0);
        assert_eq!(leaky.activate(0.0), 0.0);
        assert_eq!(leaky.activate(10.0), 10.0);
    }

    #[test]
    fn test_spike_activation() {
        let spike = SpikeActivation::default();
        assert_eq!(spike.activate(-70.0), 0.0); // Below threshold
        assert_eq!(spike.activate(-40.0), 1.0); // Above threshold
    }

    #[test]
    fn test_softmax() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        softmax(&input, &mut output);
        
        // Should sum to 1.0
        let sum: Float = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Largest input should produce largest output
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let sigmoid = Sigmoid;
        // Derivative at 0 should be 0.25
        assert!((sigmoid.derivative(0.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_approximations() {
        // Test that approximations are reasonably close to exact functions
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let exact_sigmoid = sigmoid(x);
            let approx_sigmoid = sigmoid_approx(x);
            assert!((exact_sigmoid - approx_sigmoid).abs() < 0.1);
            
            let exact_tanh = tanh(x);
            let approx_tanh = tanh_approx(x);
            assert!((exact_tanh - approx_tanh).abs() < 0.1);
        }
    }

    #[test]
    fn test_spike_processing() {
        let spike_fn = SpikeActivation::default();
        let mut membrane = [-70.0, -45.0, -30.0, -80.0];
        let mut spikes = [false; 4];
        
        spike_fn.process_membrane_batch(&mut membrane, &mut spikes);
        
        // Check which neurons spiked
        assert!(!spikes[0]); // -70.0 below threshold
        assert!(spikes[1]);  // -45.0 above threshold
        assert!(spikes[2]);  // -30.0 above threshold  
        assert!(!spikes[3]); // -80.0 below threshold
    }
}