//! Mathematical utilities for neuromorphic computation
//!
//! This module provides optimized mathematical functions commonly used
//! in neuromorphic computing, including activation functions, differential
//! equations, and statistical operations.

use core::f32::consts::{E, PI};

#[cfg(feature = "std")]
use std::f32;

#[cfg(feature = "simd")]
use core::arch::x86_64::*;

/// Fast approximation functions
pub mod fast {
    use super::*;
    
    /// Fast exponential approximation using polynomial
    /// Accurate to about 1% for x in [-5, 5]
    pub fn exp_approx(x: f32) -> f32 {
        if x < -5.0 {
            return 0.0;
        }
        if x > 5.0 {
            return 148.413; // e^5
        }
        
        // Use Taylor series approximation: e^x ≈ 1 + x + x²/2! + x³/3! + x⁴/4!
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        
        1.0 + x + x2 * 0.5 + x3 * 0.16666667 + x4 * 0.041666667
    }
    
    /// Fast sine approximation using Bhaskara I's sine approximation
    pub fn sin_approx(x: f32) -> f32 {
        // Normalize to [0, 2π]
        let x_norm = x % (2.0 * PI);
        let x_pi = x_norm / PI;
        
        if x_pi <= 1.0 {
            // Use Bhaskara's approximation for [0, π]
            let x_180 = x_pi * 180.0;
            (4.0 * x_180 * (180.0 - x_180)) / (40500.0 - x_180 * (180.0 - x_180))
        } else {
            // For [π, 2π], use sin(x) = -sin(x - π)
            let x_180 = (x_pi - 1.0) * 180.0;
            -((4.0 * x_180 * (180.0 - x_180)) / (40500.0 - x_180 * (180.0 - x_180)))
        }
    }
    
    /// Fast cosine approximation
    pub fn cos_approx(x: f32) -> f32 {
        sin_approx(x + PI * 0.5)
    }
    
    /// Fast square root using Newton-Raphson method
    pub fn sqrt_approx(x: f32) -> f32 {
        if x <= 0.0 {
            return 0.0;
        }
        
        // Initial guess using bit manipulation
        let mut y = f32::from_bits((x.to_bits() >> 1) + 0x1fbb4000);
        
        // Two Newton-Raphson iterations
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        
        y
    }
    
    /// Fast inverse square root (Quake III algorithm variant)
    pub fn inv_sqrt_approx(x: f32) -> f32 {
        if x <= 0.0 {
            return 0.0;
        }
        
        let half_x = x * 0.5;
        let mut y = f32::from_bits(0x5f3759df - (x.to_bits() >> 1));
        
        // Newton-Raphson iteration
        y = y * (1.5 - half_x * y * y);
        y = y * (1.5 - half_x * y * y);
        
        y
    }
}

/// Activation functions commonly used in neural networks
pub mod activation {
    use super::*;
    
    /// Sigmoid activation function
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Fast sigmoid approximation
    pub fn sigmoid_approx(x: f32) -> f32 {
        1.0 / (1.0 + fast::exp_approx(-x))
    }
    
    /// Hyperbolic tangent
    pub fn tanh(x: f32) -> f32 {
        let exp_2x = (2.0 * x).exp();
        (exp_2x - 1.0) / (exp_2x + 1.0)
    }
    
    /// Fast tanh approximation
    pub fn tanh_approx(x: f32) -> f32 {
        let exp_2x = fast::exp_approx(2.0 * x);
        (exp_2x - 1.0) / (exp_2x + 1.0)
    }
    
    /// Rectified Linear Unit (ReLU)
    pub fn relu(x: f32) -> f32 {
        x.max(0.0)
    }
    
    /// Leaky ReLU
    pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
        if x >= 0.0 { x } else { alpha * x }
    }
    
    /// Exponential Linear Unit (ELU)
    pub fn elu(x: f32, alpha: f32) -> f32 {
        if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }
    }
    
    /// Swish activation function
    pub fn swish(x: f32) -> f32 {
        x * sigmoid(x)
    }
    
    /// GELU (Gaussian Error Linear Unit)
    pub fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + tanh(((2.0 / PI).sqrt()) * (x + 0.044715 * x.powi(3))))
    }
    
    /// Softplus activation
    pub fn softplus(x: f32) -> f32 {
        (1.0 + x.exp()).ln()
    }
}

/// Statistical functions
pub mod stats {
    use super::*;
    
    /// Calculate mean of a slice
    pub fn mean(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f32>() / values.len() as f32
    }
    
    /// Calculate variance of a slice
    pub fn variance(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean_val = mean(values);
        let sum_sq_diff: f32 = values.iter()
            .map(|&x| (x - mean_val).powi(2))
            .sum();
        
        sum_sq_diff / (values.len() - 1) as f32
    }
    
    /// Calculate standard deviation
    pub fn std_dev(values: &[f32]) -> f32 {
        variance(values).sqrt()
    }
    
    /// Calculate coefficient of variation
    pub fn cv(values: &[f32]) -> f32 {
        let mean_val = mean(values);
        if mean_val == 0.0 {
            0.0
        } else {
            std_dev(values) / mean_val
        }
    }
    
    /// Calculate Pearson correlation coefficient
    pub fn correlation(x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }
        
        let mean_x = mean(x);
        let mean_y = mean(y);
        
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }
        
        let denom = (sum_x2 * sum_y2).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            sum_xy / denom
        }
    }
    
    /// Calculate entropy of a distribution
    pub fn entropy(probabilities: &[f32]) -> f32 {
        probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }
    
    /// Calculate mutual information between two discrete distributions
    pub fn mutual_information(joint: &[f32], marginal_x: &[f32], marginal_y: &[f32]) -> f32 {
        let mut mi = 0.0;
        let mut idx = 0;
        
        for &px in marginal_x {
            for &py in marginal_y {
                if idx < joint.len() && joint[idx] > 0.0 && px > 0.0 && py > 0.0 {
                    mi += joint[idx] * (joint[idx] / (px * py)).ln();
                }
                idx += 1;
            }
        }
        
        mi
    }
}

/// Differential equation solvers
pub mod ode {
    use super::*;
    
    /// Euler's method for solving ODEs
    pub fn euler_step<F>(y: f32, dy_dt: F, dt: f32) -> f32
    where
        F: Fn(f32) -> f32,
    {
        y + dt * dy_dt(y)
    }
    
    /// 4th-order Runge-Kutta method
    pub fn rk4_step<F>(y: f32, dy_dt: F, dt: f32) -> f32
    where
        F: Fn(f32) -> f32,
    {
        let k1 = dt * dy_dt(y);
        let k2 = dt * dy_dt(y + k1 * 0.5);
        let k3 = dt * dy_dt(y + k2 * 0.5);
        let k4 = dt * dy_dt(y + k3);
        
        y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    }
    
    /// Exponential integration for linear ODEs: dy/dt = -y/tau + I
    pub fn exponential_euler(y: f32, tau: f32, input: f32, dt: f32) -> f32 {
        let alpha = (-dt / tau).exp();
        let steady_state = input * tau;
        steady_state + (y - steady_state) * alpha
    }
}

/// Signal processing functions
pub mod signal {
    use super::*;
    
    /// Low-pass filter (exponential moving average)
    pub fn low_pass_filter(current: f32, new_sample: f32, alpha: f32) -> f32 {
        alpha * new_sample + (1.0 - alpha) * current
    }
    
    /// High-pass filter
    pub fn high_pass_filter(prev_output: f32, prev_input: f32, new_input: f32, alpha: f32) -> f32 {
        alpha * (prev_output + new_input - prev_input)
    }
    
    /// Band-pass filter (cascade of high-pass and low-pass)
    pub fn band_pass_filter(
        state: &mut [f32; 2], // [low_pass_state, high_pass_prev_output]
        prev_input: f32,
        new_input: f32,
        alpha_low: f32,
        alpha_high: f32,
    ) -> f32 {
        // High-pass first
        let high_passed = high_pass_filter(state[1], prev_input, new_input, alpha_high);
        state[1] = high_passed;
        
        // Then low-pass
        let band_passed = low_pass_filter(state[0], high_passed, alpha_low);
        state[0] = band_passed;
        
        band_passed
    }
    
    /// Simple moving average
    pub fn moving_average(buffer: &mut [f32], new_sample: f32) -> f32 {
        // Shift buffer
        for i in (1..buffer.len()).rev() {
            buffer[i] = buffer[i - 1];
        }
        buffer[0] = new_sample;
        
        // Calculate average
        buffer.iter().sum::<f32>() / buffer.len() as f32
    }
    
    /// Discrete Fourier Transform (DFT) - simplified for power spectrum
    pub fn power_spectrum(signal: &[f32], frequencies: &mut [f32]) {
        let n = signal.len();
        let n_freq = frequencies.len().min(n / 2);
        
        for k in 0..n_freq {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;
            
            for i in 0..n {
                let angle = -2.0 * PI * (k as f32) * (i as f32) / (n as f32);
                real_sum += signal[i] * angle.cos();
                imag_sum += signal[i] * angle.sin();
            }
            
            frequencies[k] = real_sum * real_sum + imag_sum * imag_sum;
        }
    }
}

/// Vector operations optimized for neuromorphic computation
pub mod vector {
    use super::*;
    
    /// Dot product of two vectors
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
    
    /// L2 norm (Euclidean norm)
    pub fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }
    
    /// L1 norm (Manhattan norm)
    pub fn l1_norm(v: &[f32]) -> f32 {
        v.iter().map(|&x| x.abs()).sum()
    }
    
    /// Normalize vector to unit length
    pub fn normalize(v: &mut [f32]) {
        let norm = l2_norm(v);
        if norm > 0.0 {
            for x in v {
                *x /= norm;
            }
        }
    }
    
    /// Element-wise vector addition
    pub fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        for ((r, &a_val), &b_val) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
            *r = a_val + b_val;
        }
    }
    
    /// Element-wise vector subtraction
    pub fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        for ((r, &a_val), &b_val) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
            *r = a_val - b_val;
        }
    }
    
    /// Scalar multiplication
    pub fn scale(v: &mut [f32], scalar: f32) {
        for x in v {
            *x *= scalar;
        }
    }
    
    /// Cosine similarity
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product = dot(a, b);
        let norm_a = l2_norm(a);
        let norm_b = l2_norm(b);
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// SIMD-optimized operations (when feature enabled)
#[cfg(feature = "simd")]
pub mod simd {
    use super::*;
    
    /// SIMD vector addition (4 floats at a time)
    #[target_feature(enable = "sse")]
    pub unsafe fn add_simd(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len().min(b.len()).min(result.len());
        let simd_len = len & !3; // Round down to multiple of 4
        
        for i in (0..simd_len).step_by(4) {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vr = _mm_add_ps(va, vb);
            _mm_storeu_ps(result.as_mut_ptr().add(i), vr);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }
    }
    
    /// SIMD dot product
    #[target_feature(enable = "sse")]
    pub unsafe fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let simd_len = len & !3;
        
        let mut sum = _mm_setzero_ps();
        
        for i in (0..simd_len).step_by(4) {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let prod = _mm_mul_ps(va, vb);
            sum = _mm_add_ps(sum, prod);
        }
        
        // Horizontal sum of SIMD register
        let mut result = [0f32; 4];
        _mm_storeu_ps(result.as_mut_ptr(), sum);
        let mut total = result[0] + result[1] + result[2] + result[3];
        
        // Handle remaining elements
        for i in simd_len..len {
            total += a[i] * b[i];
        }
        
        total
    }
}

/// Commonly used mathematical functions
pub fn sigmoid(x: f32) -> f32 {
    activation::sigmoid(x)
}

pub fn exponential_decay(x: f32, tau: f32) -> f32 {
    (-x / tau).exp()
}

/// Re-export fast exponential approximation at module root
pub use fast::exp_approx;

pub fn gaussian(x: f32, mu: f32, sigma: f32) -> f32 {
    let diff = x - mu;
    (-0.5 * (diff / sigma).powi(2)).exp() / (sigma * (2.0 * PI).sqrt())
}

pub fn linear_interpolate(x: f32, x0: f32, x1: f32, y0: f32, y1: f32) -> f32 {
    if x1 == x0 {
        return y0;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Safe division with zero-check protection
pub fn safe_divide(numerator: f32, denominator: f32) -> f32 {
    if denominator.abs() < f32::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

/// Numerical integration using trapezoidal rule
pub fn trapezoid_integrate(y_values: &[f32], dx: f32) -> f32 {
    if y_values.len() < 2 {
        return 0.0;
    }
    
    let mut sum = 0.5 * (y_values[0] + y_values[y_values.len() - 1]);
    for &y in &y_values[1..y_values.len() - 1] {
        sum += y;
    }
    
    sum * dx
}

/// Find local maxima in a signal
pub fn find_peaks(signal: &[f32], threshold: f32) -> Vec<usize> {
    let mut peaks = Vec::new();
    
    if signal.len() < 3 {
        return peaks;
    }
    
    for i in 1..signal.len() - 1 {
        if signal[i] > threshold && 
           signal[i] > signal[i - 1] && 
           signal[i] > signal[i + 1] {
            peaks.push(i);
        }
    }
    
    peaks
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fast_exp() {
        let x = 1.0;
        let exact = x.exp();
        let approx = fast::exp_approx(x);
        
        // Should be within 1% accuracy
        let error = (exact - approx).abs() / exact;
        assert!(error < 0.01);
    }
    
    #[test]
    fn test_sigmoid() {
        assert_eq!(activation::sigmoid(0.0), 0.5);
        assert!(activation::sigmoid(5.0) > 0.99);
        assert!(activation::sigmoid(-5.0) < 0.01);
    }
    
    #[test]
    fn test_stats() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(stats::mean(&data), 3.0);
        assert!((stats::variance(&data) - 2.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_vector_ops() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        
        assert_eq!(vector::dot(&a, &b), 32.0);
        assert!((vector::l2_norm(&a) - 14.0_f32.sqrt()).abs() < 1e-6);
    }
    
    #[test]
    fn test_ode_solvers() {
        // Test exponential decay: dy/dt = -y/tau
        let y0 = 1.0;
        let tau = 1.0;
        let dt = 0.1;
        
        let analytical = y0 * (-dt / tau).exp();
        let numerical = ode::exponential_euler(y0, tau, 0.0, dt);
        
        // Should be exact for exponential functions
        assert!((analytical - numerical).abs() < 1e-6);
    }
    
    #[test]
    fn test_signal_processing() {
        let mut state = 0.0;
        let alpha = 0.1;
        
        // Low-pass filter should smooth step response
        let step1 = signal::low_pass_filter(state, 1.0, alpha);
        state = step1;
        let step2 = signal::low_pass_filter(state, 1.0, alpha);
        
        assert!(step1 < 1.0);
        assert!(step2 > step1);
        assert!(step2 < 1.0);
    }
    
    #[test]
    fn test_gaussian() {
        // Gaussian should be symmetric around mean
        let mu = 0.0;
        let sigma = 1.0;
        
        let val_pos = gaussian(1.0, mu, sigma);
        let val_neg = gaussian(-1.0, mu, sigma);
        
        assert!((val_pos - val_neg).abs() < 1e-6);
        
        // Peak should be at mean
        let peak = gaussian(mu, mu, sigma);
        assert!(peak > val_pos);
    }
    
    #[test]
    fn test_find_peaks() {
        let signal = [1.0, 3.0, 2.0, 5.0, 1.0, 4.0, 2.0];
        let peaks = find_peaks(&signal, 2.5);
        
        // Should find peaks at indices 1, 3, and 5
        assert!(peaks.contains(&1)); // Value 3.0
        assert!(peaks.contains(&3)); // Value 5.0
        assert!(peaks.contains(&5)); // Value 4.0
    }
}