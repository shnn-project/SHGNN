//! Fast approximate mathematical functions for real-time neuromorphic computations
//! 
//! Provides fast approximations of common mathematical functions with controlled
//! accuracy trade-offs for performance-critical applications.

use crate::{Float, constants::EPSILON, math::FloatMath};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::vec::Vec;

/// Fast approximate exponential function
/// Uses polynomial approximation for speed over accuracy
#[inline]
pub fn exp_approx(x: Float) -> Float {
    if x > 10.0 {
        return Float::INFINITY;
    }
    if x < -10.0 {
        return 0.0;
    }

    // Polynomial approximation: e^x ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    
    1.0 + x + x2 * 0.5 + x3 / 6.0 + x4 / 24.0
}

/// Fast approximate natural logarithm
/// Uses rational approximation for |x - 1| < 1
#[inline]
pub fn ln_approx(x: Float) -> Float {
    if x <= 0.0 {
        return Float::NEG_INFINITY;
    }
    if x == 1.0 {
        return 0.0;
    }

    // For x near 1, use Taylor series: ln(1+u) ≈ u - u²/2 + u³/3 - u⁴/4
    let u = x - 1.0;
    if u.abs() < 1.0 {
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u3 * u;
        return u - u2 * 0.5 + u3 / 3.0 - u4 * 0.25;
    }

    // For larger values, use change of base and recursion
    x.ln() // Fallback to standard implementation
}

/// Fast approximate sine function using polynomial
#[inline]
pub fn sin_approx(x: Float) -> Float {
    // Normalize to [-π, π]
    let mut x = x % (2.0 * core::f32::consts::PI);
    if x > core::f32::consts::PI {
        x -= 2.0 * core::f32::consts::PI;
    }

    // Polynomial approximation on [-π, π]
    // sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;

    x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0
}

/// Fast approximate cosine function
#[inline]
pub fn cos_approx(x: Float) -> Float {
    // cos(x) = sin(x + π/2)
    sin_approx(x + core::f32::consts::FRAC_PI_2)
}

/// Fast approximate tangent function
#[inline]
pub fn tan_approx(x: Float) -> Float {
    let s = sin_approx(x);
    let c = cos_approx(x);
    
    if c.abs() < EPSILON {
        if s > 0.0 { Float::INFINITY } else { Float::NEG_INFINITY }
    } else {
        s / c
    }
}

/// Fast approximate square root using Newton's method
#[inline]
pub fn sqrt_approx(x: Float) -> Float {
    if x < 0.0 {
        return Float::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }

    // Initial guess using bit manipulation
    let mut guess = x * 0.5;
    
    // One iteration of Newton's method: x_{n+1} = (x_n + a/x_n) / 2
    guess = (guess + x / guess) * 0.5;
    
    // Second iteration for better accuracy
    guess = (guess + x / guess) * 0.5;
    
    guess
}

/// Fast approximate inverse square root (Quake algorithm inspired)
#[inline]
pub fn inv_sqrt_approx(x: Float) -> Float {
    if x <= 0.0 {
        return Float::INFINITY;
    }

    1.0 / sqrt_approx(x)
}

/// Fast approximate power function x^y
#[inline]
pub fn pow_approx(x: Float, y: Float) -> Float {
    if x <= 0.0 {
        return 0.0;
    }
    if y == 0.0 {
        return 1.0;
    }
    if y == 1.0 {
        return x;
    }

    // Use exp(y * ln(x))
    exp_approx(y * ln_approx(x))
}

/// Fast sigmoid approximation using rational function
#[inline]
pub fn sigmoid_fast(x: Float) -> Float {
    // Fast approximation: σ(x) ≈ x / (1 + |x|)
    if x >= 0.0 {
        x / (1.0 + x)
    } else {
        x / (1.0 - x)
    }
}

/// Fast tanh approximation using polynomial
#[inline]
pub fn tanh_fast(x: Float) -> Float {
    // Clamp to avoid overflow
    let x = x.clamp(-3.0, 3.0);
    
    // Polynomial approximation
    let x2 = x * x;
    let a = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let b = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    
    a / b
}

/// Fast arctangent approximation
#[inline]
pub fn atan_approx(x: Float) -> Float {
    // Rational approximation for atan
    let x_abs = x.abs();
    
    if x_abs <= 1.0 {
        // For |x| <= 1: atan(x) ≈ x / (1 + 0.28 * x²)
        let result = x / (1.0 + 0.28 * x * x);
        result
    } else {
        // For |x| > 1: atan(x) = π/2 - atan(1/x)
        let sign = if x > 0.0 { 1.0 } else { -1.0 };
        sign * core::f32::consts::FRAC_PI_2 - atan_approx(1.0 / x)
    }
}

/// Fast approximate Gaussian function
#[inline]
pub fn gaussian_approx(x: Float, mean: Float, std: Float) -> Float {
    let z = (x - mean) / std;
    exp_approx(-0.5 * z * z) / (std * (2.0 * core::f32::consts::PI).sqrt())
}

/// Fast error function approximation (erf)
#[inline]
pub fn erf_approx(x: Float) -> Float {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp_approx(-x * x);

    sign * y
}

/// Fast complementary error function approximation (erfc)
#[inline]
pub fn erfc_approx(x: Float) -> Float {
    1.0 - erf_approx(x)
}

/// Safe division with epsilon check
#[inline]
pub fn safe_divide(numerator: Float, denominator: Float) -> Float {
    if denominator.abs() < EPSILON {
        if numerator > 0.0 {
            Float::INFINITY
        } else if numerator < 0.0 {
            Float::NEG_INFINITY
        } else {
            Float::NAN
        }
    } else {
        numerator / denominator
    }
}

/// Linear interpolation
#[inline]
pub fn lerp(a: Float, b: Float, t: Float) -> Float {
    a + t * (b - a)
}

/// Smooth interpolation (smoothstep)
#[inline]
pub fn smoothstep(edge0: Float, edge1: Float, x: Float) -> Float {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Remap value from one range to another
#[inline]
pub fn remap(value: Float, from_min: Float, from_max: Float, to_min: Float, to_max: Float) -> Float {
    let t = (value - from_min) / (from_max - from_min);
    lerp(to_min, to_max, t)
}

/// Fast modulo operation for positive numbers
#[inline]
pub fn fast_mod(x: Float, y: Float) -> Float {
    x - y * (x / y).floor()
}

/// Neuromorphic-specific approximations
pub mod neuromorphic {
    use super::*;

    /// Fast membrane potential decay approximation
    #[inline]
    pub fn membrane_decay(potential: Float, tau: Float, dt: Float) -> Float {
        // Exponential decay: V(t) = V₀ * exp(-t/τ)
        // Approximation: V(t) ≈ V₀ * (1 - dt/τ) for small dt
        let decay_factor = 1.0 - dt / tau;
        potential * decay_factor.max(0.0)
    }

    /// Fast synaptic conductance profile
    #[inline]
    pub fn synaptic_conductance(t: Float, tau_rise: Float, tau_decay: Float, peak: Float) -> Float {
        if t <= 0.0 {
            return 0.0;
        }

        // Double exponential: g(t) = A * (exp(-t/τ_decay) - exp(-t/τ_rise))
        let rise = exp_approx(-t / tau_rise);
        let decay = exp_approx(-t / tau_decay);
        peak * (decay - rise)
    }

    /// Fast spike probability from membrane potential
    #[inline]
    pub fn spike_probability(membrane: Float, threshold: Float, noise: Float) -> Float {
        let z = (membrane - threshold) / noise;
        // Use fast sigmoid for probability
        sigmoid_fast(z)
    }

    /// Fast adaptation current update
    #[inline]
    pub fn adaptation_update(current: Float, spike: bool, tau: Float, dt: Float, increment: Float) -> Float {
        let decay = membrane_decay(current, tau, dt);
        if spike {
            decay + increment
        } else {
            decay
        }
    }

    /// Fast STDP weight update
    #[inline]
    pub fn stdp_update(
        weight: Float,
        dt: Float,
        a_plus: Float,
        a_minus: Float,
        tau_plus: Float,
        tau_minus: Float,
    ) -> Float {
        if dt > 0.0 {
            // Post before pre (potentiation)
            weight + a_plus * exp_approx(-dt / tau_plus)
        } else {
            // Pre before post (depression)
            weight + a_minus * exp_approx(dt / tau_minus)
        }
    }

    /// Fast frequency adaptation
    #[inline]
    pub fn frequency_adaptation(frequency: Float, target: Float, alpha: Float) -> Float {
        // Exponential moving average towards target
        frequency * (1.0 - alpha) + target * alpha
    }
}

/// Lookup table for very fast approximations
pub struct LookupTable {
    values: Vec<Float>,
    min_x: Float,
    max_x: Float,
    scale: Float,
}

impl LookupTable {
    /// Create lookup table for function f over range [min_x, max_x]
    pub fn new<F>(f: F, min_x: Float, max_x: Float, resolution: usize) -> Self 
    where
        F: Fn(Float) -> Float,
    {
        let mut values = Vec::with_capacity(resolution);
        let step = (max_x - min_x) / (resolution - 1) as Float;
        
        for i in 0..resolution {
            let x = min_x + i as Float * step;
            values.push(f(x));
        }
        
        let scale = (resolution - 1) as Float / (max_x - min_x);
        
        Self {
            values,
            min_x,
            max_x,
            scale,
        }
    }

    /// Fast lookup with linear interpolation
    pub fn lookup(&self, x: Float) -> Float {
        if x <= self.min_x {
            return self.values[0];
        }
        if x >= self.max_x {
            return self.values[self.values.len() - 1];
        }

        let float_index = (x - self.min_x) * self.scale;
        let index = float_index.floor() as usize;
        let fraction = float_index - index as Float;

        if index + 1 >= self.values.len() {
            return self.values[index];
        }

        lerp(self.values[index], self.values[index + 1], fraction)
    }

    /// Create exponential lookup table
    pub fn exp_table(min_x: Float, max_x: Float, resolution: usize) -> Self {
        Self::new(exp_approx, min_x, max_x, resolution)
    }

    /// Create sigmoid lookup table
    pub fn sigmoid_table(min_x: Float, max_x: Float, resolution: usize) -> Self {
        Self::new(sigmoid_fast, min_x, max_x, resolution)
    }

    /// Create tanh lookup table
    pub fn tanh_table(min_x: Float, max_x: Float, resolution: usize) -> Self {
        Self::new(tanh_fast, min_x, max_x, resolution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_approximation() {
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let exact = x.exp();
            let approx = exp_approx(x);
            let error = (exact - approx).abs() / exact;
            assert!(error < 0.1, "exp({}) error too large: {}", x, error);
        }
    }

    #[test]
    fn test_trigonometric_approximations() {
        for x in [-1.0, 0.0, 1.0, core::f32::consts::FRAC_PI_2] {
            let sin_exact = x.sin();
            let sin_approx = sin_approx(x);
            let sin_error = (sin_exact - sin_approx).abs();
            assert!(sin_error < 0.05, "sin({}) error: {}", x, sin_error);

            let cos_exact = x.cos();
            let cos_approx = cos_approx(x);
            let cos_error = (cos_exact - cos_approx).abs();
            assert!(cos_error < 0.05, "cos({}) error: {}", x, cos_error);
        }
    }

    #[test]
    fn test_sqrt_approximation() {
        for x in [1.0, 4.0, 9.0, 16.0, 25.0] {
            let exact = x.sqrt();
            let approx = sqrt_approx(x);
            let error = (exact - approx).abs() / exact;
            assert!(error < 0.01, "sqrt({}) error: {}", x, error);
        }
    }

    #[test]
    fn test_sigmoid_fast() {
        for x in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            let exact = 1.0 / (1.0 + (-x).exp());
            let fast = sigmoid_fast(x);
            // Fast sigmoid is an approximation, so allow larger error
            let error = (exact - fast).abs();
            assert!(error < 0.2, "sigmoid_fast({}) error: {}", x, error);
        }
    }

    #[test]
    fn test_safe_divide() {
        assert_eq!(safe_divide(1.0, 2.0), 0.5);
        assert!(safe_divide(1.0, 0.0).is_infinite());
        assert!(safe_divide(0.0, 0.0).is_nan());
    }

    #[test]
    fn test_neuromorphic_functions() {
        // Test membrane decay
        let initial = 10.0;
        let tau = 20.0;
        let dt = 1.0;
        let decayed = neuromorphic::membrane_decay(initial, tau, dt);
        assert!(decayed < initial);
        assert!(decayed > 0.0);

        // Test spike probability
        let prob = neuromorphic::spike_probability(-50.0, -55.0, 5.0);
        assert!(prob > 0.5); // Above threshold should have high probability
        
        let prob2 = neuromorphic::spike_probability(-60.0, -55.0, 5.0);
        assert!(prob2 < 0.5); // Below threshold should have low probability
    }

    #[test]
    fn test_lookup_table() {
        let table = LookupTable::exp_table(-5.0, 5.0, 1000);
        
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let exact = x.exp();
            let lookup = table.lookup(x);
            let error = (exact - lookup).abs() / exact;
            assert!(error < 0.01, "lookup exp({}) error: {}", x, error);
        }
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);

        let smooth = smoothstep(0.0, 1.0, 0.5);
        assert!(smooth > 0.4 && smooth < 0.6);
    }

    #[test]
    fn test_remap() {
        // Remap from [0, 100] to [0, 1]
        assert_eq!(remap(50.0, 0.0, 100.0, 0.0, 1.0), 0.5);
        assert_eq!(remap(0.0, 0.0, 100.0, 0.0, 1.0), 0.0);
        assert_eq!(remap(100.0, 0.0, 100.0, 0.0, 1.0), 1.0);
    }
}