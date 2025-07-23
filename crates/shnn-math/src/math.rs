//! Custom math functions for no-std compatibility
//! 
//! Provides zero-dependency implementations of common mathematical functions
//! optimized for neuromorphic computations with controlled accuracy.

use crate::Float;

/// Extension trait providing custom math functions for f32
pub trait FloatMath {
    /// Square root using Newton-Raphson method
    fn sqrt(self) -> Self;
    
    /// Exponential function using Taylor series
    fn exp(self) -> Self;
    
    /// Natural logarithm using polynomial approximation
    fn ln(self) -> Self;
    
    /// Hyperbolic tangent using rational approximation
    fn tanh(self) -> Self;
    
    /// Cosine using Taylor series
    fn cos(self) -> Self;
    
    /// Sine using Taylor series
    fn sin(self) -> Self;
    
    /// Floor function
    fn floor(self) -> Self;
    
    /// Ceiling function
    fn ceil(self) -> Self;
}

// Standalone approximation functions for compatibility
pub fn exp_approx(x: f32) -> f32 {
    x.exp()
}

pub fn ln_approx(x: f32) -> f32 {
    x.ln()
}

pub fn sqrt_approx(x: f32) -> f32 {
    x.sqrt()
}

pub fn sin_approx(x: f32) -> f32 {
    x.sin()
}

pub fn cos_approx(x: f32) -> f32 {
    x.cos()
}

impl FloatMath for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        if self < 0.0 {
            return Float::NAN;
        }
        if self == 0.0 || self == 1.0 {
            return self;
        }
        
        // Newton-Raphson method: x_{n+1} = (x_n + a/x_n) / 2
        let mut x = self * 0.5; // Initial guess
        for _ in 0..10 { // Fixed iterations for deterministic performance
            x = 0.5 * (x + self / x);
        }
        x
    }
    
    #[inline]
    fn exp(self) -> Self {
        if self > 88.0 {
            return Float::INFINITY;
        }
        if self < -88.0 {
            return 0.0;
        }
        
        // Taylor series: e^x = 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5!
        let mut result = 1.0;
        let mut term = 1.0;
        
        for i in 1..20 { // Sufficient precision for f32
            term *= self / i as f32;
            result += term;
            
            if term.abs() < 1e-7 {
                break;
            }
        }
        result
    }
    
    #[inline]
    fn ln(self) -> Self {
        if self <= 0.0 {
            return Float::NEG_INFINITY;
        }
        if self == 1.0 {
            return 0.0;
        }
        
        // For x near 1, use Taylor series: ln(1+u) ≈ u - u²/2 + u³/3 - u⁴/4
        let u = self - 1.0;
        if u.abs() < 0.5 {
            let u2 = u * u;
            let u3 = u2 * u;
            let u4 = u3 * u;
            let u5 = u4 * u;
            return u - u2 * 0.5 + u3 / 3.0 - u4 * 0.25 + u5 * 0.2;
        }
        
        // For other values, use property ln(x) = 2 * ln(sqrt(x))
        // and reduce to range near 1
        let mut x = self;
        let mut exp_adjust = 0.0;
        
        // Normalize to [0.5, 2) range
        while x >= 2.0 {
            x *= 0.5;
            exp_adjust += core::f32::consts::LN_2;
        }
        while x < 0.5 {
            x *= 2.0;
            exp_adjust -= core::f32::consts::LN_2;
        }
        
        // Now x is in [0.5, 2), apply Taylor series around 1
        let u = x - 1.0;
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u3 * u;
        let u5 = u4 * u;
        
        exp_adjust + u - u2 * 0.5 + u3 / 3.0 - u4 * 0.25 + u5 * 0.2
    }
    
    #[inline]
    fn tanh(self) -> Self {
        if self > 10.0 {
            return 1.0;
        }
        if self < -10.0 {
            return -1.0;
        }
        
        // Use identity: tanh(x) = (e^2x - 1) / (e^2x + 1)
        let exp_2x = (2.0 * self).exp();
        (exp_2x - 1.0) / (exp_2x + 1.0)
    }
    
    #[inline]
    fn cos(self) -> Self {
        // Reduce to [0, 2π] range
        let x = self - (self / (2.0 * core::f32::consts::PI)).floor() * 2.0 * core::f32::consts::PI;
        
        // Taylor series: cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
        let x2 = x * x;
        let mut result = 1.0;
        let mut term = 1.0;
        let mut sign = -1.0;
        
        for i in 1..10 {
            term *= x2 / ((2 * i - 1) * (2 * i)) as f32;
            result += sign * term;
            sign *= -1.0;
        }
        result
    }
    
    #[inline]
    fn sin(self) -> Self {
        // Use identity: sin(x) = cos(π/2 - x)
        (core::f32::consts::FRAC_PI_2 - self).cos()
    }
    
    #[inline]
    fn floor(self) -> Self {
        if self >= 0.0 {
            self as i32 as f32
        } else {
            let truncated = self as i32 as f32;
            if truncated == self {
                truncated
            } else {
                truncated - 1.0
            }
        }
    }
    
    #[inline]
    fn ceil(self) -> Self {
        if self >= 0.0 {
            let truncated = self as i32 as f32;
            if truncated == self {
                truncated
            } else {
                truncated + 1.0
            }
        } else {
            self as i32 as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt() {
        assert!((4.0f32.sqrt() - 2.0).abs() < 1e-6);
        assert!((9.0f32.sqrt() - 3.0).abs() < 1e-6);
        assert!((0.25f32.sqrt() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_exp() {
        assert!((1.0f32.exp() - core::f32::consts::E).abs() < 1e-5);
        assert!((0.0f32.exp() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ln() {
        assert!((core::f32::consts::E.ln() - 1.0).abs() < 1e-5);
        assert!((1.0f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_tanh() {
        assert!((0.0f32.tanh()).abs() < 1e-6);
        assert!((1.0f32.tanh() - 0.76159).abs() < 1e-4);
    }

    #[test]
    fn test_trig() {
        assert!((0.0f32.cos() - 1.0).abs() < 1e-5);
        assert!((0.0f32.sin()).abs() < 1e-5);
        assert!((core::f32::consts::FRAC_PI_2.cos()).abs() < 1e-4);
        assert!((core::f32::consts::FRAC_PI_2.sin() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_floor_ceil() {
        assert_eq!(3.7f32.floor(), 3.0);
        assert_eq!((-3.7f32).floor(), -4.0);
        assert_eq!(3.2f32.ceil(), 4.0);
        assert_eq!((-3.2f32).ceil(), -3.0);
    }
}