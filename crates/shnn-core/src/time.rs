//! Time handling and temporal processing for neuromorphic systems
//!
//! This module provides precise time representation and temporal operations
//! optimized for spike-based neural computation.

use crate::error::{Result, SHNNError};
use core::fmt;
use core::ops::{Add, Sub, AddAssign, SubAssign};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// High-precision time representation for neuromorphic computation
///
/// Time is represented in nanoseconds to provide sufficient precision
/// for biological time constants while maintaining efficient arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Time(u64);

impl Time {
    /// Create a new time from nanoseconds
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }
    
    /// Create a new time from microseconds
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros * 1_000)
    }
    
    /// Create a new time from milliseconds
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis * 1_000_000)
    }
    
    /// Create a new time from seconds
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }
    
    /// Create a new time from floating-point seconds
    pub fn from_secs_f64(secs: f64) -> Result<Self> {
        if secs < 0.0 || !secs.is_finite() {
            return Err(SHNNError::time_error("Invalid time value"));
        }
        Ok(Self((secs * 1_000_000_000.0) as u64))
    }
    
    /// Zero time constant
    pub const ZERO: Self = Self(0);
    
    /// Maximum representable time
    pub const MAX: Self = Self(u64::MAX);
    
    /// Get time as nanoseconds
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }
    
    /// Get time as microseconds
    pub const fn as_micros(&self) -> u64 {
        self.0 / 1_000
    }
    
    /// Get time as milliseconds
    pub const fn as_millis(&self) -> u64 {
        self.0 / 1_000_000
    }
    
    /// Get time as seconds
    pub const fn as_secs(&self) -> u64 {
        self.0 / 1_000_000_000
    }
    
    /// Get time as floating-point seconds
    pub fn as_secs_f64(&self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }
    
    /// Calculate the duration since another time
    pub fn duration_since(&self, earlier: Time) -> Result<Duration> {
        if self.0 >= earlier.0 {
            Ok(Duration(self.0 - earlier.0))
        } else {
            Err(SHNNError::time_error("Time ordering violation"))
        }
    }
    
    /// Calculate the elapsed time since another time
    pub fn elapsed_since(&self, earlier: Time) -> Duration {
        Duration(self.0.saturating_sub(earlier.0))
    }
    
    /// Check if this time is after another time
    pub const fn is_after(&self, other: Time) -> bool {
        self.0 > other.0
    }
    
    /// Check if this time is before another time
    pub const fn is_before(&self, other: Time) -> bool {
        self.0 < other.0
    }
    
    /// Saturating addition
    pub fn saturating_add(&self, duration: Duration) -> Self {
        Self(self.0.saturating_add(duration.0))
    }
    
    /// Saturating subtraction
    pub fn saturating_sub(&self, duration: Duration) -> Self {
        Self(self.0.saturating_sub(duration.0))
    }
    
    /// Checked addition
    pub fn checked_add(&self, duration: Duration) -> Option<Self> {
        self.0.checked_add(duration.0).map(Self)
    }
    
    /// Checked subtraction
    pub fn checked_sub(&self, duration: Duration) -> Option<Self> {
        self.0.checked_sub(duration.0).map(Self)
    }
}

impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 1_000_000_000 {
            write!(f, "{:.3}s", self.as_secs_f64())
        } else if self.0 >= 1_000_000 {
            write!(f, "{:.3}ms", self.0 as f64 / 1_000_000.0)
        } else if self.0 >= 1_000 {
            write!(f, "{:.3}μs", self.0 as f64 / 1_000.0)
        } else {
            write!(f, "{}ns", self.0)
        }
    }
}

/// Duration between two time points
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Duration(u64);

impl Duration {
    /// Create a new duration from nanoseconds
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }
    
    /// Create a new duration from microseconds
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros * 1_000)
    }
    
    /// Create a new duration from milliseconds
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis * 1_000_000)
    }
    
    /// Create a new duration from seconds
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }
    
    /// Create a new duration from floating-point seconds
    pub fn from_secs_f64(secs: f64) -> Result<Self> {
        if secs < 0.0 || !secs.is_finite() {
            return Err(SHNNError::time_error("Invalid duration value"));
        }
        Ok(Self((secs * 1_000_000_000.0) as u64))
    }
    
    /// Zero duration constant
    pub const ZERO: Self = Self(0);
    
    /// Maximum representable duration
    pub const MAX: Self = Self(u64::MAX);
    
    /// Get duration as nanoseconds
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }
    
    /// Get duration as microseconds
    pub const fn as_micros(&self) -> u64 {
        self.0 / 1_000
    }
    
    /// Get duration as milliseconds
    pub const fn as_millis(&self) -> u64 {
        self.0 / 1_000_000
    }
    
    /// Get duration as seconds
    pub const fn as_secs(&self) -> u64 {
        self.0 / 1_000_000_000
    }
    
    /// Get duration as floating-point seconds
    pub fn as_secs_f64(&self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }
    
    /// Check if duration is zero
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }
    
    /// Saturating multiplication
    pub fn saturating_mul(&self, rhs: u32) -> Self {
        Self(self.0.saturating_mul(rhs as u64))
    }
    
    /// Saturating division
    pub fn saturating_div(&self, rhs: u32) -> Self {
        if rhs == 0 {
            Self::ZERO
        } else {
            Self(self.0 / rhs as u64)
        }
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 1_000_000_000 {
            write!(f, "{:.3}s", self.as_secs_f64())
        } else if self.0 >= 1_000_000 {
            write!(f, "{:.3}ms", self.0 as f64 / 1_000_000.0)
        } else if self.0 >= 1_000 {
            write!(f, "{:.3}μs", self.0 as f64 / 1_000.0)
        } else {
            write!(f, "{}ns", self.0)
        }
    }
}

// Arithmetic operations for Time
impl Add<Duration> for Time {
    type Output = Self;
    
    fn add(self, rhs: Duration) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign<Duration> for Time {
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs.0;
    }
}

impl Sub<Duration> for Time {
    type Output = Self;
    
    fn sub(self, rhs: Duration) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign<Duration> for Time {
    fn sub_assign(&mut self, rhs: Duration) {
        self.0 -= rhs.0;
    }
}

impl Sub<Time> for Time {
    type Output = Duration;
    
    fn sub(self, rhs: Time) -> Self::Output {
        Duration(self.0 - rhs.0)
    }
}

// Arithmetic operations for Duration
impl Add<Duration> for Duration {
    type Output = Self;
    
    fn add(self, rhs: Duration) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign<Duration> for Duration {
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs.0;
    }
}

impl Sub<Duration> for Duration {
    type Output = Self;
    
    fn sub(self, rhs: Duration) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign<Duration> for Duration {
    fn sub_assign(&mut self, rhs: Duration) {
        self.0 -= rhs.0;
    }
}

/// Discrete time step for simulation
pub type TimeStep = u64;

/// Helper methods for TimeStep (test convenience)
#[cfg(test)]
impl TimeStep {
    /// Create TimeStep from milliseconds (for testing)
    pub fn from_ms(ms: f64) -> Self {
        (ms * 1000.0) as u64 // Convert to microseconds
    }
    
    /// Create TimeStep from microseconds
    pub fn from_micros(micros: f64) -> Self {
        micros as u64
    }
    
    /// Create TimeStep from seconds
    pub fn from_secs(secs: f64) -> Self {
        (secs * 1_000_000.0) as u64 // Convert to microseconds
    }
}

/// Time window for temporal operations
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeWindow {
    /// Start time of the window
    pub start: Time,
    /// End time of the window
    pub end: Time,
}

impl TimeWindow {
    /// Create a new time window
    pub fn new(start: Time, end: Time) -> Result<Self> {
        if start > end {
            return Err(SHNNError::time_error("Invalid time window: start > end"));
        }
        Ok(Self { start, end })
    }
    
    /// Create a time window with a specific duration
    pub fn with_duration(start: Time, duration: Duration) -> Self {
        Self {
            start,
            end: start + duration,
        }
    }
    
    /// Get the duration of the window
    pub fn duration(&self) -> Duration {
        self.end - self.start
    }
    
    /// Check if a time point is within the window
    pub fn contains(&self, time: Time) -> bool {
        time >= self.start && time <= self.end
    }
    
    /// Check if this window overlaps with another
    pub fn overlaps(&self, other: &TimeWindow) -> bool {
        self.start <= other.end && self.end >= other.start
    }
    
    /// Get the intersection with another window
    pub fn intersection(&self, other: &TimeWindow) -> Option<TimeWindow> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        
        if start <= end {
            Some(TimeWindow { start, end })
        } else {
            None
        }
    }
}

/// Clock trait for time sources
pub trait Clock {
    /// Get the current time
    fn now(&self) -> Time;
    
    /// Get elapsed time since a reference point
    fn elapsed(&self, since: Time) -> Duration {
        self.now().elapsed_since(since)
    }
}

/// System clock implementation
#[cfg(feature = "std")]
pub struct SystemClock;

#[cfg(feature = "std")]
impl Clock for SystemClock {
    fn now(&self) -> Time {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        
        Time::from_nanos(
            duration.as_secs() * 1_000_000_000 + duration.subsec_nanos() as u64
        )
    }
}

/// Mock clock for testing
pub struct MockClock {
    current_time: Time,
}

impl MockClock {
    /// Create a new mock clock
    pub fn new(initial_time: Time) -> Self {
        Self {
            current_time: initial_time,
        }
    }
    
    /// Advance the clock by a duration
    pub fn advance(&mut self, duration: Duration) {
        self.current_time += duration;
    }
    
    /// Set the clock to a specific time
    pub fn set_time(&mut self, time: Time) {
        self.current_time = time;
    }
}

impl Clock for MockClock {
    fn now(&self) -> Time {
        self.current_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_time_creation() {
        let t1 = Time::from_secs(1);
        let t2 = Time::from_millis(1000);
        let t3 = Time::from_micros(1_000_000);
        let t4 = Time::from_nanos(1_000_000_000);
        
        assert_eq!(t1, t2);
        assert_eq!(t2, t3);
        assert_eq!(t3, t4);
    }
    
    #[test]
    fn test_time_arithmetic() {
        let t1 = Time::from_secs(1);
        let d1 = Duration::from_secs(1);
        let t2 = t1 + d1;
        
        assert_eq!(t2.as_secs(), 2);
        assert_eq!(t2 - t1, d1);
    }
    
    #[test]
    fn test_duration_display() {
        assert_eq!(Duration::from_nanos(500).to_string(), "500ns");
        assert_eq!(Duration::from_micros(500).to_string(), "500.000μs");
        assert_eq!(Duration::from_millis(500).to_string(), "500.000ms");
        assert_eq!(Duration::from_secs(1).to_string(), "1.000s");
    }
    
    #[test]
    fn test_time_window() {
        let start = Time::from_secs(1);
        let end = Time::from_secs(3);
        let window = TimeWindow::new(start, end).unwrap();
        
        assert_eq!(window.duration(), Duration::from_secs(2));
        assert!(window.contains(Time::from_secs(2)));
        assert!(!window.contains(Time::from_secs(4)));
    }
    
    #[test]
    fn test_mock_clock() {
        let mut clock = MockClock::new(Time::from_secs(0));
        assert_eq!(clock.now(), Time::from_secs(0));
        
        clock.advance(Duration::from_secs(1));
        assert_eq!(clock.now(), Time::from_secs(1));
        
        clock.set_time(Time::from_secs(5));
        assert_eq!(clock.now(), Time::from_secs(5));
    }
}