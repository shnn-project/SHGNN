//! High-precision timer for neuromorphic spike timing
//! 
//! Provides deterministic timing capabilities essential for accurate neural simulation.

use core::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::time::Instant;

use crate::{SpikeTime, LockFreeQueue};

/// Fast square root approximation using Newton-Raphson method
fn sqrt_approx(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    
    // Initial guess using bit manipulation
    let mut guess = f64::from_bits((x.to_bits() >> 1) + (1023_u64 << 51));
    
    // Newton-Raphson iterations (3 iterations for good precision)
    for _ in 0..3 {
        guess = 0.5 * (guess + x / guess);
    }
    
    guess
}

/// Timer alias for compatibility
pub type Timer = PrecisionTimer;

/// High-precision timer for spike timing
pub struct PrecisionTimer {
    /// Start time reference
    #[cfg(feature = "std")]
    start_time: Instant,
    #[cfg(not(feature = "std"))]
    start_time: u64,
    /// Timer resolution
    resolution: Duration,
    /// Timer frequency in Hz
    frequency: u64,
    /// Current tick counter
    current_tick: AtomicU64,
    /// Scheduled timeouts
    timeouts: TimerWheel,
}

impl PrecisionTimer {
    /// Create new precision timer with specified resolution
    pub fn new(resolution: Duration) -> Self {
        let frequency = 1_000_000_000 / resolution.as_nanos() as u64;
        
        Self {
            #[cfg(feature = "std")]
            start_time: Instant::now(),
            #[cfg(not(feature = "std"))]
            start_time: 0,
            resolution,
            frequency,
            current_tick: AtomicU64::new(0),
            timeouts: TimerWheel::new(resolution),
        }
    }

    /// Get current time with high precision
    pub fn now(&self) -> SpikeTime {
        #[cfg(feature = "std")]
        {
            let elapsed = self.start_time.elapsed();
            SpikeTime::from_nanos(elapsed.as_nanos() as u64)
        }
        #[cfg(not(feature = "std"))]
        {
            // In no-std environment, we'd need a platform-specific timer
            // For now, use tick counter as approximation
            let ticks = self.current_tick.load(Ordering::Relaxed);
            SpikeTime::from_nanos(ticks * self.resolution.as_nanos() as u64)
        }
    }

    /// Get timer resolution
    pub fn resolution(&self) -> Duration {
        self.resolution
    }

    /// Get timer frequency
    pub fn frequency(&self) -> u64 {
        self.frequency
    }

    /// Schedule timeout
    pub fn schedule_timeout(&self, delay: SpikeTime, callback: TimeoutCallback) -> TimeoutId {
        let target_time = self.now() + delay;
        self.timeouts.schedule(target_time, callback)
    }

    /// Cancel scheduled timeout
    pub fn cancel_timeout(&self, timeout_id: TimeoutId) -> bool {
        self.timeouts.cancel(timeout_id)
    }

    /// Process expired timeouts
    pub fn process_timeouts(&self) -> Vec<TimeoutCallback> {
        let current_time = self.now();
        self.timeouts.process_expired(current_time)
    }

    /// Sleep until specified time (async)
    pub async fn sleep_until(&self, target: SpikeTime) {
        SleepUntilFuture::new(target, self).await
    }

    /// Sleep for specified duration (async)
    pub async fn sleep(&self, duration: SpikeTime) {
        let target = self.now() + duration;
        self.sleep_until(target).await
    }

    /// Advance timer tick (for no-std environments)
    #[cfg(not(feature = "std"))]
    pub fn tick(&self) {
        self.current_tick.fetch_add(1, Ordering::Relaxed);
    }
}

/// Hierarchical timer wheel for efficient timeout scheduling
pub struct TimerWheel {
    /// Timer wheels for different time scales
    wheels: [Wheel; 4],
    /// Current time base
    time_base: AtomicU64,
    /// Next timeout ID
    next_id: AtomicU64,
    /// Resolution
    resolution: Duration,
}

impl TimerWheel {
    /// Create new timer wheel
    pub fn new(resolution: Duration) -> Self {
        Self {
            wheels: [
                Wheel::new(256, 1),      // Microseconds: 256μs range
                Wheel::new(64, 256),     // Milliseconds: 16ms range  
                Wheel::new(64, 16384),   // Seconds: 1s range
                Wheel::new(64, 1048576), // Minutes: 64min range
            ],
            time_base: AtomicU64::new(0),
            next_id: AtomicU64::new(1),
            resolution,
        }
    }

    /// Schedule timeout
    pub fn schedule(&self, target_time: SpikeTime, callback: TimeoutCallback) -> TimeoutId {
        let timeout_id = TimeoutId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let timeout = Timeout {
            id: timeout_id,
            target_time,
            callback,
        };

        // Select appropriate wheel based on delay
        let current_time = self.time_base.load(Ordering::Relaxed);
        let delay_ticks = target_time.as_nanos().saturating_sub(current_time) / self.resolution.as_nanos() as u64;

        let wheel_index = if delay_ticks < 256 {
            0
        } else if delay_ticks < 16384 {
            1
        } else if delay_ticks < 1048576 {
            2
        } else {
            3
        };

        self.wheels[wheel_index].schedule(timeout, delay_ticks);
        timeout_id
    }

    /// Cancel timeout
    pub fn cancel(&self, timeout_id: TimeoutId) -> bool {
        // Search all wheels for the timeout
        for wheel in &self.wheels {
            if wheel.cancel(timeout_id) {
                return true;
            }
        }
        false
    }

    /// Process expired timeouts
    pub fn process_expired(&self, current_time: SpikeTime) -> Vec<TimeoutCallback> {
        let mut expired = Vec::new();
        let current_ticks = current_time.as_nanos() / self.resolution.as_nanos() as u64;
        
        self.time_base.store(current_ticks, Ordering::Relaxed);

        // Check each wheel for expired timeouts
        for wheel in &self.wheels {
            wheel.collect_expired(current_time, &mut expired);
        }

        expired
    }
}

/// Individual timer wheel
struct Wheel {
    /// Timeout buckets
    buckets: Vec<LockFreeQueue<Timeout>>,
    /// Current position
    current_pos: AtomicU64,
    /// Tick granularity
    granularity: u64,
}

impl Wheel {
    /// Create new wheel
    fn new(bucket_count: usize, granularity: u64) -> Self {
        let buckets = (0..bucket_count)
            .map(|_| LockFreeQueue::with_capacity(64))
            .collect();

        Self {
            buckets,
            current_pos: AtomicU64::new(0),
            granularity,
        }
    }

    /// Schedule timeout in this wheel
    fn schedule(&self, timeout: Timeout, delay_ticks: u64) {
        let bucket_index = (delay_ticks / self.granularity) as usize % self.buckets.len();
        let _ = self.buckets[bucket_index].try_push(timeout);
    }

    /// Cancel timeout in this wheel
    fn cancel(&self, _timeout_id: TimeoutId) -> bool {
        // For simplicity, we don't implement cancellation in this example
        // A real implementation would need to search buckets or maintain a separate index
        false
    }

    /// Collect expired timeouts
    fn collect_expired(&self, current_time: SpikeTime, expired: &mut Vec<TimeoutCallback>) {
        let current_pos = self.current_pos.load(Ordering::Relaxed);
        
        // Check current bucket
        let bucket_index = (current_pos / self.granularity) as usize % self.buckets.len();
        let bucket = &self.buckets[bucket_index];

        let mut remaining = Vec::new();
        
        while let Some(timeout) = bucket.try_pop() {
            if timeout.target_time <= current_time {
                expired.push(timeout.callback);
            } else {
                remaining.push(timeout);
            }
        }

        // Put back non-expired timeouts
        for timeout in remaining {
            let _ = bucket.try_push(timeout);
        }
    }
}

/// Timeout entry
struct Timeout {
    id: TimeoutId,
    target_time: SpikeTime,
    callback: TimeoutCallback,
}

/// Unique timeout identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeoutId(u64);

/// Timeout callback function
pub type TimeoutCallback = fn();

/// Future for sleeping until a specific time
struct SleepUntilFuture<'a> {
    target_time: SpikeTime,
    timer: &'a PrecisionTimer,
}

impl<'a> SleepUntilFuture<'a> {
    fn new(target_time: SpikeTime, timer: &'a PrecisionTimer) -> Self {
        Self { target_time, timer }
    }
}

impl<'a> core::future::Future for SleepUntilFuture<'a> {
    type Output = ();

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        let current_time = self.timer.now();
        
        if current_time >= self.target_time {
            core::task::Poll::Ready(())
        } else {
            // In a real implementation, we'd register the waker with the timer
            // For now, just wake immediately to re-check
            cx.waker().wake_by_ref();
            core::task::Poll::Pending
        }
    }
}

/// Timing utilities for spike processing
pub struct SpikeTimer {
    /// Timer instance
    timer: PrecisionTimer,
    /// Spike timing history
    spike_times: LockFreeQueue<SpikeTime>,
    /// Inter-spike interval statistics
    isi_stats: ISIStats,
}

impl SpikeTimer {
    /// Create new spike timer
    pub fn new(resolution: Duration) -> Self {
        Self {
            timer: PrecisionTimer::new(resolution),
            spike_times: LockFreeQueue::with_capacity(1024),
            isi_stats: ISIStats::new(),
        }
    }

    /// Record spike time
    pub fn record_spike(&self, spike_time: SpikeTime) {
        if let Some(last_spike) = self.spike_times.try_pop() {
            let isi = spike_time - last_spike;
            self.isi_stats.record_interval(isi);
            
            // Put back the last spike time
            let _ = self.spike_times.try_push(spike_time);
        } else {
            let _ = self.spike_times.try_push(spike_time);
        }
    }

    /// Get current time
    pub fn now(&self) -> SpikeTime {
        self.timer.now()
    }

    /// Get inter-spike interval statistics
    pub fn isi_stats(&self) -> &ISIStats {
        &self.isi_stats
    }

    /// Check if spike timing is regular
    pub fn is_regular_firing(&self, threshold: f64) -> bool {
        self.isi_stats.coefficient_of_variation() < threshold
    }
}

/// Inter-spike interval statistics
pub struct ISIStats {
    /// Number of intervals recorded
    count: AtomicU64,
    /// Sum of intervals (for mean calculation)
    sum: AtomicU64,
    /// Sum of squared intervals (for variance calculation)
    sum_squared: AtomicU64,
    /// Minimum interval observed
    min_interval: AtomicU64,
    /// Maximum interval observed
    max_interval: AtomicU64,
}

impl ISIStats {
    /// Create new ISI statistics tracker
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            sum_squared: AtomicU64::new(0),
            min_interval: AtomicU64::new(u64::MAX),
            max_interval: AtomicU64::new(0),
        }
    }

    /// Record new inter-spike interval
    pub fn record_interval(&self, interval: SpikeTime) {
        let interval_ns = interval.as_nanos();
        
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(interval_ns, Ordering::Relaxed);
        self.sum_squared.fetch_add(interval_ns * interval_ns, Ordering::Relaxed);
        
        // Update min
        let mut current_min = self.min_interval.load(Ordering::Relaxed);
        while interval_ns < current_min {
            match self.min_interval.compare_exchange_weak(
                current_min,
                interval_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }
        
        // Update max
        let mut current_max = self.max_interval.load(Ordering::Relaxed);
        while interval_ns > current_max {
            match self.max_interval.compare_exchange_weak(
                current_max,
                interval_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    /// Get mean inter-spike interval
    pub fn mean_interval(&self) -> Option<SpikeTime> {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            None
        } else {
            let sum = self.sum.load(Ordering::Relaxed);
            Some(SpikeTime::from_nanos(sum / count))
        }
    }

    /// Get coefficient of variation (CV)
    pub fn coefficient_of_variation(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count < 2 {
            return f64::NAN;
        }

        let sum = self.sum.load(Ordering::Relaxed) as f64;
        let sum_squared = self.sum_squared.load(Ordering::Relaxed) as f64;
        let n = count as f64;

        let mean = sum / n;
        let variance = (sum_squared - (sum * sum / n)) / (n - 1.0);
        let std_dev = sqrt_approx(variance);

        if mean == 0.0 {
            f64::INFINITY
        } else {
            std_dev / mean
        }
    }

    /// Get minimum interval
    pub fn min_interval(&self) -> Option<SpikeTime> {
        let min = self.min_interval.load(Ordering::Relaxed);
        if min == u64::MAX {
            None
        } else {
            Some(SpikeTime::from_nanos(min))
        }
    }

    /// Get maximum interval
    pub fn max_interval(&self) -> Option<SpikeTime> {
        let max = self.max_interval.load(Ordering::Relaxed);
        if max == 0 {
            None
        } else {
            Some(SpikeTime::from_nanos(max))
        }
    }

    /// Reset statistics
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.sum_squared.store(0, Ordering::Relaxed);
        self.min_interval.store(u64::MAX, Ordering::Relaxed);
        self.max_interval.store(0, Ordering::Relaxed);
    }
}

impl Default for ISIStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_timer() {
        let timer = PrecisionTimer::new(Duration::from_micros(1));
        assert_eq!(timer.resolution(), Duration::from_micros(1));
        assert_eq!(timer.frequency(), 1_000_000);
        
        let time1 = timer.now();
        let time2 = timer.now();
        assert!(time2 >= time1);
    }

    #[test]
    fn test_spike_timer() {
        let spike_timer = SpikeTimer::new(Duration::from_micros(1));
        
        let time1 = SpikeTime::from_millis(10);
        let time2 = SpikeTime::from_millis(20);
        
        spike_timer.record_spike(time1);
        spike_timer.record_spike(time2);
        
        // Should have recorded one interval
        assert!(spike_timer.isi_stats().mean_interval().is_some());
    }

    #[test]
    fn test_isi_stats() {
        let stats = ISIStats::new();
        
        stats.record_interval(SpikeTime::from_millis(10));
        stats.record_interval(SpikeTime::from_millis(10));
        stats.record_interval(SpikeTime::from_millis(10));
        
        let mean = stats.mean_interval().unwrap();
        assert_eq!(mean.as_millis(), 10);
        
        let cv = stats.coefficient_of_variation();
        assert!(cv < 0.1); // Should be very low for regular intervals
    }

    #[test]
    fn test_timer_wheel() {
        let wheel = TimerWheel::new(Duration::from_micros(1));
        
        fn dummy_callback() {}
        
        let timeout_id = wheel.schedule(
            SpikeTime::from_millis(10),
            dummy_callback
        );
        
        // Should be able to cancel
        assert!(!wheel.cancel(timeout_id)); // Our simple implementation doesn't support cancel
    }
}