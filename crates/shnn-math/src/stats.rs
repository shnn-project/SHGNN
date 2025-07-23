//! Statistical functions for neuromorphic data analysis
//! 
//! Provides efficient statistical computations for spike train analysis,
//! neural activity monitoring, and network performance metrics.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::{Float, Vector, approx::safe_divide, constants::EPSILON, math::FloatMath};

/// Basic statistical measures
#[derive(Debug, Clone)]
pub struct BasicStats {
    pub mean: Float,
    pub variance: Float,
    pub std_dev: Float,
    pub min: Float,
    pub max: Float,
    pub count: usize,
}

impl BasicStats {
    /// Compute basic statistics from data slice
    pub fn from_slice(data: &[Float]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        let count = data.len();
        let sum: Float = data.iter().sum();
        let mean = sum / count as Float;

        let mut min = data[0];
        let mut max = data[0];
        let mut sum_sq_diff = 0.0;

        for &value in data {
            min = min.min(value);
            max = max.max(value);
            let diff = value - mean;
            sum_sq_diff += diff * diff;
        }

        let variance = sum_sq_diff / count as Float;
        let std_dev = variance.sqrt();

        Some(BasicStats {
            mean,
            variance,
            std_dev,
            min,
            max,
            count,
        })
    }

    /// Compute statistics from vector
    pub fn from_vector(vector: &Vector) -> Option<Self> {
        Self::from_slice(vector.as_slice())
    }

    /// Get coefficient of variation (std_dev / mean)
    pub fn coefficient_of_variation(&self) -> Float {
        safe_divide(self.std_dev, self.mean.abs())
    }

    /// Get range (max - min)
    pub fn range(&self) -> Float {
        self.max - self.min
    }
}

/// Running statistics for online computation
#[derive(Debug, Clone)]
pub struct RunningStats {
    count: usize,
    mean: Float,
    m2: Float, // Sum of squares of differences from mean
    min: Float,
    max: Float,
}

impl RunningStats {
    /// Create new running statistics
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: Float::INFINITY,
            max: Float::NEG_INFINITY,
        }
    }

    /// Add new value using Welford's algorithm
    pub fn update(&mut self, value: Float) {
        self.count += 1;
        
        // Update min/max
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        
        // Welford's algorithm for mean and variance
        let delta = value - self.mean;
        self.mean += delta / self.count as Float;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get current mean
    pub fn mean(&self) -> Float {
        self.mean
    }

    /// Get current variance
    pub fn variance(&self) -> Float {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as Float
        }
    }

    /// Get current standard deviation
    pub fn std_dev(&self) -> Float {
        self.variance().sqrt()
    }

    /// Get current minimum
    pub fn min(&self) -> Float {
        self.min
    }

    /// Get current maximum
    pub fn max(&self) -> Float {
        self.max
    }

    /// Get count of values
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Histogram for frequency analysis
#[derive(Debug, Clone)]
pub struct Histogram {
    bins: Vec<usize>,
    bin_edges: Vec<Float>,
    min_value: Float,
    max_value: Float,
    bin_width: Float,
}

impl Histogram {
    /// Create new histogram with given number of bins
    pub fn new(min_value: Float, max_value: Float, num_bins: usize) -> Self {
        let bin_width = (max_value - min_value) / num_bins as Float;
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        
        for i in 0..=num_bins {
            bin_edges.push(min_value + i as Float * bin_width);
        }

        Self {
            bins: vec![0; num_bins],
            bin_edges,
            min_value,
            max_value,
            bin_width,
        }
    }

    /// Add value to histogram
    pub fn add(&mut self, value: Float) {
        if value < self.min_value || value > self.max_value {
            return; // Out of range
        }

        let bin_index = ((value - self.min_value) / self.bin_width).floor() as usize;
        let bin_index = bin_index.min(self.bins.len() - 1);
        self.bins[bin_index] += 1;
    }

    /// Add multiple values
    pub fn add_slice(&mut self, values: &[Float]) {
        for &value in values {
            self.add(value);
        }
    }

    /// Get bin counts
    pub fn bins(&self) -> &[usize] {
        &self.bins
    }

    /// Get bin edges
    pub fn edges(&self) -> &[Float] {
        &self.bin_edges
    }

    /// Get total count
    pub fn total_count(&self) -> usize {
        self.bins.iter().sum()
    }

    /// Get normalized histogram (probabilities)
    pub fn normalized(&self) -> Vec<Float> {
        let total = self.total_count() as Float;
        if total == 0.0 {
            return vec![0.0; self.bins.len()];
        }
        
        self.bins.iter().map(|&count| count as Float / total).collect()
    }

    /// Find mode (most frequent bin)
    pub fn mode(&self) -> Option<Float> {
        let max_bin = self.bins.iter().enumerate().max_by_key(|(_, &count)| count)?;
        let bin_index = max_bin.0;
        Some(self.bin_edges[bin_index] + self.bin_width * 0.5)
    }
}

/// Correlation analysis
pub struct Correlation;

impl Correlation {
    /// Compute Pearson correlation coefficient
    pub fn pearson(x: &[Float], y: &[Float]) -> Option<Float> {
        if x.len() != y.len() || x.is_empty() {
            return None;
        }

        let n = x.len() as Float;
        let sum_x: Float = x.iter().sum();
        let sum_y: Float = y.iter().sum();
        let sum_xy: Float = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: Float = x.iter().map(|a| a * a).sum();
        let sum_y2: Float = y.iter().map(|b| b * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator.abs() < EPSILON {
            None
        } else {
            Some(numerator / denominator)
        }
    }

    /// Compute cross-correlation at different lags
    pub fn cross_correlation(x: &[Float], y: &[Float], max_lag: usize) -> Vec<Float> {
        let mut correlations = Vec::with_capacity(2 * max_lag + 1);
        
        for lag in -(max_lag as i32)..=(max_lag as i32) {
            let correlation = Self::cross_correlation_at_lag(x, y, lag);
            correlations.push(correlation);
        }
        
        correlations
    }

    /// Cross-correlation at specific lag
    fn cross_correlation_at_lag(x: &[Float], y: &[Float], lag: i32) -> Float {
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..x.len() {
            let j = i as i32 + lag;
            if j >= 0 && (j as usize) < y.len() {
                sum += x[i] * y[j as usize];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as Float
        } else {
            0.0
        }
    }

    /// Auto-correlation function
    pub fn auto_correlation(data: &[Float], max_lag: usize) -> Vec<Float> {
        Self::cross_correlation(data, data, max_lag)
    }
}

/// Spike train analysis
pub struct SpikeAnalysis;

impl SpikeAnalysis {
    /// Compute firing rate from spike times
    pub fn firing_rate(spike_times: &[Float], window_duration: Float) -> Float {
        if window_duration <= 0.0 {
            return 0.0;
        }
        spike_times.len() as Float / window_duration
    }

    /// Compute inter-spike intervals
    pub fn inter_spike_intervals(spike_times: &[Float]) -> Vec<Float> {
        if spike_times.len() < 2 {
            return Vec::new();
        }

        spike_times.windows(2)
            .map(|window| window[1] - window[0])
            .collect()
    }

    /// Compute coefficient of variation of ISIs (CV_ISI)
    pub fn cv_isi(spike_times: &[Float]) -> Option<Float> {
        let isis = Self::inter_spike_intervals(spike_times);
        if isis.is_empty() {
            return None;
        }

        let stats = BasicStats::from_slice(&isis)?;
        Some(stats.coefficient_of_variation())
    }

    /// Compute spike count in time bins
    pub fn spike_count_histogram(
        spike_times: &[Float],
        bin_size: Float,
        duration: Float,
    ) -> Vec<usize> {
        let num_bins = (duration / bin_size).ceil() as usize;
        let mut counts = vec![0; num_bins];

        for &spike_time in spike_times {
            if spike_time >= 0.0 && spike_time < duration {
                let bin_index = (spike_time / bin_size).floor() as usize;
                if bin_index < counts.len() {
                    counts[bin_index] += 1;
                }
            }
        }

        counts
    }

    /// Compute local variation (measure of irregularity)
    pub fn local_variation(spike_times: &[Float]) -> Option<Float> {
        let isis = Self::inter_spike_intervals(spike_times);
        if isis.len() < 2 {
            return None;
        }

        let mut lv_sum = 0.0;
        for i in 0..isis.len() - 1 {
            let isi1 = isis[i];
            let isi2 = isis[i + 1];
            let diff = isi1 - isi2;
            let sum = isi1 + isi2;
            if sum > EPSILON {
                lv_sum += (diff * diff) / (sum * sum);
            }
        }

        Some(3.0 * lv_sum / (isis.len() - 1) as Float)
    }

    /// Detect bursts in spike train
    pub fn detect_bursts(
        spike_times: &[Float],
        max_isi_in_burst: Float,
        min_spikes_in_burst: usize,
    ) -> Vec<(Float, Float, usize)> {
        let mut bursts = Vec::new();
        let isis = Self::inter_spike_intervals(spike_times);
        
        if isis.len() < min_spikes_in_burst - 1 {
            return bursts;
        }

        let mut burst_start = 0;
        let mut in_burst = false;

        for (i, &isi) in isis.iter().enumerate() {
            if isi <= max_isi_in_burst {
                if !in_burst {
                    burst_start = i;
                    in_burst = true;
                }
            } else {
                if in_burst {
                    let burst_length = i - burst_start + 1;
                    if burst_length >= min_spikes_in_burst {
                        let start_time = spike_times[burst_start];
                        let end_time = spike_times[i];
                        bursts.push((start_time, end_time, burst_length));
                    }
                    in_burst = false;
                }
            }
        }

        // Check if last burst extends to end
        if in_burst {
            let burst_length = isis.len() - burst_start + 1;
            if burst_length >= min_spikes_in_burst {
                let start_time = spike_times[burst_start];
                let end_time = spike_times[spike_times.len() - 1];
                bursts.push((start_time, end_time, burst_length));
            }
        }

        bursts
    }
}

/// Network connectivity analysis
pub struct NetworkAnalysis;

impl NetworkAnalysis {
    /// Compute pairwise synchrony (correlation-based)
    pub fn pairwise_synchrony(spike_train1: &[Float], spike_train2: &[Float], bin_size: Float, duration: Float) -> Float {
        let bins1 = SpikeAnalysis::spike_count_histogram(spike_train1, bin_size, duration);
        let bins2 = SpikeAnalysis::spike_count_histogram(spike_train2, bin_size, duration);
        
        let float_bins1: Vec<Float> = bins1.iter().map(|&x| x as Float).collect();
        let float_bins2: Vec<Float> = bins2.iter().map(|&x| x as Float).collect();
        
        Correlation::pearson(&float_bins1, &float_bins2).unwrap_or(0.0)
    }

    /// Compute population synchrony (mean pairwise correlation)
    pub fn population_synchrony(spike_trains: &[Vec<Float>], bin_size: Float, duration: Float) -> Float {
        if spike_trains.len() < 2 {
            return 0.0;
        }

        let mut total_correlation = 0.0;
        let mut pair_count = 0;

        for i in 0..spike_trains.len() {
            for j in (i + 1)..spike_trains.len() {
                let corr = Self::pairwise_synchrony(&spike_trains[i], &spike_trains[j], bin_size, duration);
                total_correlation += corr;
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_correlation / pair_count as Float
        } else {
            0.0
        }
    }

    /// Compute population firing rate over time
    pub fn population_rate(spike_trains: &[Vec<Float>], bin_size: Float, duration: Float) -> Vec<Float> {
        let num_bins = (duration / bin_size).ceil() as usize;
        let mut rate_bins = vec![0.0; num_bins];

        for spike_train in spike_trains {
            let counts = SpikeAnalysis::spike_count_histogram(spike_train, bin_size, duration);
            for (i, &count) in counts.iter().enumerate() {
                if i < rate_bins.len() {
                    rate_bins[i] += count as Float;
                }
            }
        }

        // Normalize by number of neurons and bin size
        let normalization = spike_trains.len() as Float * bin_size;
        for rate in &mut rate_bins {
            *rate /= normalization;
        }

        rate_bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_stats() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = BasicStats::from_slice(&data).unwrap();
        
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_running_stats() {
        let mut stats = RunningStats::new();
        
        for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.update(value);
        }
        
        assert_eq!(stats.mean(), 3.0);
        assert_eq!(stats.count(), 5);
        assert!(stats.std_dev() > 0.0);
    }

    #[test]
    fn test_histogram() {
        let mut hist = Histogram::new(0.0, 10.0, 5);
        let data = [1.0, 2.0, 2.5, 5.0, 8.0, 9.0];
        
        hist.add_slice(&data);
        
        assert_eq!(hist.total_count(), 6);
        assert!(hist.bins().iter().any(|&count| count > 0));
    }

    #[test]
    fn test_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation
        
        let corr = Correlation::pearson(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_spike_analysis() {
        let spike_times = [0.1, 0.3, 0.7, 1.2, 1.8];
        
        let rate = SpikeAnalysis::firing_rate(&spike_times, 2.0);
        assert_eq!(rate, 2.5); // 5 spikes in 2 seconds
        
        let isis = SpikeAnalysis::inter_spike_intervals(&spike_times);
        assert_eq!(isis.len(), 4);
        assert_eq!(isis[0], 0.2);
        assert_eq!(isis[1], 0.4);
    }

    #[test]
    fn test_cv_isi() {
        // Regular spike train
        let regular_spikes = [0.0, 1.0, 2.0, 3.0, 4.0];
        let cv_regular = SpikeAnalysis::cv_isi(&regular_spikes).unwrap();
        assert!(cv_regular < 0.1); // Should be very low for regular
        
        // Irregular spike train
        let irregular_spikes = [0.0, 0.1, 1.5, 1.7, 4.0];
        let cv_irregular = SpikeAnalysis::cv_isi(&irregular_spikes).unwrap();
        assert!(cv_irregular > cv_regular);
    }

    #[test]
    fn test_burst_detection() {
        // Spike train with bursts: two bursts of 3 spikes each
        let spike_times = [0.1, 0.15, 0.2, 1.0, 1.05, 1.1, 2.5];
        let bursts = SpikeAnalysis::detect_bursts(&spike_times, 0.1, 3);
        
        assert_eq!(bursts.len(), 2);
        assert_eq!(bursts[0].2, 3); // First burst has 3 spikes
        assert_eq!(bursts[1].2, 3); // Second burst has 3 spikes
    }

    #[test]
    fn test_network_synchrony() {
        // Two identical spike trains should have perfect synchrony
        let train1 = vec![0.1, 0.3, 0.7];
        let train2 = vec![0.1, 0.3, 0.7];
        
        let sync = NetworkAnalysis::pairwise_synchrony(&train1, &train2, 0.1, 1.0);
        assert!(sync > 0.8); // Should be high correlation
        
        // Test population synchrony
        let trains = vec![train1, train2];
        let pop_sync = NetworkAnalysis::population_synchrony(&trains, 0.1, 1.0);
        assert!(pop_sync > 0.8);
    }

    #[test]
    fn test_population_rate() {
        let train1 = vec![0.1, 0.3];
        let train2 = vec![0.2, 0.4];
        let trains = vec![train1, train2];
        
        let pop_rate = NetworkAnalysis::population_rate(&trains, 0.2, 1.0);
        assert_eq!(pop_rate.len(), 5); // 1.0 / 0.2 = 5 bins
        assert!(pop_rate.iter().any(|&rate| rate > 0.0));
    }
}

// Type alias for compatibility
pub type Statistics = BasicStats;

// Standalone stats functions for compatibility
pub fn mean(data: &[f32]) -> f32 {
    BasicStats::from_slice(data).map(|stats| stats.mean).unwrap_or(0.0)
}

pub fn variance(data: &[f32]) -> f32 {
    BasicStats::from_slice(data).map(|stats| stats.variance).unwrap_or(0.0)
}

pub fn standard_deviation(data: &[f32]) -> f32 {
    BasicStats::from_slice(data).map(|stats| stats.std_dev).unwrap_or(0.0)
}

pub fn correlation(x: &[f32], y: &[f32]) -> f32 {
    // Basic correlation implementation - Pearson correlation coefficient
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    
    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;
    
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    
    let denominator = (sum_x2 * sum_y2).sqrt();
    if denominator > 0.0 {
        sum_xy / denominator
    } else {
        0.0
    }
}

pub fn normalize(data: &mut [f32]) {
    let stats = BasicStats::from_slice(data);
    if let Some(stats) = stats {
        if stats.std_dev > 0.0 {
            for value in data.iter_mut() {
                *value = (*value - stats.mean) / stats.std_dev;
            }
        }
    }
}