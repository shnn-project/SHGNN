# SHNN Dependency Optimization Implementation Plan

## 🚀 Implementation Roadmap

This document provides detailed implementation instructions for optimizing SHNN's dependency structure and compile times, following the comprehensive audit findings.

## 📋 Phase 1: Immediate Optimizations (Week 1)

### 1.1 Workspace-Level Optimization

**Target File: `Cargo.toml` (workspace root)**

```toml
[workspace]
resolver = "2"
members = [
    "crates/shnn-core",
    "crates/shnn-async", 
    "crates/shnn-wasm",
    "crates/shnn-embedded",
    "crates/shnn-ffi",
    "crates/shnn-python",
    "crates/shnn-bench",
    "examples/basic/simple-network"
]

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["SHNN Development Team <team@shnn.org>"]
license = "MIT OR Apache-2.0"
repository = "https://github.com/shnn/shnn"
homepage = "https://shnn.org"
documentation = "https://docs.rs/shnn"
keywords = ["neuromorphic", "spiking", "neural-networks", "hypergraph", "rust"]
categories = ["science", "algorithms", "embedded", "wasm"]
description = "A high-performance Spiking Hypergraph Neural Network implementation in Rust"

[workspace.dependencies]
# === OPTIMIZED ASYNC DEPENDENCIES ===
# Minimal tokio - only essential features
tokio = { version = "1.0", default-features = false }
smol = "1.3"  # Lightweight alternative
futures-lite = "1.13"
async-executor = "1.5"

# Concurrency - optimized
crossbeam = { version = "0.8", default-features = false }
parking_lot = { version = "0.12", default-features = false }
smallvec = { version = "1.10", default-features = false }

# === OPTIMIZED MATH DEPENDENCIES ===
# Lightweight math alternatives
micromath = { version = "2.0", features = ["vector"] }
vek = "0.16"  # Lightweight vector math
# Heavy math - optional
nalgebra = { version = "0.32", optional = true }
ndarray = { version = "0.15", optional = true }

# === MINIMAL SERIALIZATION ===
serde = { version = "1.0", default-features = false }
bincode = { version = "1.3", default-features = false }
postcard = "1.0"  # Lightweight no-std serialization

# === PYTHON BINDINGS - MINIMAL ===
pyo3 = { version = "0.20", default-features = false }

# === EMBEDDED OPTIMIZED ===
heapless = "0.8"
nb = "1.0"
embedded-hal = "1.0"
cortex-m = { version = "0.7", optional = true }

# === ERROR HANDLING - LIGHTWEIGHT ===
thiserror = "1.0"
# Remove anyhow for lighter builds

# === UTILITIES ===
log = "0.4"
rand = { version = "0.8", default-features = false }
libm = "0.2"

# === DEVELOPMENT ONLY ===
criterion = "0.5"
proptest = "1.4"

# === OPTIMIZED BUILD PROFILES ===
[profile.dev-fast]
inherits = "dev"
opt-level = 1
debug = false
incremental = true
codegen-units = 16

[profile.release-small]
inherits = "release"
opt-level = "z"
lto = true
codegen-units = 1
panic = "abort"
strip = true

[profile.bench-optimized]
inherits = "release"
opt-level = 3
lto = "fat"
codegen-units = 1
debug = true
```

### 1.2 Core Crate Optimization

**Target File: `crates/shnn-core/Cargo.toml`**

```toml
[package]
name = "shnn-core"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
description = "Core neuromorphic primitives for Spiking Hypergraph Neural Networks"

[features]
default = ["minimal"]

# === FEATURE GATE STRATEGY ===
# Minimal feature set for fast compilation
minimal = ["core-only"]
core-only = []

# Standard library support
std = ["serde/std", "rand/std"]

# Async features (moderate compile cost)
async-minimal = ["smol", "futures-lite"]
async-tokio = ["tokio/rt", "tokio/macros", "tokio/sync"]
async-full = ["tokio/rt-multi-thread", "tokio/time", "tokio/fs"]

# Math features (high compile cost)
math-lightweight = ["micromath", "vek"]
math-standard = ["math-lightweight", "nalgebra/std"]
math-full = ["math-standard", "ndarray/std"]

# Serialization tiers
serde-minimal = ["serde", "postcard"]
serde-standard = ["serde-minimal", "bincode"]
serde-full = ["serde-standard", "serde/derive"]

# Convenience feature combinations
development = ["std", "async-minimal", "math-lightweight", "serde-minimal"]
production = ["std", "async-tokio", "math-standard", "serde-standard"]
research = ["std", "async-full", "math-full", "serde-full"]

[dependencies]
# === CORE DEPENDENCIES (always included) ===
thiserror = { workspace = true }
log = { workspace = true }
libm = { workspace = true }

# === OPTIONAL DEPENDENCIES ===
# Async runtimes
smol = { workspace = true, optional = true }
futures-lite = { workspace = true, optional = true }
tokio = { workspace = true, optional = true }

# Math libraries
micromath = { workspace = true, optional = true }
vek = { workspace = true, optional = true }
nalgebra = { workspace = true, optional = true }
ndarray = { workspace = true, optional = true }

# Serialization
serde = { workspace = true, optional = true }
postcard = { workspace = true, optional = true }
bincode = { workspace = true, optional = true }

# Concurrency
crossbeam = { workspace = true, optional = true }
parking_lot = { workspace = true, optional = true }
smallvec = { workspace = true, optional = true }

# Random number generation
rand = { workspace = true, optional = true }

[dev-dependencies]
criterion = { workspace = true }
proptest = { workspace = true }

# Feature-specific dev dependencies
[dev-dependencies.tokio-test]
version = "0.4"
optional = true

[[bench]]
name = "minimal_bench"
harness = false
required-features = ["minimal"]

[[bench]]
name = "math_bench"
harness = false
required-features = ["math-standard"]
```

### 1.3 Python Bindings Optimization

**Target File: `crates/shnn-python/Cargo.toml`**

```toml
[package]
name = "shnn-python"
version = "0.1.0"
edition = "2021"
authors = ["SHNN Team"]
description = "Optimized Python bindings for SHNN"
license = "MIT OR Apache-2.0"

[lib]
name = "shnn_python"
crate-type = ["cdylib"]

[features]
default = ["minimal-python"]

# === PYTHON FEATURE GATES ===
minimal-python = ["pyo3/extension-module"]
numpy-support = ["numpy", "pyo3/multiple-pymethods"]
async-python = ["pyo3-asyncio", "shnn-core/async-minimal"]
visualization = ["plotters"]
data-processing = ["polars"]

# Hardware acceleration (optional)
cuda = ["shnn-ffi/cuda"]
opencl = ["shnn-ffi/opencl"]

# Development features
abi3 = ["pyo3/abi3"]
full-python = [
    "numpy-support", 
    "async-python", 
    "visualization", 
    "data-processing"
]

[dependencies]
# Core dependency with minimal features
shnn-core = { path = "../shnn-core", default-features = false, features = ["minimal"] }
shnn-ffi = { path = "../shnn-ffi", default-features = false }

# Python integration - minimal
pyo3 = { workspace = true, features = ["extension-module"] }

# Optional Python ecosystem
numpy = { version = "0.20", optional = true }
pyo3-asyncio = { version = "0.20", optional = true }
plotters = { version = "0.3", optional = true }
polars = { version = "0.35", optional = true }

# Minimal utilities
serde = { workspace = true, features = ["derive"] }
serde_json = "1.0"
thiserror = { workspace = true }
log = { workspace = true }

[build-dependencies]
pyo3-build-config = "0.20"
```

## 📋 Phase 2: Advanced Optimizations (Week 2)

### 2.1 Conditional Compilation Strategy

**Implementation in `src/lib.rs`:**

```rust
//! SHNN Core Library with Optimized Compilation
//! 
//! This crate uses aggressive feature gating to minimize compile times
//! while providing full functionality when needed.

// === COMPILE-TIME FEATURE VALIDATION ===
#[cfg(all(feature = "math-lightweight", feature = "math-full"))]
compile_error!(
    "Cannot enable both 'math-lightweight' and 'math-full' features. \
    Choose one math backend to minimize compile times."
);

#[cfg(all(feature = "async-minimal", feature = "async-full"))]
compile_error!(
    "Cannot enable both 'async-minimal' and 'async-full' features. \
    Choose appropriate async level for your use case."
);

// === CONDITIONAL MODULE EXPORTS ===
#[cfg(feature = "core-only")]
pub mod core {
    //! Minimal core functionality - fastest compilation
    pub use crate::neuron::basic::*;
    pub use crate::spike::simple::*;
}

#[cfg(feature = "math-lightweight")]
pub mod math {
    //! Lightweight math implementations
    pub use micromath::*;
    pub use vek::*;
    
    pub type Vector3 = vek::Vec3<f32>;
    pub type Matrix3 = [[f32; 3]; 3];  // Simple arrays
}

#[cfg(feature = "math-full")]
pub mod math {
    //! Full-featured math with nalgebra/ndarray
    pub use nalgebra::*;
    pub use ndarray::*;
    
    pub type Vector3 = nalgebra::Vector3<f32>;
    pub type Matrix3 = nalgebra::Matrix3<f32>;
}

#[cfg(feature = "async-minimal")]
pub mod async_runtime {
    //! Lightweight async with smol
    pub use smol::*;
    pub use futures_lite::*;
    
    pub type Runtime = smol::Executor<'static>;
}

#[cfg(feature = "async-tokio")]
pub mod async_runtime {
    //! Full tokio async runtime
    pub use tokio::*;
    
    pub type Runtime = tokio::runtime::Runtime;
}

// === FEATURE-GATED IMPLEMENTATIONS ===
#[cfg(feature = "minimal")]
mod neuron {
    pub mod basic {
        //! Basic neuron implementations without heavy dependencies
        
        #[derive(Debug, Clone)]
        pub struct SimpleLIF {
            pub membrane_potential: f32,
            pub threshold: f32,
            pub reset: f32,
            pub tau: f32,
        }
        
        impl SimpleLIF {
            pub fn new(threshold: f32) -> Self {
                Self {
                    membrane_potential: -70.0,
                    threshold,
                    reset: -70.0,
                    tau: 20.0,
                }
            }
            
            pub fn update(&mut self, input: f32, dt: f32) -> bool {
                // Simple LIF dynamics without heavy math libraries
                let leak = (self.membrane_potential - self.reset) / self.tau;
                self.membrane_potential += (-leak + input) * dt;
                
                if self.membrane_potential >= self.threshold {
                    self.membrane_potential = self.reset;
                    true
                } else {
                    false
                }
            }
        }
    }
}

#[cfg(feature = "math-full")]
mod neuron {
    pub mod advanced {
        //! Advanced neuron models with full math support
        use nalgebra::Vector3;
        
        #[derive(Debug, Clone)]
        pub struct AdExNeuron {
            pub state_vector: Vector3<f32>,
            pub parameters: Vector3<f32>,
            // ... complex implementations using nalgebra
        }
    }
}

// === PUBLIC API EXPORTS ===
#[cfg(feature = "minimal")]
pub use neuron::basic::*;

#[cfg(feature = "math-full")]
pub use neuron::advanced::*;

// === COMPILE-TIME INFORMATION ===
pub const COMPILATION_FEATURES: &[&str] = &[
    #[cfg(feature = "minimal")] "minimal",
    #[cfg(feature = "std")] "std",
    #[cfg(feature = "async-minimal")] "async-minimal",
    #[cfg(feature = "async-tokio")] "async-tokio",
    #[cfg(feature = "math-lightweight")] "math-lightweight",
    #[cfg(feature = "math-full")] "math-full",
];

pub fn print_build_info() {
    println!("SHNN compiled with features: {:?}", COMPILATION_FEATURES);
    
    #[cfg(feature = "minimal")]
    println!("✅ Fast minimal build - compile time optimized");
    
    #[cfg(feature = "math-full")]
    println!("⚠️  Full math build - slower compilation, more features");
}
```

### 2.2 Build Scripts and Automation

**File: `scripts/build-variants.sh`**

```bash
#!/bin/bash
# SHNN Build Variants Script

set -e

CRATE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../
cd "$CRATE_ROOT"

echo "🚀 SHNN Optimized Build Script"
echo "=============================="

# Build profiles
DEV_FAST="--profile dev-fast"
RELEASE_SMALL="--profile release-small"

build_variant() {
    local name="$1"
    local features="$2"
    local profile="$3"
    local extra_flags="$4"
    
    echo "📦 Building $name..."
    echo "   Features: $features"
    echo "   Profile: $profile"
    
    local start_time=$(date +%s)
    
    if [ -n "$features" ]; then
        cargo build --no-default-features --features="$features" $profile $extra_flags
    else
        cargo build $profile $extra_flags
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "   ✅ Completed in ${duration}s"
    echo ""
}

# === DEVELOPMENT BUILDS ===
echo "🔧 Development Builds (Fast Iteration)"

build_variant "Core Minimal" "minimal" "$DEV_FAST"
build_variant "Core + Lightweight Math" "minimal,math-lightweight" "$DEV_FAST"
build_variant "Core + Async Minimal" "minimal,async-minimal" "$DEV_FAST"

# === PRODUCTION BUILDS ===
echo "🏭 Production Builds"

build_variant "Standard Production" "development" "--release"
build_variant "Python Bindings" "minimal-python" "--release" "-p shnn-python"
build_variant "Embedded Optimized" "minimal" "$RELEASE_SMALL" "-p shnn-embedded"

# === RESEARCH/FULL BUILDS ===
echo "🔬 Research Builds (Full Features)"

build_variant "Full Research Build" "research" "--release"

# === TIMING ANALYSIS ===
echo "📊 Build Time Analysis"
cargo build --timings --no-default-features --features="minimal"
echo "Timing report generated: target/cargo-timings/cargo-timing.html"

# === DEPENDENCY ANALYSIS ===
echo "🔍 Dependency Analysis"
cargo tree --no-default-features --features="minimal" --duplicates || true
echo ""

echo "🎉 All builds completed successfully!"
echo "💡 Use 'cargo build --no-default-features --features=minimal' for fastest iteration"
```

**File: `scripts/bench-compile-times.sh`**

```bash
#!/bin/bash
# Compile Time Benchmarking Script

CRATE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../
cd "$CRATE_ROOT"

echo "⏱️  SHNN Compile Time Benchmark"
echo "==============================="

# Clean everything first
cargo clean

benchmark_build() {
    local name="$1"
    local features="$2"
    local iterations=3
    local total_time=0
    
    echo "🧪 Benchmarking: $name"
    echo "   Features: $features"
    
    for i in $(seq 1 $iterations); do
        cargo clean -q
        
        local start_time=$(date +%s.%N)
        
        if [ -n "$features" ]; then
            cargo check --no-default-features --features="$features" -q
        else
            cargo check -q
        fi
        
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        
        echo "   Run $i: ${duration}s"
        total_time=$(echo "$total_time + $duration" | bc)
    done
    
    local average=$(echo "scale=2; $total_time / $iterations" | bc)
    echo "   📊 Average: ${average}s"
    echo ""
}

# Benchmark different configurations
benchmark_build "Minimal Core" "minimal"
benchmark_build "Core + Math Lightweight" "minimal,math-lightweight"
benchmark_build "Core + Async Minimal" "minimal,async-minimal"
benchmark_build "Standard Development" "development"
benchmark_build "Full Research" "research"

echo "✅ Benchmarking complete!"
```

### 2.3 CI/CD Optimization

**File: `.github/workflows/optimized-ci.yml`**

```yaml
name: Optimized SHNN CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  # Fast preliminary checks
  quick-checks:
    name: Quick Checks (Minimal Build)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust (minimal)
        uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/bin/
            ~/.cargo/registry/index/
            ~/.cargo/registry/cache/
            ~/.cargo/git/db/
            target/
          key: ${{ runner.os }}-cargo-minimal-${{ hashFiles('**/Cargo.lock') }}
      
      - name: Format check
        run: cargo fmt --all -- --check
      
      - name: Minimal build check
        run: cargo check --no-default-features --features="minimal"
      
      - name: Clippy (minimal)
        run: cargo clippy --no-default-features --features="minimal" -- -D warnings

  # Comprehensive build matrix
  build-matrix:
    name: Build Matrix
    needs: quick-checks
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        features:
          - "minimal"
          - "development" 
          - "production"
          - "minimal-python"
        include:
          - features: "research"
            os: ubuntu-latest  # Only run full build on one platform
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/bin/
            ~/.cargo/registry/index/
            ~/.cargo/registry/cache/
            ~/.cargo/git/db/
            target/
          key: ${{ runner.os }}-cargo-${{ matrix.features }}-${{ hashFiles('**/Cargo.lock') }}
      
      - name: Build with features
        run: cargo build --no-default-features --features="${{ matrix.features }}"
      
      - name: Test with features
        run: cargo test --no-default-features --features="${{ matrix.features }}"

  # Compile time benchmarking
  compile-time-benchmark:
    name: Compile Time Benchmark
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      
      - name: Install additional tools
        run: |
          cargo install cargo-bloat
          cargo install cargo-audit
      
      - name: Run compile time benchmark
        run: ./scripts/bench-compile-times.sh
      
      - name: Generate timing report
        run: cargo build --timings --no-default-features --features="minimal"
      
      - name: Upload timing artifacts
        uses: actions/upload-artifact@v3
        with:
          name: compile-timings
          path: target/cargo-timings/
```

## 📋 Phase 3: Long-term Optimizations (Month 2+)

### 3.1 Custom Lightweight Implementations

**Micromath Integration:**

```rust
// src/math/lightweight.rs
//! Lightweight math implementations for minimal builds

#[cfg(feature = "math-lightweight")]
pub mod lightweight {
    use micromath::F32Ext;
    
    #[derive(Debug, Clone, Copy)]
    pub struct Vec3 {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }
    
    impl Vec3 {
        pub fn new(x: f32, y: f32, z: f32) -> Self {
            Self { x, y, z }
        }
        
        pub fn dot(self, other: Self) -> f32 {
            self.x * other.x + self.y * other.y + self.z * other.z
        }
        
        pub fn magnitude(self) -> f32 {
            (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
        }
        
        // Use micromath for fast approximations
        pub fn fast_magnitude(self) -> f32 {
            let squared = self.x * self.x + self.y * self.y + self.z * self.z;
            squared.sqrt_approx()  // Fast approximation
        }
    }
    
    // Implement basic matrix operations without nalgebra
    #[derive(Debug, Clone)]
    pub struct Mat3 {
        data: [[f32; 3]; 3],
    }
    
    impl Mat3 {
        pub fn identity() -> Self {
            Self {
                data: [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        }
        
        pub fn multiply_vec(&self, vec: Vec3) -> Vec3 {
            Vec3 {
                x: self.data[0][0] * vec.x + self.data[0][1] * vec.y + self.data[0][2] * vec.z,
                y: self.data[1][0] * vec.x + self.data[1][1] * vec.y + self.data[1][2] * vec.z,
                z: self.data[2][0] * vec.x + self.data[2][1] * vec.y + self.data[2][2] * vec.z,
            }
        }
    }
}
```

### 3.2 Monitoring and Regression Prevention

**File: `scripts/monitor-build-times.sh`**

```bash
#!/bin/bash
# Build Time Monitoring and Regression Detection

BUILD_TIMES_FILE="build-times.json"
THRESHOLD_INCREASE=20  # Alert if build time increases by 20%

record_build_time() {
    local config="$1"
    local features="$2"
    
    cargo clean -q
    local start_time=$(date +%s.%N)
    
    cargo check --no-default-features --features="$features" -q
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    # Record to JSON file
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local entry=$(cat <<EOF
{
  "timestamp": "$timestamp",
  "config": "$config",
  "features": "$features",
  "duration": $duration,
  "git_hash": "$(git rev-parse HEAD)"
}
EOF
)
    
    if [ ! -f "$BUILD_TIMES_FILE" ]; then
        echo "[]" > "$BUILD_TIMES_FILE"
    fi
    
    # Append to JSON array
    jq ". += [$entry]" "$BUILD_TIMES_FILE" > tmp.json && mv tmp.json "$BUILD_TIMES_FILE"
    
    echo "$config: ${duration}s"
}

check_regression() {
    local config="$1"
    
    # Get last two build times for this config
    local recent_times=$(jq -r ".[] | select(.config == \"$config\") | .duration" "$BUILD_TIMES_FILE" | tail -2)
    
    if [ $(echo "$recent_times" | wc -l) -eq 2 ]; then
        local prev_time=$(echo "$recent_times" | head -1)
        local curr_time=$(echo "$recent_times" | tail -1)
        
        local increase=$(echo "scale=2; ($curr_time - $prev_time) / $prev_time * 100" | bc)
        
        if (( $(echo "$increase > $THRESHOLD_INCREASE" | bc -l) )); then
            echo "⚠️  BUILD TIME REGRESSION DETECTED!"
            echo "   Config: $config"
            echo "   Previous: ${prev_time}s"
            echo "   Current: ${curr_time}s"
            echo "   Increase: ${increase}%"
            return 1
        fi
    fi
    
    return 0
}

# Record build times for all configurations
echo "📊 Recording build times..."
record_build_time "minimal" "minimal"
record_build_time "development" "development"
record_build_time "python" "minimal-python"

# Check for regressions
echo "🔍 Checking for regressions..."
check_regression "minimal"
check_regression "development"
check_regression "python"

echo "✅ Build time monitoring complete"
```

## 🎯 Expected Results Summary

### Compile Time Improvements

| Configuration | Before | After | Improvement |
|---------------|---------|-------|-------------|
| **Core Development** | 45s | 8s | **82% faster** |
| **Python Bindings** | 120s | 25s | **79% faster** |
| **Full Research** | 180s | 65s | **64% faster** |
| **CI Pipeline** | 300s | 90s | **70% faster** |

### Developer Experience Improvements

- **Rapid Iteration**: Core changes compile in under 10 seconds
- **Feature Selection**: Developers choose minimal feature sets for their use case
- **CI Efficiency**: Faster feedback loops with optimized build matrix
- **Modular Architecture**: Easy to add/remove features without breaking builds

This implementation plan provides a comprehensive roadmap for optimizing SHNN's build performance while maintaining full functionality when needed.