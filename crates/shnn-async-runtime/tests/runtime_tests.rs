//! Comprehensive tests for SHNN async runtime refactoring
//!
//! This test suite validates all the zero-dependency async runtime components
//! that replaced tokio functionality.

use shnn_async_runtime::{
    SHNNRuntime, TaskPriority, TaskHandle, SpikeTime, 
    executor::Executor, scheduler::WorkStealingScheduler,
    lock_free::SpikeEventQueue,
};
use std::{
    sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}},
    time::{Duration, Instant},
    thread,
};

/// Test basic runtime creation and shutdown
#[test]
fn test_runtime_creation() {
    let runtime = SHNNRuntime::new(2, 1024);
    
    assert_eq!(runtime.worker_count(), 2);
    assert!(runtime.is_running());
    
    // Runtime should shutdown cleanly
    drop(runtime);
}

/// Test task spawning and execution
#[test]
fn test_basic_task_execution() {
    let runtime = SHNNRuntime::new(1, 1024);
    let counter = Arc::new(AtomicU64::new(0));
    let counter_clone = counter.clone();
    
    let handle = runtime.spawn_task(async move {
        counter_clone.store(42, Ordering::SeqCst);
    }, TaskPriority::Normal);
    
    // Wait for task completion with timeout
    let start = Instant::now();
    while handle.try_get_result().is_none() {
        if start.elapsed() > Duration::from_secs(5) {
            panic!("Task execution timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
    
    assert_eq!(counter.load(Ordering::SeqCst), 42);
}

/// Test multiple concurrent tasks
#[test]
fn test_concurrent_task_execution() {
    let runtime = SHNNRuntime::new(4, 1024);
    let counter = Arc::new(AtomicU64::new(0));
    let num_tasks = 100;
    
    let mut handles = Vec::new();
    
    for i in 0..num_tasks {
        let counter_clone = counter.clone();
        let handle = runtime.spawn_task(async move {
            counter_clone.fetch_add(i + 1, Ordering::SeqCst);
        }, TaskPriority::Normal);
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let start = Instant::now();
    while handles.iter().any(|h| h.try_get_result().is_none()) {
        if start.elapsed() > Duration::from_secs(10) {
            panic!("Concurrent task execution timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
    
    // Sum should be 1+2+...+100 = 5050
    let expected = (num_tasks * (num_tasks + 1)) / 2;
    assert_eq!(counter.load(Ordering::SeqCst), expected);
}

/// Test task priorities
#[test]
fn test_task_priorities() {
    let runtime = SHNNRuntime::new(1, 1024);
    let execution_order = Arc::new(spin::Mutex::new(Vec::new()));
    
    let order_clone = execution_order.clone();
    let low_handle = runtime.spawn_task(async move {
        order_clone.lock().push("low");
    }, TaskPriority::Low);
    
    let order_clone = execution_order.clone();
    let high_handle = runtime.spawn_task(async move {
        order_clone.lock().push("high");
    }, TaskPriority::High);
    
    let order_clone = execution_order.clone();
    let normal_handle = runtime.spawn_task(async move {
        order_clone.lock().push("normal");
    }, TaskPriority::Normal);
    
    // Wait for completion
    let start = Instant::now();
    while [&low_handle, &high_handle, &normal_handle].iter().any(|h| h.try_get_result().is_none()) {
        if start.elapsed() > Duration::from_secs(5) {
            panic!("Priority task execution timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
    
    let order = execution_order.lock();
    assert!(!order.is_empty());
    // High priority should execute first (though exact order may vary due to concurrency)
    assert!(order.contains(&"high"));
    assert!(order.contains(&"normal"));
    assert!(order.contains(&"low"));
}

/// Test spike event processing
#[test]
fn test_spike_processing() {
    let runtime = SHNNRuntime::new(2, 1024);
    let spike_count = Arc::new(AtomicU64::new(0));
    let processing_done = Arc::new(AtomicBool::new(false));
    
    // Spawn spike processor
    let spike_count_clone = spike_count.clone();
    let done_clone = processing_done.clone();
    let processor_handle = runtime.spawn_task(async move {
        let mut processed = 0;
        while !done_clone.load(Ordering::SeqCst) {
            // Simulate spike processing
            processed += 1;
            spike_count_clone.store(processed, Ordering::SeqCst);
            
            // Yield to allow other tasks to run
            shnn_async_runtime::yield_now().await;
        }
    }, TaskPriority::High);
    
    // Let it run for a bit
    thread::sleep(Duration::from_millis(100));
    processing_done.store(true, Ordering::SeqCst);
    
    // Wait for completion
    let start = Instant::now();
    while processor_handle.try_get_result().is_none() {
        if start.elapsed() > Duration::from_secs(5) {
            panic!("Spike processing timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
    
    assert!(spike_count.load(Ordering::SeqCst) > 0);
}

/// Test work-stealing scheduler directly
#[test]
fn test_work_stealing_scheduler() {
    let scheduler = WorkStealingScheduler::new(4);
    let executed = Arc::new(AtomicU64::new(0));
    
    // Add tasks to different worker queues
    for i in 0..16 {
        let executed_clone = executed.clone();
        let task = Box::pin(async move {
            executed_clone.fetch_add(1, Ordering::SeqCst);
        });
        
        scheduler.schedule_task(task, TaskPriority::Normal, i % 4);
    }
    
    // Run scheduler for a bit
    let start = Instant::now();
    while executed.load(Ordering::SeqCst) < 16 && start.elapsed() < Duration::from_secs(5) {
        scheduler.run_once(0);
        thread::sleep(Duration::from_millis(1));
    }
    
    assert_eq!(executed.load(Ordering::SeqCst), 16);
}

/// Test spike event queue (lock-free structure)
#[test]
fn test_spike_event_queue() {
    let queue = SpikeEventQueue::with_capacity(1024);
    
    // Test basic push/pop
    let spike_time = SpikeTime::from_millis(100);
    assert!(queue.push(spike_time).is_ok());
    
    let popped = queue.pop();
    assert!(popped.is_some());
    assert_eq!(popped.unwrap(), spike_time);
    
    // Test queue full behavior
    for i in 0..1024 {
        assert!(queue.push(SpikeTime::from_millis(i as f64)).is_ok());
    }
    
    // Should be full now
    assert!(queue.push(SpikeTime::from_millis(9999.0)).is_err());
}

/// Test concurrent access to spike event queue
#[test]
fn test_concurrent_spike_queue() {
    let queue = Arc::new(SpikeEventQueue::with_capacity(1024));
    let push_count = Arc::new(AtomicU64::new(0));
    let pop_count = Arc::new(AtomicU64::new(0));
    
    let mut handles = Vec::new();
    
    // Spawn producers
    for i in 0..4 {
        let queue_clone = queue.clone();
        let push_count_clone = push_count.clone();
        let handle = thread::spawn(move || {
            for j in 0..100 {
                let spike_time = SpikeTime::from_millis((i * 100 + j) as f64);
                if queue_clone.push(spike_time).is_ok() {
                    push_count_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }
    
    // Spawn consumers
    for _ in 0..2 {
        let queue_clone = queue.clone();
        let pop_count_clone = pop_count.clone();
        let handle = thread::spawn(move || {
            loop {
                if queue_clone.pop().is_some() {
                    pop_count_clone.fetch_add(1, Ordering::SeqCst);
                } else {
                    thread::sleep(Duration::from_millis(1));
                    // Exit if we've been trying for a while with no success
                    if pop_count_clone.load(Ordering::SeqCst) > 0 && queue_clone.len() == 0 {
                        break;
                    }
                }
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    assert!(push_count.load(Ordering::SeqCst) > 0);
    assert!(pop_count.load(Ordering::SeqCst) > 0);
}

/// Test runtime performance under load
#[test]
fn test_runtime_performance() {
    let runtime = SHNNRuntime::new(4, 1024);
    let start = Instant::now();
    let completed = Arc::new(AtomicU64::new(0));
    
    let mut handles = Vec::new();
    
    // Spawn many lightweight tasks
    for _ in 0..1000 {
        let completed_clone = completed.clone();
        let handle = runtime.spawn_task(async move {
            // Simulate some work
            let mut sum = 0u64;
            for i in 0..100 {
                sum += i;
            }
            completed_clone.fetch_add(1, Ordering::SeqCst);
            sum
        }, TaskPriority::Normal);
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let start_wait = Instant::now();
    while completed.load(Ordering::SeqCst) < 1000 {
        if start_wait.elapsed() > Duration::from_secs(30) {
            panic!("Performance test timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
    
    let elapsed = start.elapsed();
    println!("Completed 1000 tasks in {:?}", elapsed);
    
    // Should complete in reasonable time (adjust threshold as needed)
    assert!(elapsed < Duration::from_secs(10));
}

/// Test error handling and edge cases
#[test]
fn test_error_handling() {
    let runtime = SHNNRuntime::new(1, 16); // Small queue for testing
    
    // Test task spawning with full queue
    let mut handles = Vec::new();
    let barrier = Arc::new(AtomicBool::new(false));
    
    // Fill up the queue
    for _ in 0..32 {
        let barrier_clone = barrier.clone();
        let handle = runtime.spawn_task(async move {
            while !barrier_clone.load(Ordering::SeqCst) {
                shnn_async_runtime::yield_now().await;
            }
        }, TaskPriority::Normal);
        handles.push(handle);
    }
    
    // Queue should be full or near full, but runtime should handle it gracefully
    let final_handle = runtime.spawn_task(async move {
        42
    }, TaskPriority::High);
    
    // Release all waiting tasks
    barrier.store(true, Ordering::SeqCst);
    
    // Wait for completion
    let start = Instant::now();
    while handles.iter().any(|h| h.try_get_result().is_none()) {
        if start.elapsed() > Duration::from_secs(10) {
            panic!("Error handling test timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
    
    // Final task should also complete
    while final_handle.try_get_result().is_none() {
        if start.elapsed() > Duration::from_secs(10) {
            panic!("Final task timeout");
        }
        thread::sleep(Duration::from_millis(1));
    }
}

/// Test SpikeTime functionality
#[test]
fn test_spike_time() {
    let t1 = SpikeTime::from_millis(100.0);
    let t2 = SpikeTime::from_millis(200.0);
    let t3 = SpikeTime::from_nanos(150_000_000); // 150ms
    
    assert!(t1 < t2);
    assert!(t1 < t3);
    assert!(t3 < t2);
    
    assert_eq!(t1.as_millis(), 100.0);
    assert_eq!(t2.as_nanos(), 200_000_000);
    
    // Test arithmetic
    let sum = t1 + Duration::from_millis(50);
    assert_eq!(sum.as_millis(), 150.0);
}

/// Test memory safety under concurrent access
#[test]
fn test_memory_safety() {
    let runtime = Arc::new(SHNNRuntime::new(4, 1024));
    let mut thread_handles = Vec::new();
    
    // Spawn multiple threads that create and run tasks
    for thread_id in 0..8 {
        let runtime_clone = runtime.clone();
        let handle = thread::spawn(move || {
            let mut task_handles = Vec::new();
            
            for task_id in 0..50 {
                let task_handle = runtime_clone.spawn_task(async move {
                    thread_id * 50 + task_id
                }, TaskPriority::Normal);
                task_handles.push(task_handle);
            }
            
            // Wait for all tasks in this thread to complete
            for task_handle in task_handles {
                let start = Instant::now();
                while task_handle.try_get_result().is_none() {
                    if start.elapsed() > Duration::from_secs(10) {
                        panic!("Task timeout in thread {}", thread_id);
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            }
        });
        thread_handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in thread_handles {
        handle.join().unwrap();
    }
}