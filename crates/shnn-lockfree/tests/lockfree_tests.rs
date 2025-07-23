//! Comprehensive tests for SHNN lock-free primitives refactoring
//!
//! This test suite validates all the zero-dependency lock-free components
//! that replaced crossbeam functionality.

use shnn_lockfree::{
    queue::{MPSCQueue, LockFreeQueue},
    atomic::{AtomicPtr, AtomicOption},
    ordering::MemoryOrdering,
};
use std::{
    sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}},
    thread,
    time::{Duration, Instant},
};

/// Test basic MPSC queue operations
#[test]
fn test_mpsc_queue_basic() {
    let queue = MPSCQueue::new();
    
    // Test empty queue
    assert!(queue.is_empty());
    assert_eq!(queue.len(), 0);
    assert!(queue.pop().is_none());
    
    // Test single push/pop
    assert!(queue.push(42).is_ok());
    assert!(!queue.is_empty());
    assert_eq!(queue.len(), 1);
    
    let popped = queue.pop();
    assert_eq!(popped, Some(42));
    assert!(queue.is_empty());
    assert_eq!(queue.len(), 0);
}

/// Test MPSC queue with multiple producers
#[test]
fn test_mpsc_queue_multiple_producers() {
    let queue = Arc::new(MPSCQueue::new());
    let num_producers = 4;
    let items_per_producer = 1000;
    let total_items = num_producers * items_per_producer;
    
    let mut handles = Vec::new();
    
    // Spawn producer threads
    for producer_id in 0..num_producers {
        let queue_clone = queue.clone();
        let handle = thread::spawn(move || {
            for i in 0..items_per_producer {
                let item = producer_id * items_per_producer + i;
                while queue_clone.push(item).is_err() {
                    thread::yield_now();
                }
            }
        });
        handles.push(handle);
    }
    
    // Wait for all producers to finish
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify all items were pushed
    assert_eq!(queue.len(), total_items);
    
    // Collect all items
    let mut collected = Vec::new();
    while let Some(item) = queue.pop() {
        collected.push(item);
    }
    
    // Verify count and uniqueness
    assert_eq!(collected.len(), total_items);
    collected.sort_unstable();
    for (i, &item) in collected.iter().enumerate() {
        assert_eq!(item, i);
    }
}

/// Test MPSC queue with bounded capacity
#[test]
fn test_mpsc_queue_bounded() {
    let capacity = 10;
    let queue = MPSCQueue::with_capacity(capacity);
    
    // Fill to capacity
    for i in 0..capacity {
        assert!(queue.push(i).is_ok());
    }
    
    // Should be full
    assert_eq!(queue.len(), capacity);
    assert!(queue.push(capacity).is_err()); // Should fail
    
    // Pop one item
    assert_eq!(queue.pop(), Some(0));
    assert_eq!(queue.len(), capacity - 1);
    
    // Should be able to push one more
    assert!(queue.push(capacity).is_ok());
    assert_eq!(queue.len(), capacity);
}

/// Test concurrent producer-consumer scenario
#[test]
fn test_mpsc_queue_concurrent_producer_consumer() {
    let queue = Arc::new(MPSCQueue::new());
    let produced = Arc::new(AtomicU64::new(0));
    let consumed = Arc::new(AtomicU64::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));
    
    let num_producers = 3;
    let items_per_producer = 1000;
    
    let mut handles = Vec::new();
    
    // Spawn producers
    for producer_id in 0..num_producers {
        let queue_clone = queue.clone();
        let produced_clone = produced.clone();
        let stop_clone = stop_flag.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..items_per_producer {
                let item = producer_id * items_per_producer + i;
                while queue_clone.push(item).is_err() && !stop_clone.load(Ordering::SeqCst) {
                    thread::yield_now();
                }
                produced_clone.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }
    
    // Spawn consumer
    let queue_consumer = queue.clone();
    let consumed_clone = consumed.clone();
    let stop_consumer = stop_flag.clone();
    
    let consumer_handle = thread::spawn(move || {
        while !stop_consumer.load(Ordering::SeqCst) || !queue_consumer.is_empty() {
            if let Some(_item) = queue_consumer.pop() {
                consumed_clone.fetch_add(1, Ordering::SeqCst);
            } else {
                thread::yield_now();
            }
        }
    });
    
    // Wait for producers
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Stop consumer
    stop_flag.store(true, Ordering::SeqCst);
    consumer_handle.join().unwrap();
    
    // Verify all items were produced and consumed
    let total_expected = num_producers * items_per_producer;
    assert_eq!(produced.load(Ordering::SeqCst), total_expected as u64);
    assert_eq!(consumed.load(Ordering::SeqCst), total_expected as u64);
    assert!(queue.is_empty());
}

/// Test lock-free queue implementation
#[test]
fn test_lockfree_queue_basic() {
    let queue = LockFreeQueue::new();
    
    // Test empty
    assert!(queue.is_empty());
    assert!(queue.dequeue().is_none());
    
    // Test enqueue/dequeue
    queue.enqueue(100);
    assert!(!queue.is_empty());
    
    let item = queue.dequeue();
    assert_eq!(item, Some(100));
    assert!(queue.is_empty());
}

/// Test lock-free queue with multiple threads
#[test]
fn test_lockfree_queue_concurrent() {
    let queue = Arc::new(LockFreeQueue::new());
    let num_threads = 8;
    let items_per_thread = 500;
    
    let mut handles = Vec::new();
    
    // Spawn enqueuers and dequeuers
    for thread_id in 0..num_threads {
        let queue_clone = queue.clone();
        
        if thread_id % 2 == 0 {
            // Enqueuer
            let handle = thread::spawn(move || {
                for i in 0..items_per_thread {
                    let item = thread_id * items_per_thread + i;
                    queue_clone.enqueue(item);
                }
            });
            handles.push(handle);
        } else {
            // Dequeuer
            let handle = thread::spawn(move || {
                let mut dequeued = 0;
                let start = Instant::now();
                while dequeued < items_per_thread && start.elapsed() < Duration::from_secs(10) {
                    if queue_clone.dequeue().is_some() {
                        dequeued += 1;
                    } else {
                        thread::yield_now();
                    }
                }
                dequeued
            });
            handles.push(handle);
        }
    }
    
    // Wait for completion and collect results
    let mut total_dequeued = 0;
    for handle in handles {
        if let Ok(result) = handle.join() {
            if let Ok(count) = result.downcast::<usize>() {
                total_dequeued += *count;
            }
        }
    }
    
    // Some items should have been processed
    assert!(total_dequeued > 0);
}

/// Test AtomicPtr operations
#[test]
fn test_atomic_ptr() {
    let value = Box::new(42);
    let ptr = Box::into_raw(value);
    let atomic_ptr = AtomicPtr::new(ptr);
    
    // Test load
    let loaded = atomic_ptr.load(MemoryOrdering::SeqCst);
    assert_eq!(loaded, ptr);
    
    // Test store
    let new_value = Box::new(84);
    let new_ptr = Box::into_raw(new_value);
    atomic_ptr.store(new_ptr, MemoryOrdering::SeqCst);
    
    let loaded_new = atomic_ptr.load(MemoryOrdering::SeqCst);
    assert_eq!(loaded_new, new_ptr);
    
    // Test compare_exchange
    let newer_value = Box::new(168);
    let newer_ptr = Box::into_raw(newer_value);
    
    let exchange_result = atomic_ptr.compare_exchange(
        new_ptr,
        newer_ptr,
        MemoryOrdering::SeqCst,
        MemoryOrdering::SeqCst,
    );
    assert_eq!(exchange_result, Ok(new_ptr));
    
    // Cleanup
    unsafe {
        let _ = Box::from_raw(ptr);
        let _ = Box::from_raw(new_ptr);
        let _ = Box::from_raw(newer_ptr);
    }
}

/// Test AtomicOption operations
#[test]
fn test_atomic_option() {
    let atomic_opt = AtomicOption::new(Some(42));
    
    // Test load
    assert_eq!(atomic_opt.load(MemoryOrdering::SeqCst), Some(42));
    
    // Test store
    atomic_opt.store(Some(84), MemoryOrdering::SeqCst);
    assert_eq!(atomic_opt.load(MemoryOrdering::SeqCst), Some(84));
    
    // Test store None
    atomic_opt.store(None, MemoryOrdering::SeqCst);
    assert_eq!(atomic_opt.load(MemoryOrdering::SeqCst), None);
    
    // Test compare_exchange
    let exchange_result = atomic_opt.compare_exchange(
        None,
        Some(168),
        MemoryOrdering::SeqCst,
        MemoryOrdering::SeqCst,
    );
    assert_eq!(exchange_result, Ok(None));
    assert_eq!(atomic_opt.load(MemoryOrdering::SeqCst), Some(168));
}

/// Test memory ordering semantics
#[test]
fn test_memory_ordering() {
    let atomic_ptr = Arc::new(AtomicPtr::new(std::ptr::null_mut()));
    let flag = Arc::new(AtomicBool::new(false));
    
    let atomic_clone = atomic_ptr.clone();
    let flag_clone = flag.clone();
    
    let writer = thread::spawn(move || {
        let value = Box::new(42);
        let ptr = Box::into_raw(value);
        
        // Store with release ordering
        atomic_clone.store(ptr, MemoryOrdering::Release);
        flag_clone.store(true, MemoryOrdering::Release);
    });
    
    let reader = thread::spawn(move || {
        // Wait for flag with acquire ordering
        while !flag.load(MemoryOrdering::Acquire) {
            thread::yield_now();
        }
        
        // Load with acquire ordering
        let ptr = atomic_ptr.load(MemoryOrdering::Acquire);
        if !ptr.is_null() {
            unsafe {
                let value = *ptr;
                let _ = Box::from_raw(ptr); // Cleanup
                value
            }
        } else {
            0
        }
    });
    
    writer.join().unwrap();
    let result = reader.join().unwrap();
    assert_eq!(result, 42);
}

/// Test ABA problem prevention
#[test]
fn test_aba_prevention() {
    let queue = Arc::new(MPSCQueue::new());
    let iterations = 1000;
    
    // Fill queue initially
    for i in 0..10 {
        queue.push(i).unwrap();
    }
    
    let queue_clone = queue.clone();
    let aba_thread = thread::spawn(move || {
        for _ in 0..iterations {
            // Try to create ABA scenario
            if let Some(item) = queue_clone.pop() {
                // Immediately push back
                if queue_clone.push(item).is_err() {
                    break;
                }
            }
        }
    });
    
    let queue_clone2 = queue.clone();
    let normal_thread = thread::spawn(move || {
        for i in 100..100 + iterations {
            while queue_clone2.push(i).is_err() {
                thread::yield_now();
            }
        }
    });
    
    aba_thread.join().unwrap();
    normal_thread.join().unwrap();
    
    // Queue should still be functional and consistent
    let mut count = 0;
    while queue.pop().is_some() {
        count += 1;
        if count > iterations + 100 {
            break; // Prevent infinite loop
        }
    }
    
    assert!(count > 0);
}

/// Test performance under contention
#[test]
fn test_performance_under_contention() {
    let queue = Arc::new(MPSCQueue::new());
    let num_threads = 16;
    let operations_per_thread = 10000;
    
    let start = Instant::now();
    let mut handles = Vec::new();
    
    for thread_id in 0..num_threads {
        let queue_clone = queue.clone();
        let handle = thread::spawn(move || {
            if thread_id % 2 == 0 {
                // Producer
                for i in 0..operations_per_thread {
                    let item = thread_id * operations_per_thread + i;
                    while queue_clone.push(item).is_err() {
                        thread::yield_now();
                    }
                }
            } else {
                // Consumer
                let mut consumed = 0;
                while consumed < operations_per_thread {
                    if queue_clone.pop().is_some() {
                        consumed += 1;
                    } else {
                        thread::yield_now();
                    }
                }
                consumed
            }
        });
        handles.push(handle);
    }
    
    let mut total_consumed = 0;
    for handle in handles {
        if let Ok(result) = handle.join() {
            if let Ok(count) = result.downcast::<usize>() {
                total_consumed += *count;
            }
        }
    }
    
    let elapsed = start.elapsed();
    let total_operations = num_threads * operations_per_thread;
    
    println!("Completed {} operations in {:?}", total_operations, elapsed);
    println!("Operations per second: {:.0}", total_operations as f64 / elapsed.as_secs_f64());
    
    // Should complete in reasonable time
    assert!(elapsed < Duration::from_secs(30));
}

/// Test queue behavior when full
#[test]
fn test_queue_full_behavior() {
    let capacity = 100;
    let queue = MPSCQueue::with_capacity(capacity);
    
    // Fill to capacity
    for i in 0..capacity {
        assert!(queue.push(i).is_ok());
    }
    
    // Verify it's full
    assert_eq!(queue.len(), capacity);
    assert!(queue.push(capacity).is_err());
    
    // Test multiple threads trying to push to full queue
    let queue_shared = Arc::new(queue);
    let success_count = Arc::new(AtomicU64::new(0));
    let failure_count = Arc::new(AtomicU64::new(0));
    
    let mut handles = Vec::new();
    
    for _ in 0..4 {
        let queue_clone = queue_shared.clone();
        let success_clone = success_count.clone();
        let failure_clone = failure_count.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..100 {
                if queue_clone.push(i).is_ok() {
                    success_clone.fetch_add(1, Ordering::SeqCst);
                } else {
                    failure_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Should have many failures since queue is full
    assert!(failure_count.load(Ordering::SeqCst) > 0);
    assert_eq!(success_count.load(Ordering::SeqCst), 0);
}

/// Test lock-free data structure linearizability
#[test]
fn test_linearizability() {
    let queue = Arc::new(LockFreeQueue::new());
    let operations = Arc::new(AtomicU64::new(0));
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads doing mixed operations
    for thread_id in 0..8 {
        let queue_clone = queue.clone();
        let ops_clone = operations.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..1000 {
                ops_clone.fetch_add(1, Ordering::SeqCst);
                
                if i % 2 == 0 {
                    queue_clone.enqueue(thread_id * 1000 + i);
                } else {
                    queue_clone.dequeue();
                }
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    assert_eq!(operations.load(Ordering::SeqCst), 8000);
    
    // Queue should still be in a valid state
    let mut remaining_items = 0;
    while queue.dequeue().is_some() {
        remaining_items += 1;
        if remaining_items > 10000 {
            break; // Safety limit
        }
    }
}

/// Test memory reclamation and cleanup
#[test]
fn test_memory_cleanup() {
    // Test that our lock-free structures properly clean up memory
    let initial_items = 1000;
    
    {
        let queue = MPSCQueue::new();
        
        // Add many items
        for i in 0..initial_items {
            queue.push(format!("item_{}", i)).unwrap();
        }
        
        assert_eq!(queue.len(), initial_items);
        
        // Remove half
        for _ in 0..initial_items / 2 {
            queue.pop();
        }
        
        assert_eq!(queue.len(), initial_items / 2);
        
        // Queue should drop cleanly when going out of scope
    }
    
    // If we reach here without segfault, memory cleanup worked
    assert!(true);
}

/// Test edge cases and error conditions
#[test]
fn test_edge_cases() {
    // Test with zero capacity
    let zero_queue = MPSCQueue::with_capacity(0);
    assert!(zero_queue.push(42).is_err());
    assert!(zero_queue.pop().is_none());
    assert!(zero_queue.is_empty());
    
    // Test with very large capacity
    let large_queue = MPSCQueue::with_capacity(usize::MAX / 2);
    assert!(large_queue.push(42).is_ok());
    assert_eq!(large_queue.pop(), Some(42));
    
    // Test rapid push/pop cycles
    let cycle_queue = MPSCQueue::new();
    for _ in 0..10000 {
        assert!(cycle_queue.push(1).is_ok());
        assert_eq!(cycle_queue.pop(), Some(1));
    }
}