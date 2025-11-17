//! Concurrency and Parallelism Module
//!
//! This module demonstrates different concurrency and parallelism patterns in Rust,
//! showing how they interact with memory (stack, heap, and static).
//!
//! ## Concurrency vs Parallelism
//!
//! - **Concurrency**: Multiple tasks making progress (not necessarily simultaneously)
//! - **Parallelism**: Multiple tasks executing simultaneously on different CPU cores
//!
//! ## Memory Patterns in Concurrent Code
//!
//! ### Stack Memory with Threads
//! - Each thread gets its own stack (typically 2MB on Linux)
//! - Local variables live on each thread's stack
//! - Fast but limited in size
//!
//! ### Heap Memory with Threads
//! - Shared data must be on the heap (via Arc, Box, etc.)
//! - Allows data sharing between threads
//! - Requires synchronization (Mutex, RwLock, atomic operations)
//!
//! ### Static Memory
//! - Accessible from all threads without synchronization
//! - Read-only constants are thread-safe by default
//! - Mutable statics require `unsafe` or atomic operations

use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use crossbeam::channel;
use rayon::prelude::*;
use crate::domain::{Embedding, Vector, EmbeddingMetadata, EmbeddingCollection};
use crate::error::Result;

// ============================================================================
// STATIC MEMORY: Thread-safe constants
// ============================================================================

/// Maximum number of worker threads (Static Memory)
/// Safe to access from any thread without synchronization
pub const MAX_WORKER_THREADS: usize = 16;

/// Default worker thread count (Static Memory)
pub const DEFAULT_WORKER_THREADS: usize = 4;

// ============================================================================
// Explicit Thread-based Parallelism
// ============================================================================

/// Process embeddings using explicit threads (OS-level parallelism)
///
/// ## Memory Allocation Pattern:
/// - `embeddings` parameter: Passed as reference (stack pointer)
/// - `Arc<Vec<Embedding>>`: Heap allocation for shared ownership
/// - Thread handles: Stack allocated in this function
/// - Cloned Arc: Only increments reference count (no data copy)
///
/// ## Concurrency Pattern:
/// - Spawns multiple OS threads
/// - Each thread processes a chunk of data
/// - Uses Arc for shared read-only access (heap memory)
/// - Join all threads before returning
pub fn process_with_threads(
    embeddings: &[Embedding],
    num_threads: usize,
) -> Result<Vec<f32>> {
    let num_threads = num_threads.min(MAX_WORKER_THREADS).max(1);
    let chunk_size = (embeddings.len() + num_threads - 1) / num_threads;

    // Clone embeddings to heap and wrap in Arc for sharing between threads
    // Arc = Atomic Reference Counted pointer (heap allocation)
    let shared_data = Arc::new(embeddings.to_vec());

    // Vector to store thread handles (stack allocated)
    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        // Clone Arc (only increments reference count, doesn't copy data)
        let data = Arc::clone(&shared_data);

        // Spawn OS thread - gets its own stack (typically 2MB)
        let handle = thread::spawn(move || {
            let start = thread_id * chunk_size;
            let end = (start + chunk_size).min(data.len());

            // Process chunk - all computation happens on this thread's stack
            let mut local_sum = 0.0_f32; // Stack allocated

            for embedding in &data[start..end] {
                // Access heap data (embedding) but accumulate on stack (local_sum)
                for value in embedding.vector().as_slice() {
                    local_sum += value;
                }
            }

            local_sum // Return value moves to calling thread
        });

        handles.push(handle);
    }

    // Collect results from all threads
    let mut results = Vec::with_capacity(num_threads);
    for handle in handles {
        let result = handle.join().map_err(|_| {
            crate::error::EmbeddingError::processing("Thread panicked")
        })?;
        results.push(result);
    }

    Ok(results)
}

// ============================================================================
// Channel-based Communication (Message Passing)
// ============================================================================

/// Process embeddings using channels for thread communication
///
/// ## Memory Allocation Pattern:
/// - Channel: Heap allocated queue for message passing
/// - Messages: Moved through channel (ownership transfer)
/// - Sender/Receiver: Stack allocated handles to heap structure
///
/// ## Concurrency Pattern:
/// - Producer thread generates work items
/// - Multiple consumer threads process items
/// - Channel handles synchronization automatically
/// - No shared memory (message passing instead)
pub fn process_with_channels(
    texts: Vec<String>,
    num_workers: usize,
) -> Result<Vec<Embedding>> {
    let num_workers = num_workers.min(MAX_WORKER_THREADS).max(1);

    // Create bounded channel (heap allocated queue)
    // Channel capacity prevents unbounded memory growth
    let (sender, receiver) = channel::bounded::<String>(100);

    // Shared receiver for work distribution (Arc for thread-safe sharing)
    let receiver = Arc::new(receiver);

    // Shared result collection (Mutex for exclusive write access)
    // Mutex<Vec<T>> lives on heap
    let results = Arc::new(Mutex::new(Vec::new()));

    // Spawn worker threads
    let mut workers = Vec::with_capacity(num_workers);

    for worker_id in 0..num_workers {
        let receiver = Arc::clone(&receiver);
        let results = Arc::clone(&results);

        let handle = thread::spawn(move || {
            // Worker loop - each thread has its own stack
            loop {
                match receiver.recv() {
                    Ok(text) => {
                        // Process on this thread's stack
                        let vector_data: Vec<f32> = (0..128)
                            .map(|i| (text.len() as f32 + i as f32) * 0.01)
                            .collect();

                        let vector = Vector::new(&vector_data).unwrap();
                        let metadata = EmbeddingMetadata::builder()
                            .source(text)
                            .model(format!("worker-{}", worker_id))
                            .build()
                            .unwrap();

                        let embedding = Embedding::new(vector, metadata);

                        // Lock mutex to write result (heap access)
                        let mut results = results.lock().unwrap();
                        results.push(embedding);
                    }
                    Err(_) => break, // Channel closed, exit loop
                }
            }
        });

        workers.push(handle);
    }

    // Producer: Send work items through channel
    for text in texts {
        sender.send(text).map_err(|_| {
            crate::error::EmbeddingError::processing("Failed to send work item")
        })?;
    }

    // Close sender to signal workers to finish
    drop(sender);

    // Wait for all workers to complete
    for handle in workers {
        handle.join().map_err(|_| {
            crate::error::EmbeddingError::processing("Worker thread panicked")
        })?;
    }

    // Extract results from Arc<Mutex<Vec>>
    let results = Arc::try_unwrap(results)
        .map_err(|_| crate::error::EmbeddingError::processing("Arc still has references"))?
        .into_inner()
        .map_err(|_| crate::error::EmbeddingError::processing("Mutex poisoned"))?;

    Ok(results)
}

// ============================================================================
// Rayon-based Data Parallelism
// ============================================================================

/// Process embeddings using Rayon's data parallelism (Work Stealing)
///
/// ## Memory Allocation Pattern:
/// - Rayon uses a global thread pool (heap allocated)
/// - Work items distributed via work-stealing algorithm
/// - Minimal overhead compared to explicit threads
///
/// ## Concurrency Pattern:
/// - Automatic parallelization via parallel iterators
/// - Work stealing for load balancing
/// - Uses all available CPU cores by default
/// - Suitable for CPU-bound operations
pub fn process_with_rayon(embeddings: &[Embedding]) -> Result<Vec<f32>> {
    // Parallel iterator - Rayon automatically distributes work
    let results: Vec<f32> = embeddings
        .par_iter() // Parallel iterator
        .map(|embedding| {
            // This closure runs on thread pool worker threads
            // Each thread has its own stack for local variables
            let mut sum = 0.0_f32;
            for value in embedding.vector().as_slice() {
                sum += value;
            }
            sum
        })
        .collect(); // Collect results back to single vector

    Ok(results)
}

/// Parallel batch processing with Rayon
///
/// Processes large batches of text using data parallelism
pub fn parallel_batch_process(
    texts: &[String],
    batch_size: usize,
) -> Result<Vec<Embedding>> {
    texts
        .par_chunks(batch_size) // Split into parallel chunks
        .map(|chunk| {
            // Each chunk processed on a thread pool worker
            chunk
                .iter()
                .map(|text| {
                    let vector_data: Vec<f32> = (0..128)
                        .map(|i| (text.len() as f32 + i as f32) * 0.01)
                        .collect();

                    let vector = Vector::new(&vector_data)?;
                    let metadata = EmbeddingMetadata::builder()
                        .source(text.clone())
                        .model("rayon-parallel")
                        .build()?;

                    Ok(Embedding::new(vector, metadata))
                })
                .collect::<Result<Vec<Embedding>>>()
        })
        .try_reduce(
            || Vec::new(),
            |mut acc, chunk| {
                acc.extend(chunk);
                Ok(acc)
            },
        )
}

// ============================================================================
// RwLock for Reader-Writer Concurrency
// ============================================================================

/// Thread-safe cache using RwLock (Multiple readers OR single writer)
///
/// ## Memory Allocation Pattern:
/// - RwLock<HashMap>: Heap allocated
/// - Keys and values: Heap allocated
///
/// ## Concurrency Pattern:
/// - Multiple threads can read simultaneously
/// - Only one thread can write at a time
/// - Readers block writers, writers block everyone
pub struct ThreadSafeCache {
    cache: RwLock<std::collections::HashMap<String, Embedding>>,
}

impl ThreadSafeCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Read from cache (multiple readers allowed)
    pub fn get(&self, key: &str) -> Option<Embedding> {
        // Acquire read lock (doesn't block other readers)
        let cache = self.cache.read().unwrap();
        cache.get(key).cloned()
    }

    /// Write to cache (exclusive access)
    pub fn insert(&self, key: String, value: Embedding) {
        // Acquire write lock (blocks all readers and writers)
        let mut cache = self.cache.write().unwrap();
        cache.insert(key, value);
    }

    /// Get cache size (read operation)
    pub fn len(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// Performance Comparison Utilities
// ============================================================================

/// Compare performance of different concurrency approaches
pub struct ConcurrencyBenchmark {
    pub sequential_time: Duration,
    pub threads_time: Duration,
    pub channels_time: Duration,
    pub rayon_time: Duration,
}

impl ConcurrencyBenchmark {
    pub fn run(embeddings: Vec<Embedding>) -> Result<Self> {
        println!("\n=== Concurrency Benchmark ===");
        println!("Dataset size: {} embeddings\n", embeddings.len());

        // Sequential baseline
        let start = Instant::now();
        let _seq_results: Vec<f32> = embeddings
            .iter()
            .map(|e| e.vector().as_slice().iter().sum())
            .collect();
        let sequential_time = start.elapsed();
        println!("Sequential: {:?}", sequential_time);

        // Explicit threads
        let start = Instant::now();
        let _thread_results = process_with_threads(&embeddings, num_cpus::get())?;
        let threads_time = start.elapsed();
        println!("Threads ({}): {:?}", num_cpus::get(), threads_time);

        // Channels (using text dummy data for demo)
        let texts: Vec<String> = (0..embeddings.len())
            .map(|i| format!("text-{}", i))
            .collect();
        let start = Instant::now();
        let _channel_results = process_with_channels(texts, num_cpus::get())?;
        let channels_time = start.elapsed();
        println!("Channels ({}): {:?}", num_cpus::get(), channels_time);

        // Rayon
        let start = Instant::now();
        let _rayon_results = process_with_rayon(&embeddings)?;
        let rayon_time = start.elapsed();
        println!("Rayon: {:?}", rayon_time);

        println!("\n=== Speedups vs Sequential ===");
        println!("Threads: {:.2}x", sequential_time.as_secs_f64() / threads_time.as_secs_f64());
        println!("Channels: {:.2}x", sequential_time.as_secs_f64() / channels_time.as_secs_f64());
        println!("Rayon: {:.2}x", sequential_time.as_secs_f64() / rayon_time.as_secs_f64());

        Ok(Self {
            sequential_time,
            threads_time,
            channels_time,
            rayon_time,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_embeddings(count: usize) -> Vec<Embedding> {
        (0..count)
            .map(|i| {
                let vector = Vector::new(&vec![i as f32; 128]).unwrap();
                let metadata = EmbeddingMetadata::builder()
                    .source(format!("test-{}", i))
                    .model("test")
                    .build()
                    .unwrap();
                Embedding::new(vector, metadata)
            })
            .collect()
    }

    #[test]
    fn test_process_with_threads() {
        let embeddings = create_test_embeddings(100);
        let results = process_with_threads(&embeddings, 4).unwrap();
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_process_with_channels() {
        let texts: Vec<String> = (0..100).map(|i| format!("text-{}", i)).collect();
        let results = process_with_channels(texts, 4).unwrap();
        assert_eq!(results.len(), 100);
    }

    #[test]
    fn test_process_with_rayon() {
        let embeddings = create_test_embeddings(100);
        let results = process_with_rayon(&embeddings).unwrap();
        assert_eq!(results.len(), 100);
    }

    #[test]
    fn test_thread_safe_cache() {
        let cache = Arc::new(ThreadSafeCache::new());
        let mut handles = vec![];

        // Spawn multiple threads accessing cache
        for i in 0..10 {
            let cache = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                let embedding = create_test_embeddings(1).pop().unwrap();
                cache.insert(format!("key-{}", i), embedding);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(cache.len(), 10);
    }
}
