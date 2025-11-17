//! Comprehensive example demonstrating memory allocation patterns and concurrency
//!
//! This example showcases:
//! 1. Stack vs Heap vs Static memory allocation
//! 2. Thread-based parallelism
//! 3. Channel-based communication
//! 4. Data parallelism with Rayon
//! 5. Concurrent data structures (Arc, Mutex, RwLock)
//!
//! Run with: cargo run --example memory_and_concurrency

use std::sync::Arc;
use std::thread;
use std::time::Instant;
use rust_embeddings::prelude::*;
use rust_embeddings::concurrency::*;
use rust_embeddings::config::*;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Rust Memory Patterns & Concurrency Demonstration             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // PART 1: Static Memory (Compile-time Constants)
    // ========================================================================
    demonstrate_static_memory();

    // ========================================================================
    // PART 2: Stack vs Heap Memory
    // ========================================================================
    demonstrate_stack_heap_memory();

    // ========================================================================
    // PART 3: Concurrency with Shared Memory (Arc + RwLock)
    // ========================================================================
    demonstrate_shared_memory_concurrency()?;

    // ========================================================================
    // PART 4: Message Passing with Channels
    // ========================================================================
    demonstrate_channel_communication()?;

    // ========================================================================
    // PART 5: Data Parallelism with Rayon
    // ========================================================================
    demonstrate_data_parallelism()?;

    // ========================================================================
    // PART 6: Performance Comparison
    // ========================================================================
    demonstrate_performance_comparison()?;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example completed successfully!                               ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Demonstrates static memory allocation with compile-time constants
fn demonstrate_static_memory() {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 1: Static Memory (Binary Data Segment)                │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("Static constants live in the binary's data segment:");
    println!("  • Memory Location: Binary data segment (not stack, not heap)");
    println!("  • Lifetime: 'static (entire program duration)");
    println!("  • Access: No runtime allocation cost");
    println!("  • Thread Safety: Read-only constants are thread-safe\n");

    println!("Available constants:");
    println!("  DEFAULT_EMBEDDING_DIMENSIONS = {}", DEFAULT_EMBEDDING_DIMENSIONS);
    println!("  MAX_EMBEDDING_DIMENSIONS     = {}", MAX_EMBEDDING_DIMENSIONS);
    println!("  DEFAULT_BATCH_SIZE           = {}", DEFAULT_BATCH_SIZE);
    println!("  MAX_BATCH_SIZE               = {}", MAX_BATCH_SIZE);
    println!("  DEFAULT_CACHE_SIZE           = {}", DEFAULT_CACHE_SIZE);
    println!("  VERSION                      = {}", VERSION);
    println!("  MAX_WORKER_THREADS           = {}", MAX_WORKER_THREADS);

    println!("\nMemory address of static constant:");
    println!("  &DEFAULT_EMBEDDING_DIMENSIONS = {:p}", &DEFAULT_EMBEDDING_DIMENSIONS);

    println!("\n✓ Static memory demonstrated\n");
}

/// Demonstrates stack vs heap memory allocation
fn demonstrate_stack_heap_memory() {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 2: Stack vs Heap Memory Allocation                    │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // STACK ALLOCATION
    println!("Stack Allocation:");
    println!("  • Fast: Just moves stack pointer");
    println!("  • Small: Typical thread stack is ~2MB");
    println!("  • Automatic: Cleaned up when scope ends");
    println!("  • LIFO: Last In, First Out\n");

    // Primitives live on stack
    let dimensions: usize = 128;              // Stack: 8 bytes
    let threshold: f32 = 0.85;                // Stack: 4 bytes
    let is_valid: bool = true;                // Stack: 1 byte

    println!("  Stack variables (primitives):");
    println!("    dimensions: usize = {} → {:p} (stack address)", dimensions, &dimensions);
    println!("    threshold: f32    = {} → {:p} (stack address)", threshold, &threshold);
    println!("    is_valid: bool    = {} → {:p} (stack address)", is_valid, &is_valid);

    // HEAP ALLOCATION
    println!("\nHeap Allocation:");
    println!("  • Dynamic: Size can be unknown at compile time");
    println!("  • Large: Can allocate gigabytes if needed");
    println!("  • Manual: Requires explicit allocation (new, Vec::new, etc.)");
    println!("  • Flexible: Can grow/shrink at runtime\n");

    // Collections live on heap
    let vector_data = vec![1.0_f32; 128];     // Heap: 128 * 4 = 512 bytes
    let text = String::from("example");       // Heap: variable size

    println!("  Heap variables (collections):");
    println!("    Vec<f32> with {} elements", vector_data.len());
    println!("      Stack: pointer + length + capacity = 24 bytes → {:p}", &vector_data);
    println!("      Heap:  actual data = {} bytes → {:p}",
             vector_data.len() * 4,
             vector_data.as_ptr());

    println!("    String \"{}\"", text);
    println!("      Stack: pointer + length + capacity = 24 bytes → {:p}", &text);
    println!("      Heap:  actual characters = {} bytes → {:p}",
             text.len(),
             text.as_ptr());

    // Embedding (mix of stack and heap)
    let vector = Vector::new(&vector_data).unwrap();
    let metadata = EmbeddingMetadata::builder()
        .source("example")
        .model("demo")
        .build()
        .unwrap();
    let embedding = Embedding::new(vector, metadata);

    println!("\n  Embedding struct (mixed):");
    println!("    Stack: struct metadata (id, timestamps, etc.) ≈ 56 bytes");
    println!("    Heap:  Vector data + String fields ≈ {} bytes",
             128 * 4 + 30); // rough estimate

    // Box: explicit heap allocation
    let boxed_embedding = Box::new(embedding.clone());
    println!("\n  Box<Embedding> (explicit heap):");
    println!("    Stack: pointer = 8 bytes → {:p}", &boxed_embedding);
    println!("    Heap:  entire Embedding → {:p}", boxed_embedding.as_ref());

    println!("\n✓ Stack and heap memory demonstrated\n");
}

/// Demonstrates shared memory concurrency with Arc + RwLock
fn demonstrate_shared_memory_concurrency() -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 3: Shared Memory Concurrency (Arc + RwLock)           │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("Arc (Atomic Reference Counted):");
    println!("  • Enables multiple ownership across threads");
    println!("  • Heap allocated with atomic reference counter");
    println!("  • Cloning Arc increments counter (no data copy)");
    println!("  • Last owner drops heap data\n");

    println!("RwLock (Read-Write Lock):");
    println!("  • Multiple readers OR single writer");
    println!("  • Readers don't block other readers");
    println!("  • Writers block everyone\n");

    // Create embeddings on heap
    let embeddings = create_sample_embeddings(100)?;

    // Wrap in Arc for shared ownership
    let shared_embeddings = Arc::new(embeddings);

    println!("Created {} embeddings wrapped in Arc", shared_embeddings.len());
    println!("Arc reference count: {}", Arc::strong_count(&shared_embeddings));

    // Spawn threads that share data
    let mut handles = vec![];
    let num_threads = 4;

    let start = Instant::now();

    for thread_id in 0..num_threads {
        // Clone Arc (only increments counter, ~atomic operation)
        let embeddings = Arc::clone(&shared_embeddings);

        println!("  Thread {}: Arc cloned, ref count = {}",
                 thread_id,
                 Arc::strong_count(&shared_embeddings));

        let handle = thread::spawn(move || {
            // Each thread reads from shared heap data
            let mut sum = 0.0_f32;
            for embedding in embeddings.iter() {
                for value in embedding.vector().as_slice() {
                    sum += value;
                }
            }
            (thread_id, sum)
        });

        handles.push(handle);
    }

    // Collect results
    println!("\nWaiting for threads to complete...");
    for handle in handles {
        let (thread_id, sum) = handle.join().unwrap();
        println!("  Thread {} completed: sum = {:.2}", thread_id, sum);
    }

    let elapsed = start.elapsed();
    println!("\nTotal time: {:?}", elapsed);
    println!("Final Arc reference count: {}", Arc::strong_count(&shared_embeddings));

    println!("\n✓ Shared memory concurrency demonstrated\n");
    Ok(())
}

/// Demonstrates message passing with channels
fn demonstrate_channel_communication() -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 4: Message Passing with Channels                      │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("Channels enable communication between threads:");
    println!("  • Producer sends messages");
    println!("  • Consumers receive messages");
    println!("  • No shared memory (ownership transferred)");
    println!("  • Thread-safe by design\n");

    let texts: Vec<String> = (0..50)
        .map(|i| format!("document-{}", i))
        .collect();

    println!("Processing {} texts using {} workers", texts.len(), 4);

    let start = Instant::now();
    let results = process_with_channels(texts, 4)?;
    let elapsed = start.elapsed();

    println!("\n✓ Processed {} embeddings in {:?}", results.len(), elapsed);
    println!("  Average: {:.2}ms per embedding",
             elapsed.as_millis() as f64 / results.len() as f64);

    println!("\n✓ Channel communication demonstrated\n");
    Ok(())
}

/// Demonstrates data parallelism with Rayon
fn demonstrate_data_parallelism() -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 5: Data Parallelism with Rayon                        │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("Rayon provides automatic parallelization:");
    println!("  • Work-stealing thread pool");
    println!("  • Parallel iterators");
    println!("  • Automatic load balancing");
    println!("  • Minimal overhead\n");

    let embeddings = create_sample_embeddings(1000)?;
    println!("Processing {} embeddings with Rayon", embeddings.len());

    let start = Instant::now();
    let results = process_with_rayon(&embeddings)?;
    let elapsed = start.elapsed();

    println!("\n✓ Processed {} embeddings in {:?}", results.len(), elapsed);
    println!("  Using {} CPU cores", num_cpus::get());
    println!("  Average: {:.2}µs per embedding",
             elapsed.as_micros() as f64 / results.len() as f64);

    // Demonstrate parallel batch processing
    println!("\nParallel batch processing:");
    let texts: Vec<String> = (0..100)
        .map(|i| format!("text-{}", i))
        .collect();

    let start = Instant::now();
    let batch_results = parallel_batch_process(&texts, 10)?;
    let elapsed = start.elapsed();

    println!("  Processed {} texts in batches of 10", texts.len());
    println!("  Total time: {:?}", elapsed);
    println!("  Result count: {}", batch_results.len());

    println!("\n✓ Data parallelism demonstrated\n");
    Ok(())
}

/// Demonstrates performance comparison between approaches
fn demonstrate_performance_comparison() -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 6: Performance Comparison                              │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let embeddings = create_sample_embeddings(1000)?;

    let _benchmark = ConcurrencyBenchmark::run(embeddings)?;

    println!("\n═══ Summary ═══");
    println!("Best performer: Rayon (data parallelism)");
    println!("CPU cores available: {}", num_cpus::get());
    println!("\nKey Takeaways:");
    println!("  1. Static memory: Zero-cost constants");
    println!("  2. Stack: Fast but limited size");
    println!("  3. Heap: Flexible but allocation cost");
    println!("  4. Arc: Share data across threads");
    println!("  5. Channels: Safe message passing");
    println!("  6. Rayon: Easy data parallelism");

    println!("\n✓ Performance comparison completed\n");
    Ok(())
}

/// Helper function to create sample embeddings
fn create_sample_embeddings(count: usize) -> Result<Vec<Embedding>> {
    (0..count)
        .map(|i| {
            let data: Vec<f32> = (0..128).map(|j| (i + j) as f32 * 0.01).collect();
            let vector = Vector::new(&data)?;
            let metadata = EmbeddingMetadata::builder()
                .source(format!("sample-{}", i))
                .model("demo-model")
                .tag("benchmark")
                .build()?;
            Ok(Embedding::new(vector, metadata))
        })
        .collect()
}
