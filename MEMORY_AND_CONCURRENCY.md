# Memory Allocation & Concurrency Patterns in Rust

This document explains how this codebase demonstrates Rust's memory allocation patterns (stack, heap, static) and concurrency/parallelism features.

## Table of Contents

1. [Memory Allocation Patterns](#memory-allocation-patterns)
2. [Concurrency Patterns](#concurrency-patterns)
3. [Where to Find Examples](#where-to-find-examples)
4. [Performance Characteristics](#performance-characteristics)

---

## Memory Allocation Patterns

### 1. Static Memory

**Location:** Binary's data segment (`.data` or `.rodata` section)

**Characteristics:**
- Lifetime: `'static` (entire program duration)
- Known at compile time
- Zero runtime allocation cost
- Thread-safe for read-only constants

**Examples in Codebase:**

```rust
// src/config.rs
pub const DEFAULT_EMBEDDING_DIMENSIONS: usize = 384;
pub const MAX_EMBEDDING_DIMENSIONS: usize = 4096;
pub const DEFAULT_BATCH_SIZE: usize = 32;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VALID_LOG_LEVELS: &[&str] = &["trace", "debug", "info", "warn", "error"];
```

**When to Use:**
- Configuration constants
- Lookup tables
- Error messages
- Version strings

### 2. Stack Memory

**Location:** Thread's stack (typically 2MB on Linux)

**Characteristics:**
- Very fast allocation (just moving stack pointer)
- Automatic cleanup (LIFO - Last In, First Out)
- Fixed size, known at compile time
- Each thread has its own stack
- Limited size (stack overflow risk)

**Examples in Codebase:**

```rust
// Primitives
let dimensions: usize = 128;        // Stack: 8 bytes
let threshold: f32 = 0.85;          // Stack: 4 bytes
let is_valid: bool = true;          // Stack: 1 byte

// References (just pointers)
fn process_embedding(emb: &Embedding) {
    // emb is just a pointer (8 bytes on stack)
}

// Local variables in loops
for i in 0..100 {
    let sum = 0.0_f32;  // Stack allocated, freed each iteration
}
```

**When to Use:**
- Small, fixed-size data
- Function parameters
- Local variables
- References and pointers

### 3. Heap Memory

**Location:** Process heap (can grow to gigabytes)

**Characteristics:**
- Dynamic sizing (size can be unknown at compile time)
- Manual allocation (`Box::new`, `Vec::new`, `String`, etc.)
- Slower than stack (requires allocator)
- Can be shared between threads (with proper synchronization)
- Can cause fragmentation

**Examples in Codebase:**

```rust
// Collections
let vector_data = vec![1.0_f32; 128];  // Heap: 512 bytes
let text = String::from("example");     // Heap: variable size

// Domain entities
// src/domain/entities.rs
pub struct Embedding {
    vector: Vector,              // Contains heap-allocated Array1<f32>
    metadata: EmbeddingMetadata, // Contains String and Vec (heap)
    // ... stack-allocated fields
}

// Explicit heap allocation
let boxed = Box::new(embedding);  // Forces heap allocation

// Arc for shared ownership
let shared = Arc::new(embeddings); // Heap + atomic ref count
```

**When to Use:**
- Collections (Vec, HashMap, String)
- Large data structures
- Data with unknown size at compile time
- Shared data between threads

---

## Concurrency Patterns

### 1. Thread-based Parallelism (std::thread)

**Description:** Spawn OS threads for true parallelism

**Memory Pattern:**
- Each thread gets its own stack (~2MB)
- Shared data must be on heap (Arc)
- `Arc` = Atomic Reference Counting

**Example:**

```rust
// src/concurrency.rs:process_with_threads
let shared_data = Arc::new(embeddings.to_vec()); // Heap allocation

for thread_id in 0..num_threads {
    let data = Arc::clone(&shared_data);  // Only increments ref count

    thread::spawn(move || {
        // This closure runs on a new OS thread with its own stack
        let mut sum = 0.0_f32;  // Stack allocated on this thread
        for embedding in &data[..] {
            // Access shared heap data
        }
        sum
    });
}
```

**Use Cases:**
- CPU-bound tasks
- Need precise control over threads
- Long-running background tasks

### 2. Channel-based Communication (crossbeam::channel)

**Description:** Message passing between threads (no shared memory)

**Memory Pattern:**
- Channel is heap-allocated queue
- Messages transfer ownership (move semantics)
- No shared memory = no synchronization overhead

**Example:**

```rust
// src/concurrency.rs:process_with_channels
let (sender, receiver) = channel::bounded::<String>(100);

// Producer thread
for text in texts {
    sender.send(text)?;  // Ownership transferred through channel
}

// Consumer threads
thread::spawn(move || {
    while let Ok(text) = receiver.recv() {
        // Process text (now owned by this thread)
    }
});
```

**Use Cases:**
- Producer-consumer patterns
- Pipeline architectures
- Event-driven systems
- When you want to avoid shared state

### 3. Data Parallelism (Rayon)

**Description:** Automatic parallelization via parallel iterators

**Memory Pattern:**
- Work-stealing thread pool (heap-allocated)
- Minimal overhead
- Automatic load balancing

**Example:**

```rust
// src/concurrency.rs:process_with_rayon
use rayon::prelude::*;

let results: Vec<f32> = embeddings
    .par_iter()  // Parallel iterator
    .map(|embedding| {
        // This runs on thread pool workers
        embedding.vector().as_slice().iter().sum()
    })
    .collect();
```

**Use Cases:**
- Batch processing
- CPU-bound operations on collections
- When you want easy parallelism
- Map-reduce style operations

### 4. Shared State with RwLock

**Description:** Multiple readers OR single writer

**Memory Pattern:**
- `RwLock<T>` wraps heap data
- Readers don't block other readers
- Writers get exclusive access

**Example:**

```rust
// src/infrastructure/repository.rs
pub struct InMemoryEmbeddingRepository {
    storage: RwLock<HashMap<EmbeddingId, Embedding>>,  // Heap
}

// Multiple readers can access concurrently
let storage = self.storage.read().unwrap();
let value = storage.get(key);

// Writer gets exclusive lock
let mut storage = self.storage.write().unwrap();
storage.insert(key, value);
```

**Use Cases:**
- Read-heavy workloads
- Shared caches
- Configuration that rarely changes

### 5. Async/Await (Tokio)

**Description:** Concurrent I/O without OS threads

**Memory Pattern:**
- Small stack frames (state machines on heap)
- Many tasks can run on few threads
- Efficient for I/O-bound work

**Example:**

```rust
// src/application/services.rs
pub async fn create_from_text(&self, text: &str) -> Result<Embedding> {
    let embedding = self.generator.generate_from_text(text).await?;
    self.repository.save(&embedding).await?;
    self.vector_store.index(&embedding).await?;
    Ok(embedding)
}
```

**Use Cases:**
- I/O-bound operations
- Network requests
- Database queries
- File operations

---

## Where to Find Examples

### Static Memory
- **File:** `src/config.rs`
- **Lines:** 29-67
- **Constants:** `DEFAULT_EMBEDDING_DIMENSIONS`, `MAX_BATCH_SIZE`, etc.

### Stack Memory
- **Everywhere:** Function parameters, local variables
- **Example:** `src/concurrency.rs:73` - local `sum` variable in thread

### Heap Memory
- **Entities:** `src/domain/entities.rs` - `Embedding`, `Vector`
- **Collections:** `src/domain/entities.rs:202-203` - `EmbeddingCollection`
- **Repository:** `src/infrastructure/repository.rs:15` - `HashMap` storage

### Thread-based Parallelism
- **File:** `src/concurrency.rs`
- **Function:** `process_with_threads` (lines 62-109)

### Channel Communication
- **File:** `src/concurrency.rs`
- **Function:** `process_with_channels` (lines 125-197)

### Rayon Data Parallelism
- **File:** `src/concurrency.rs`
- **Function:** `process_with_rayon` (lines 217-232)
- **Function:** `parallel_batch_process` (lines 237-257)

### RwLock (Reader-Writer Lock)
- **File:** `src/infrastructure/repository.rs`
- **Struct:** `InMemoryEmbeddingRepository` (lines 14-138)
- **Usage:** Lines 36-37 (write), 44-45 (read)

### Arc (Atomic Reference Counting)
- **File:** `src/application/services.rs`
- **Lines:** 19-23 - Service dependencies

### Comprehensive Example
- **File:** `examples/memory_and_concurrency.rs`
- **Run:** `cargo run --example memory_and_concurrency`

---

## Performance Characteristics

### Memory Allocation Performance

| Type | Allocation Speed | Size Limit | Cleanup |
|------|-----------------|------------|---------|
| **Static** | Compile-time (zero cost) | Program size | Never (static lifetime) |
| **Stack** | Very fast (~ns) | ~2MB per thread | Automatic (scope) |
| **Heap** | Moderate (~100ns) | Available RAM | Manual/RAII |

### Concurrency Performance

Based on benchmarks in `src/concurrency.rs`:

| Approach | Best For | Overhead | Scalability |
|----------|----------|----------|-------------|
| **Sequential** | Baseline | None | 1x |
| **Threads** | CPU-bound | Thread creation (~µs) | # of CPU cores |
| **Channels** | Pipeline | Message passing | # of CPU cores |
| **Rayon** | Data parallel | Work stealing (low) | # of CPU cores |
| **Async** | I/O-bound | Task switching (ns) | Thousands of tasks |

### When to Use What

```
CPU-bound + Known parallelism → Explicit threads
CPU-bound + Collection → Rayon
I/O-bound → Async/await
Producer-consumer → Channels
Read-heavy shared state → RwLock
Write-heavy shared state → Mutex or channels
```

---

## Running Examples

```bash
# Run the comprehensive example
cargo run --example memory_and_concurrency

# Run tests for concurrency module
cargo test concurrency

# Run with logging
RUST_LOG=debug cargo run --example memory_and_concurrency

# Build in release mode for accurate benchmarks
cargo run --release --example memory_and_concurrency
```

---

## Further Reading

- [The Rust Programming Language - Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
- [The Rust Programming Language - Concurrency](https://doc.rust-lang.org/book/ch16-00-concurrency.html)
- [Rayon Documentation](https://docs.rs/rayon/)
- [Tokio Documentation](https://docs.rs/tokio/)
- [Crossbeam Documentation](https://docs.rs/crossbeam/)
