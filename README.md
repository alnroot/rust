# Rust Embeddings Library

A well-architected Rust library for managing embeddings with clean code and software design patterns. Production-ready with ONNX inference, SQLite persistence, and HNSW approximate nearest neighbor search.

## Features

- **Clean Architecture**: Layered separation (Domain, Application, Infrastructure, Ports)
- **SOLID Principles**: Maintainable and extensible code
- **Design Patterns**: Repository, Strategy, Factory, Builder, Dependency Injection
- **Production Storage**: SQLite with ACID compliance
- **Fast ANN Search**: HNSW index for O(log n) similarity search
- **Local Inference**: ONNX Runtime for embedding generation (all-MiniLM-L6-v2)
- **Concurrency**: Threads, Channels, Rayon, Arc, RwLock
- **Async/Await**: Full async support with Tokio
- **Type-Safe**: Leverages Rust's type system
- **Well Tested**: 32 unit tests passing

## Architecture

The library follows **Hexagonal Architecture** (Ports and Adapters):

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│  (Use Cases, Services, Orchestration)                    │
└────────────────┬────────────────────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
┌────────▼──────┐  ┌────▼──────────┐
│  Domain Layer │  │  Ports Layer   │
│  (Entities,   │  │  (Interfaces,  │
│   Values)     │  │   Abstractions)│
└───────────────┘  └────┬───────────┘
                        │
              ┌─────────▼────────────┐
              │ Infrastructure Layer │
              │  (Implementations)   │
              └──────────────────────┘
```

### Layers

| Layer | Location | Contents |
|-------|----------|----------|
| **Domain** | `src/domain/` | `Embedding`, `Vector`, `EmbeddingMetadata`, `LogEntry` |
| **Ports** | `src/ports/` | `EmbeddingRepository`, `EmbeddingGenerator`, `VectorStore` |
| **Application** | `src/application/` | `EmbeddingService`, `LogAnalyzer`, Use Cases |
| **Infrastructure** | `src/infrastructure/` | `SqliteRepository`, `HnswVectorStore`, `OnnxEmbeddingGenerator` |

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rust-embeddings = "0.1.0"
```

### Basic Example

```rust
use std::sync::Arc;
use rust_embeddings::prelude::*;
use rust_embeddings::infrastructure::*;
use rust_embeddings::ports::SearchParams;

#[tokio::main]
async fn main() -> Result<()> {
    // Production setup with SQLite + HNSW + ONNX
    let repository = Arc::new(SqliteRepository::new("embeddings.db")?);
    let vector_store = Arc::new(HnswVectorStore::new());
    let generator = Arc::new(
        OnnxEmbeddingGenerator::new("all-MiniLM-L6-v2", None).await?
    );

    let service = EmbeddingService::new(generator, repository, vector_store);

    // Create embeddings
    let embedding = service.create_embedding("Rust programming language").await?;
    println!("Created: {}", embedding.id());

    // Search similar
    let results = service.find_similar(
        "systems programming",
        SearchParams::builder().k(5).build(),
    ).await?;

    for result in results {
        println!("Score: {:.4}", result.score);
    }

    Ok(())
}
```

## Examples

The library includes 5 production-ready examples:

```bash
# Memory patterns and concurrency (Stack, Heap, Static, Threads, Rayon)
cargo run --example memory_and_concurrency

# SQLite + HNSW persistent storage
cargo run --example sqlite_persistent --release

# HNSW ANN search benchmark (1K, 10K, 100K vectors)
cargo run --example ann_hnsw --release

# Log analysis with semantic search
cargo run --example log_analysis --release

# ONNX batch vs server mode benchmark
cargo run --example benchmark_pool --release
```

### Example: Log Analysis

```rust
use rust_embeddings::application::LogAnalyzer;
use rust_embeddings::domain::{LogEntry, LogLevel, TimeWindow};

// Create log analyzer
let analyzer = LogAnalyzer::new(generator, repository, vector_store);

// Index logs
let log = LogEntry::builder()
    .message("Database connection timeout")
    .level(LogLevel::Error)
    .source("api-server")
    .build();

analyzer.index_log(&log).await?;

// Semantic search
let results = analyzer.search_logs("connection problems", 10, None).await?;

// Find errors in last hour
let errors = analyzer.find_errors(TimeWindow::last_hours(1), 10).await?;

// Detect anomalies
let anomalies = analyzer.detect_anomalies(TimeWindow::last_hours(1), 5).await?;
```

## Infrastructure Components

### ONNX Embedding Generator

Local inference with Hugging Face models:

```rust
// Batch mode (optimal for CLI/batch jobs)
let generator = OnnxEmbeddingGenerator::new("all-MiniLM-L6-v2", None).await?;

// Server mode (optimal for API servers)
let generator = OnnxEmbeddingGenerator::with_mode(
    "all-MiniLM-L6-v2",
    None,
    ExecutionMode::Server,
).await?;

let embedding = generator.generate_from_text("Hello world").await?;
```

### SQLite Repository

ACID-compliant persistent storage:

```rust
// File-based
let repo = SqliteRepository::new("embeddings.db")?;

// In-memory (for testing)
let repo = SqliteRepository::in_memory()?;

repo.save(&embedding).await?;
let found = repo.find_by_id(embedding.id()).await?;
```

### HNSW Vector Store

Fast approximate nearest neighbor search:

```rust
let config = HnswConfig {
    m: 16,                  // Connections per node
    ef_construction: 200,   // Build quality
    ef_search: 100,         // Search quality
};

let store = HnswVectorStore::with_config(config);
store.index(&embedding).await?;

let results = store.search(
    &query_vector,
    SearchParams::builder().k(10).build(),
).await?;
```

## Design Patterns

### Repository Pattern
```rust
#[async_trait]
pub trait EmbeddingRepository: Send + Sync {
    async fn save(&self, embedding: &Embedding) -> Result<()>;
    async fn find_by_id(&self, id: &str) -> Result<Option<Embedding>>;
    async fn list(&self) -> Result<Vec<Embedding>>;
    async fn delete(&self, id: &str) -> Result<()>;
}
```

### Strategy Pattern
```rust
#[async_trait]
pub trait EmbeddingGenerator: Send + Sync {
    async fn generate_from_text(&self, text: &str) -> Result<Embedding>;
    async fn generate_batch(&self, texts: &[String]) -> Result<Vec<Embedding>>;
    fn model_name(&self) -> &str;
    fn dimensions(&self) -> usize;
}
```

### Builder Pattern
```rust
let metadata = EmbeddingMetadata::builder()
    .source("document.txt")
    .model("all-MiniLM-L6-v2")
    .tag("production")
    .property("version", "1.0")
    .build()?;
```

## Memory and Concurrency

### Memory Types

| Type | Location | Example |
|------|----------|---------|
| **Static** | Binary data segment | `const DEFAULT_DIMENSIONS: usize = 384;` |
| **Stack** | Thread stack (~2MB) | `let threshold: f32 = 0.85;` |
| **Heap** | Dynamic allocation | `Vec::new()`, `Arc::new()` |

### Concurrency Patterns

```rust
// Thread-based parallelism
let shared = Arc::new(data);
thread::spawn(move || process(&shared));

// Channel communication
let (tx, rx) = channel::bounded(100);
tx.send(message)?;

// Data parallelism with Rayon
embeddings.par_iter().map(|e| process(e)).collect();

// Async/Await
let result = service.create_embedding(text).await?;
```

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific module tests
cargo test infrastructure::hnsw
```

## Feature Flags

```toml
[features]
default = ["sqlite"]
sqlite = []
postgres = ["sqlx"]
server = ["axum", "tower", "tower-http", "prometheus"]
full = ["sqlite", "postgres", "server"]
```

## Performance

Benchmarks on 12-core CPU:

| Operation | Throughput |
|-----------|------------|
| ONNX Embedding Generation | ~100 docs/sec |
| HNSW Index (10K vectors) | ~300K vectors/sec |
| HNSW Search (10K vectors) | ~27K QPS |
| SQLite Save | ~1K embeddings/sec |

## Error Handling

```rust
pub enum EmbeddingError {
    NotFound(String),
    InvalidDimensions { expected: usize, actual: usize },
    InvalidVector(String),
    StorageError(String),
    GenerationError(String),
    ValidationError(String),
}
```

## License

MIT License

## Author

- alnroot - [GitHub](https://github.com/alnroot)
