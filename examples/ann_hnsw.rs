//! HNSW (Hierarchical Navigable Small World) Example
//!
//! This example demonstrates production-ready Approximate Nearest Neighbor (ANN) search
//! using HNSW index for efficient similarity search at scale.
//!
//! ## What is HNSW?
//! - Hierarchical graph structure for fast similarity search
//! - O(log n) search complexity vs O(n) for brute-force
//! - Configurable recall/speed tradeoff
//! - Production-ready for 10K-10M+ vectors
//!
//! ## Use Cases:
//! - Semantic search in large document collections
//! - Image/video similarity search
//! - Recommendation systems
//! - Duplicate detection
//!
//! Run with: cargo run --example ann_hnsw --release

use rust_embeddings::prelude::*;
use rust_embeddings::infrastructure::{HnswVectorStore, HnswConfig};
use rust_embeddings::ports::{SearchParams, DistanceMetric};
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  HNSW Approximate Nearest Neighbor Search Demo            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Step 1: Create HNSW vector store with custom configuration
    println!("📊 Configuring HNSW Index...");
    let hnsw_config = HnswConfig {
        m: 16,                  // Number of connections per node (memory/recall tradeoff)
        ef_construction: 200,   // Construction quality (higher = better quality, slower build)
        ef_search: 100,         // Search quality (higher = better recall, slower search)
    };

    println!("   • M (connections per node): {}", hnsw_config.m);
    println!("   • efConstruction: {}", hnsw_config.ef_construction);
    println!("   • efSearch: {}", hnsw_config.ef_search);
    println!();

    // Step 2: Generate sample dataset
    println!("🔧 Generating sample dataset...");
    let dataset_sizes = vec![1_000, 10_000, 100_000];

    for size in dataset_sizes {
        println!("\n┌─────────────────────────────────────────────────────────┐");
        println!("│ Testing with {} vectors", size);
        println!("└─────────────────────────────────────────────────────────┘");

        // Create fresh vector store for each test size
        let vector_store = Arc::new(HnswVectorStore::with_config(hnsw_config.clone()));

        // Create embeddings
        let embeddings = create_sample_embeddings(size)?;
        println!("✓ Created {} embeddings", embeddings.len());

        // Step 3: Index embeddings (build HNSW graph)
        println!("\n📥 Indexing embeddings...");
        let index_start = Instant::now();

        for embedding in &embeddings {
            vector_store.index(embedding).await?;
        }

        let index_time = index_start.elapsed();
        let vectors_per_sec = size as f64 / index_time.as_secs_f64();

        println!("✓ Indexed {} vectors in {:?}", size, index_time);
        println!("  Throughput: {:.0} vectors/sec", vectors_per_sec);

        // Step 4: Perform similarity searches
        println!("\n🔍 Performing similarity searches...");

        let query_vector = &embeddings[0].vector().clone();
        let search_params = SearchParams {
            k: 10,
            threshold: None,
            metric: DistanceMetric::Cosine,
        };

        // Benchmark search performance
        let num_queries = 100;
        let search_start = Instant::now();

        for _ in 0..num_queries {
            let _results = vector_store.search(query_vector, search_params.clone()).await?;
        }

        let search_time = search_start.elapsed();
        let avg_search_time = search_time / num_queries;
        let qps = num_queries as f64 / search_time.as_secs_f64();

        println!("✓ Ran {} queries", num_queries);
        println!("  Average query time: {:?}", avg_search_time);
        println!("  Throughput: {:.0} QPS (queries per second)", qps);

        // Show sample results
        let results = vector_store.search(query_vector, search_params.clone()).await?;
        println!("\n📋 Sample Results (top 5):");
        for (i, result) in results.iter().take(5).enumerate() {
            println!("  {}. ID: {} | Score: {:.4}",
                     i + 1,
                     result.embedding.id(),
                     result.score);
        }

        // Memory estimation
        let memory_per_vector = 512; // bytes (approximate for HNSW overhead)
        let total_memory_mb = (size * memory_per_vector) / (1024 * 1024);
        println!("\n💾 Memory Usage (estimated):");
        println!("  ~{} MB for {} vectors", total_memory_mb, size);
    }

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  Performance Summary                                       ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  HNSW provides:                                            ║");
    println!("║  • 100x+ faster search than brute-force                   ║");
    println!("║  • O(log n) complexity vs O(n)                            ║");
    println!("║  • 95%+ recall with proper tuning                         ║");
    println!("║  • ~500 bytes overhead per vector                         ║");
    println!("║                                                            ║");
    println!("║  Ideal for:                                                ║");
    println!("║  • 10K-10M+ vector datasets                               ║");
    println!("║  • Real-time similarity search                            ║");
    println!("║  • Production semantic search systems                     ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Create sample embeddings for benchmarking
fn create_sample_embeddings(count: usize) -> Result<Vec<Embedding>> {
    (0..count)
        .map(|i| {
            // Generate pseudo-random but deterministic vectors
            let data: Vec<f32> = (0..384)
                .map(|j| {
                    let val = ((i * 7 + j * 13) % 1000) as f32 / 1000.0;
                    val * 2.0 - 1.0 // Normalize to [-1, 1]
                })
                .collect();

            let vector = Vector::new(data)?;
            let metadata = EmbeddingMetadata::builder()
                .source(format!("doc-{}", i))
                .model("hnsw-demo")
                .tag("benchmark")
                .build()?;

            Ok(Embedding::new(vector, metadata))
        })
        .collect()
}
