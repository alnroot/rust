//! Benchmark: ExecutionMode::Batch vs ExecutionMode::Server
//!
//! Demonstrates the difference between execution modes for production use.

use rust_embeddings::infrastructure::{OnnxEmbeddingGenerator, ExecutionMode};
use rust_embeddings::ports::EmbeddingGenerator;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let texts: Vec<String> = (0..100)
        .map(|i| format!("This is test document number {} for embedding generation benchmark", i))
        .collect();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Production Benchmark: Batch vs Server Mode             ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    println!("Documents: {}", texts.len());
    println!("CPUs: {}\n", num_cpus::get());

    // ===== TEST 1: Batch Mode (default, optimal for throughput) =====
    println!("━━━ ExecutionMode::Batch (CLI/batch processing) ━━━");
    let gen_batch = OnnxEmbeddingGenerator::new("all-MiniLM-L6-v2", None).await?;
    println!("Mode: {:?}", gen_batch.mode());
    println!("Sessions: {} (all threads per session)", gen_batch.pool_size());

    let _ = gen_batch.generate_from_text("warmup").await?;

    let start = Instant::now();
    let embeddings = gen_batch.generate_batch(&texts).await?;
    let batch_time = start.elapsed();
    let batch_throughput = texts.len() as f64 / batch_time.as_secs_f64();
    println!("Time: {:?}", batch_time);
    println!("Throughput: {:.1} docs/sec\n", batch_throughput);

    // ===== TEST 2: Server Mode (optimal for concurrent requests) =====
    println!("━━━ ExecutionMode::Server (API server) ━━━");
    let gen_server = OnnxEmbeddingGenerator::with_mode(
        "all-MiniLM-L6-v2",
        None,
        ExecutionMode::Server,
    ).await?;
    println!("Mode: {:?}", gen_server.mode());
    println!("Sessions: {} (distributed threads)", gen_server.pool_size());

    let _ = gen_server.generate_from_text("warmup").await?;

    let start = Instant::now();
    let _ = gen_server.generate_batch(&texts).await?;
    let server_time = start.elapsed();
    let server_throughput = texts.len() as f64 / server_time.as_secs_f64();
    println!("Time: {:?}", server_time);
    println!("Throughput: {:.1} docs/sec\n", server_throughput);

    // ===== RESULTS =====
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS                               ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Batch mode:   {:>6.1} docs/sec  ← Best for batch jobs    ║", batch_throughput);
    println!("║ Server mode:  {:>6.1} docs/sec  ← Best for API servers   ║", server_throughput);
    println!("╠══════════════════════════════════════════════════════════╣");
    if batch_throughput > server_throughput {
        println!("║ Batch is {:.1}x faster for sequential processing        ║",
                 batch_throughput / server_throughput);
    } else {
        println!("║ Server is {:.1}x faster (unusual for batch workload)    ║",
                 server_throughput / batch_throughput);
    }
    println!("╚══════════════════════════════════════════════════════════╝");

    println!("\n📋 Production Recommendations:");
    println!("   • CLI tools, batch jobs    → ExecutionMode::Batch (default)");
    println!("   • API servers, web apps    → ExecutionMode::Server");
    println!("   • Worker queues            → ExecutionMode::Server");

    // Sanity check
    let e1 = embeddings[0].vector().data();
    let e2 = embeddings[1].vector().data();
    let diff: f32 = e1.iter().zip(e2.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("\n✓ Sanity check: embeddings differ by {:.2}", diff);
    println!("✓ Generated {} embeddings × {} dimensions", embeddings.len(), embeddings[0].vector().data().len());

    Ok(())
}
