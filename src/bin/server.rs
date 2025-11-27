//! Embedding API Server Binary
//!
//! Production-ready REST API server for embeddings.
//!
//! ## Usage
//!
//! ```bash
//! # Run with default settings (localhost:8080)
//! cargo run --bin server --features server
//!
//! # Run with custom settings
//! HOST=0.0.0.0 PORT=3000 cargo run --bin server --features server
//! ```

use rust_embeddings::{
    api::{ApiServer, server::ServerConfig},
    application::EmbeddingService,
    infrastructure::{InMemoryRepository, InMemoryVectorStore, OnnxEmbeddingGenerator},
    error::Result,
};
use std::sync::Arc;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing/logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .init();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Rust Embeddings API Server                               ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Read configuration from environment
    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let config = ServerConfig {
        host,
        port,
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    // Setup infrastructure components
    tracing::info!("🔧 Initializing components...");

    // For demo: use in-memory storage
    // For production: use SqliteRepository or PostgresRepository
    let repository = Arc::new(InMemoryRepository::new());
    let vector_store = Arc::new(InMemoryVectorStore::new());

    // Try to initialize ONNX generator, fallback to mock if model not available
    let generator = match OnnxEmbeddingGenerator::new("all-MiniLM-L6-v2", None).await {
        Ok(gen) => {
            tracing::info!("✓ ONNX generator initialized");
            Arc::new(gen) as Arc<dyn rust_embeddings::ports::EmbeddingGenerator>
        }
        Err(e) => {
            tracing::warn!("Could not initialize ONNX generator: {}", e);
            tracing::info!("Using mock generator for development");
            Arc::new(create_mock_generator()) as Arc<dyn rust_embeddings::ports::EmbeddingGenerator>
        }
    };

    // Create service
    let service = Arc::new(EmbeddingService::new(
        generator,
        repository,
        vector_store,
    ));

    tracing::info!("✓ All components initialized");

    // Create and start server
    let server = ApiServer::new(config, service);

    println!("\n💡 Tips:");
    println!("  • Health check: curl http://{}:{}/api/v1/health",
             server.config.host, server.config.port);
    println!("  • Create embedding: curl -X POST http://{}:{}/api/v1/embeddings \\",
             server.config.host, server.config.port);
    println!("      -H 'Content-Type: application/json' \\");
    println!("      -d '{{\"text\": \"Hello world\"}}'");
    println!();

    // Run server
    server.serve().await
}

/// Create a mock embedding generator for development
fn create_mock_generator() -> impl rust_embeddings::ports::EmbeddingGenerator {
    use rust_embeddings::{domain::*, error::Result};

    struct MockGenerator;

    #[async_trait::async_trait]
    impl rust_embeddings::ports::EmbeddingGenerator for MockGenerator {
        async fn generate_from_text(&self, text: &str) -> Result<Embedding> {
            // Generate deterministic embeddings based on text
            let data: Vec<f32> = text
                .chars()
                .chain(std::iter::repeat(' '))
                .take(384)
                .enumerate()
                .map(|(i, c)| {
                    let char_val = c as u32 as f32 / 1000.0;
                    let pos_val = i as f32 / 100.0;
                    (char_val + pos_val).sin()
                })
                .collect();

            let vector = Vector::new(data)?;
            let metadata = EmbeddingMetadata::builder()
                .source("api")
                .model("mock-generator-v1")
                .tag("development")
                .build()?;

            Ok(Embedding::new(vector, metadata))
        }

        fn model_name(&self) -> &str {
            "mock-generator-v1"
        }

        fn dimensions(&self) -> usize {
            384
        }

        async fn generate_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
            let mut embeddings = Vec::with_capacity(texts.len());
            for text in texts {
                embeddings.push(self.generate_from_text(text).await?);
            }
            Ok(embeddings)
        }
    }

    MockGenerator
}
