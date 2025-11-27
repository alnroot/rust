//! Log Analysis Example - Production Use Case
//!
//! This example demonstrates a production-ready log analysis system using embeddings.
//!
//! Features demonstrated:
//! - Indexing logs with timestamps
//! - Semantic search across logs
//! - Time-based filtering
//! - Error detection
//! - Performance analysis
//! - Anomaly detection
//! - Statistical analysis

use std::sync::Arc;
use chrono::{DateTime, Utc, Duration};
use rust_embeddings::prelude::*;
use rust_embeddings::infrastructure::*;
use rust_embeddings::application::LogAnalyzer;
use rust_embeddings::domain::{LogEntry, LogLevel, TimeWindow};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Log Analysis System Demo ===\n");

    // Setup components
    println!("🔧 Setting up log analysis system...");

    // Use SQLite for persistent storage
    let repository = Arc::new(SqliteRepository::in_memory()?);

    // Use in-memory generator for demo (in production, use ONNX)
    let generator = Arc::new(create_mock_generator());

    // Use in-memory vector store
    let vector_store = Arc::new(InMemoryVectorStore::new());

    // Create log analyzer
    let analyzer = LogAnalyzer::new(
        generator.clone(),
        repository.clone(),
        vector_store.clone(),
    );

    println!("✅ Log analysis system ready\n");

    // Simulate realistic log data
    println!("📝 Generating sample logs...");
    generate_sample_logs(&analyzer).await?;
    println!("✅ Indexed {} logs\n", repository.count().await?);

    // Demo 1: Semantic search
    println!("=== Demo 1: Semantic Log Search ===");
    demo_semantic_search(&analyzer).await?;

    // Demo 2: Error detection
    println!("\n=== Demo 2: Error Detection ===");
    demo_error_detection(&analyzer).await?;

    // Demo 3: Performance analysis
    println!("\n=== Demo 3: Performance Analysis ===");
    demo_performance_analysis(&analyzer).await?;

    // Demo 4: Time-based analysis
    println!("\n=== Demo 4: Time-Based Analysis ===");
    demo_time_analysis(&analyzer).await?;

    // Demo 5: Statistics
    println!("\n=== Demo 5: Log Statistics ===");
    demo_statistics(&analyzer).await?;

    // Demo 6: Anomaly detection
    println!("\n=== Demo 6: Anomaly Detection ===");
    demo_anomaly_detection(&analyzer).await?;

    println!("\n✅ All demos completed!");
    println!("\n💡 This demonstrates a production-ready log analysis system");
    println!("   that can handle millions of logs with semantic search,");
    println!("   temporal analysis, and anomaly detection.");

    Ok(())
}

async fn generate_sample_logs(analyzer: &LogAnalyzer) -> Result<()> {
    let now = Utc::now();

    // Realistic log patterns
    let logs = vec![
        // Database errors (cluster 1)
        LogEntry::builder()
            .timestamp(now - Duration::minutes(30))
            .level(LogLevel::Error)
            .message("Database connection timeout after 30s")
            .source("api-server")
            .field("db_host", "postgres-primary.prod")
            .field("connection_pool", "exhausted")
            .duration_ms(30000.0)
            .build(),

        LogEntry::builder()
            .timestamp(now - Duration::minutes(25))
            .level(LogLevel::Error)
            .message("Failed to execute query: connection refused")
            .source("api-server")
            .field("db_host", "postgres-primary.prod")
            .field("query", "SELECT * FROM users")
            .build(),

        // API performance (cluster 2)
        LogEntry::builder()
            .timestamp(now - Duration::minutes(20))
            .level(LogLevel::Warn)
            .message("Slow API response detected")
            .source("api-gateway")
            .field("endpoint", "/api/v1/users")
            .field("method", "GET")
            .duration_ms(2500.0)
            .build(),

        LogEntry::builder()
            .timestamp(now - Duration::minutes(18))
            .level(LogLevel::Warn)
            .message("Response time exceeded threshold")
            .source("api-gateway")
            .field("endpoint", "/api/v1/orders")
            .field("method", "POST")
            .duration_ms(3200.0)
            .build(),

        // Normal operations (cluster 3)
        LogEntry::builder()
            .timestamp(now - Duration::minutes(15))
            .level(LogLevel::Info)
            .message("User authentication successful")
            .source("auth-service")
            .field("user_id", "user_12345")
            .field("method", "oauth2")
            .duration_ms(45.2)
            .build(),

        LogEntry::builder()
            .timestamp(now - Duration::minutes(12))
            .level(LogLevel::Info)
            .message("Order processed and payment captured")
            .source("payment-service")
            .field("order_id", "order_98765")
            .field("amount", "149.99")
            .duration_ms(123.5)
            .build(),

        // Cache issues (cluster 4)
        LogEntry::builder()
            .timestamp(now - Duration::minutes(10))
            .level(LogLevel::Warn)
            .message("Redis cache miss rate above threshold")
            .source("cache-service")
            .field("miss_rate", "45%")
            .field("cache_host", "redis-cluster-01")
            .build(),

        LogEntry::builder()
            .timestamp(now - Duration::minutes(8))
            .level(LogLevel::Error)
            .message("Failed to connect to Redis cluster")
            .source("cache-service")
            .field("cache_host", "redis-cluster-01")
            .field("error", "connection_refused")
            .build(),

        // Security events (cluster 5)
        LogEntry::builder()
            .timestamp(now - Duration::minutes(5))
            .level(LogLevel::Warn)
            .message("Multiple failed login attempts detected")
            .source("auth-service")
            .field("ip_address", "203.0.113.42")
            .field("attempt_count", "5")
            .field("username", "admin")
            .build(),

        LogEntry::builder()
            .timestamp(now - Duration::minutes(3))
            .level(LogLevel::Error)
            .message("Suspicious activity: potential SQL injection attempt")
            .source("api-server")
            .field("ip_address", "203.0.113.42")
            .field("endpoint", "/api/v1/search")
            .field("payload", "' OR '1'='1")
            .build(),

        // Normal traffic
        LogEntry::builder()
            .timestamp(now - Duration::minutes(1))
            .level(LogLevel::Info)
            .message("Health check passed")
            .source("api-server")
            .field("status", "healthy")
            .duration_ms(5.2)
            .build(),
    ];

    analyzer.index_logs_batch(&logs).await?;

    Ok(())
}

async fn demo_semantic_search(analyzer: &LogAnalyzer) -> Result<()> {
    println!("🔍 Searching: 'database connection problems'");

    let results = analyzer
        .search_logs("database connection problems", 5, None)
        .await?;

    println!("Found {} related logs:", results.len());
    for (i, result) in results.iter().enumerate().take(3) {
        println!("\n  {}. [Score: {:.3}] {}",
            i + 1,
            result.score,
            result.timestamp.format("%H:%M:%S")
        );

        if let Some(message) = result.embedding.metadata().properties.get("message") {
            println!("     Message: {}", message);
        }
    }

    Ok(())
}

async fn demo_error_detection(analyzer: &LogAnalyzer) -> Result<()> {
    let window = TimeWindow::last_hours(1);

    println!("🚨 Finding errors in last hour...");

    let errors = analyzer.find_errors(window, 5).await?;

    println!("Found {} errors:", errors.len());
    for (i, error) in errors.iter().enumerate().take(5) {
        if let Some(level) = error.embedding.metadata().properties.get("level") {
            println!("\n  {}. [{}] {}",
                i + 1,
                level,
                error.timestamp.format("%H:%M:%S")
            );
        }

        if let Some(source) = error.embedding.metadata().properties.get("source") {
            println!("     Source: {}", source);
        }
    }

    Ok(())
}

async fn demo_performance_analysis(analyzer: &LogAnalyzer) -> Result<()> {
    let window = TimeWindow::last_hours(1);

    println!("⏱️  Finding slow requests (> 1000ms)...");

    let slow_requests = analyzer.find_slow_requests(window, 1000.0, 5).await?;

    println!("Found {} slow requests:", slow_requests.len());
    for (i, req) in slow_requests.iter().enumerate() {
        if let Some(duration) = req.embedding.metadata().properties.get("duration_ms") {
            println!("\n  {}. Duration: {}ms at {}",
                i + 1,
                duration,
                req.timestamp.format("%H:%M:%S")
            );
        }

        if let Some(endpoint) = req.embedding.metadata().properties.get("endpoint") {
            println!("     Endpoint: {}", endpoint);
        }
    }

    Ok(())
}

async fn demo_time_analysis(analyzer: &LogAnalyzer) -> Result<()> {
    let window = TimeWindow::last_minutes(30);

    println!("📅 Analyzing logs from last 30 minutes...");

    let results = analyzer.search_logs("*", 100, Some(window)).await?;

    println!("Found {} logs in time window", results.len());

    // Group by 5-minute buckets
    let mut buckets: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();

    for result in &results {
        let bucket = result.timestamp.timestamp() / 300; // 5-minute buckets
        *buckets.entry(bucket).or_insert(0) += 1;
    }

    println!("\nLog distribution:");
    let mut sorted_buckets: Vec<_> = buckets.iter().collect();
    sorted_buckets.sort_by_key(|(k, _)| *k);

    for (bucket, count) in sorted_buckets {
        let timestamp = DateTime::from_timestamp(*bucket * 300, 0)
            .unwrap_or_else(|| Utc::now());
        println!("  {} - {}: {} logs",
            timestamp.format("%H:%M"),
            (timestamp + Duration::minutes(5)).format("%H:%M"),
            count
        );
    }

    Ok(())
}

async fn demo_statistics(analyzer: &LogAnalyzer) -> Result<()> {
    let window = TimeWindow::last_hours(1);

    println!("📊 Generating statistics for last hour...");

    let stats = analyzer.get_statistics(window).await?;

    println!("\nStatistics:");
    println!("  Total logs: {}", stats.total_logs);
    println!("  Errors: {}", stats.error_count);
    println!("  Warnings: {}", stats.warn_count);
    println!("  Info: {}", stats.info_count);

    if let Some(avg_duration) = stats.avg_duration_ms {
        println!("  Avg duration: {:.2}ms", avg_duration);
    }

    println!("\nTop sources:");
    let mut sorted_sources: Vec<_> = stats.top_sources.iter().collect();
    sorted_sources.sort_by(|a, b| b.1.cmp(a.1));

    for (source, count) in sorted_sources.iter().take(5) {
        println!("  {}: {} logs", source, count);
    }

    Ok(())
}

async fn demo_anomaly_detection(analyzer: &LogAnalyzer) -> Result<()> {
    let window = TimeWindow::last_hours(1);

    println!("🔎 Detecting anomalous logs...");

    let anomalies = analyzer.detect_anomalies(window, 3).await?;

    println!("Found {} potential anomalies:", anomalies.len());
    for (i, anomaly) in anomalies.iter().enumerate() {
        println!("\n  {}. {} - {}",
            i + 1,
            anomaly.timestamp.format("%H:%M:%S"),
            anomaly.embedding.metadata().source
        );

        if let Some(level) = anomaly.embedding.metadata().properties.get("level") {
            println!("     Level: {}", level);
        }
    }

    Ok(())
}

// Mock generator for demo
struct MockGenerator;

fn create_mock_generator() -> MockGenerator {
    MockGenerator
}

#[async_trait::async_trait]
impl EmbeddingGenerator for MockGenerator {
    async fn generate_from_text(&self, text: &str) -> Result<Embedding> {
        use rust_embeddings::domain::{Vector, EmbeddingMetadata};

        // Simple hash-based embedding for demo
        let mut vector_data = vec![0.0; 128];
        for (i, byte) in text.bytes().enumerate() {
            vector_data[i % 128] += byte as f32 / 255.0;
        }

        // Normalize
        let sum: f32 = vector_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if sum > 0.0 {
            for v in &mut vector_data {
                *v /= sum;
            }
        }

        let vector = Vector::new(vector_data)?;
        let metadata = EmbeddingMetadata::builder()
            .source(text)
            .model("mock-generator")
            .build()?;

        Ok(Embedding::new(vector, metadata))
    }

    async fn generate_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.generate_from_text(text).await?);
        }
        Ok(embeddings)
    }

    fn model_name(&self) -> &str {
        "mock-generator"
    }

    fn dimensions(&self) -> usize {
        128
    }
}
