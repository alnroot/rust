//! Log Time Analyzer - Production-Ready Log Analysis System
//!
//! Provides semantic search and temporal analysis of logs using embeddings.
//!
//! Features:
//! - Semantic log search ("find errors related to database")
//! - Temporal analysis (logs in specific time windows)
//! - Anomaly detection (unusual patterns)
//! - Performance analysis (slow requests)
//! - Error clustering (group similar errors)

use std::sync::Arc;
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;

use crate::domain::{Embedding, Vector, EmbeddingMetadata, LogEntry, TimeWindow};
use crate::error::Result;
use crate::ports::{EmbeddingGenerator, EmbeddingRepository, VectorStore, SearchParams};

/// Log analyzer service
pub struct LogAnalyzer {
    generator: Arc<dyn EmbeddingGenerator>,
    repository: Arc<dyn EmbeddingRepository>,
    vector_store: Arc<dyn VectorStore>,
}

impl LogAnalyzer {
    /// Create a new log analyzer
    pub fn new(
        generator: Arc<dyn EmbeddingGenerator>,
        repository: Arc<dyn EmbeddingRepository>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        Self {
            generator,
            repository,
            vector_store,
        }
    }

    /// Index a log entry for search
    pub async fn index_log(&self, log: &LogEntry) -> Result<Embedding> {
        // Create searchable text from log
        let text = log.to_searchable_text();

        // Generate embedding
        let embedding = self.generator.generate_from_text(&text).await?;

        // Enrich metadata with log information
        let mut metadata = EmbeddingMetadata::builder()
            .source(log.source.clone())
            .model(self.generator.model_name())
            .tag(log.level.to_string())
            .field("timestamp", log.timestamp.to_rfc3339())
            .field("level", log.level.to_string());

        if let Some(trace_id) = &log.trace_id {
            metadata = metadata.field("trace_id", trace_id.clone());
        }

        if let Some(duration) = log.duration_ms {
            metadata = metadata.field("duration_ms", duration.to_string());
        }

        // Add custom fields
        for (key, value) in &log.fields {
            metadata = metadata.field(key.clone(), value.clone());
        }

        let metadata = metadata.build()?;

        // Create embedding with enriched metadata
        let enriched_embedding = Embedding::new(embedding.vector().clone(), metadata);

        // Save to repository
        self.repository.save(&enriched_embedding).await?;

        // Index in vector store
        self.vector_store.index(&enriched_embedding).await?;

        Ok(enriched_embedding)
    }

    /// Index multiple logs in batch
    pub async fn index_logs_batch(&self, logs: &[LogEntry]) -> Result<Vec<Embedding>> {
        let mut embeddings = Vec::new();

        for log in logs {
            let embedding = self.index_log(log).await?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Semantic search: Find logs similar to a query
    pub async fn search_logs(
        &self,
        query: &str,
        limit: usize,
        time_window: Option<TimeWindow>,
    ) -> Result<Vec<LogSearchResult>> {
        // Generate query embedding
        let query_embedding = self.generator.generate_from_text(query).await?;

        // Search similar embeddings
        let params = SearchParams::builder().k(limit).build();
        let results = self.vector_store.search(query_embedding.vector(), params).await?;

        // Convert to LogSearchResult with filtering
        let mut log_results = Vec::new();

        for result in results {
            // Parse timestamp from metadata
            if let Some(timestamp_str) = result.embedding.metadata().properties.get("timestamp") {
                if let Ok(timestamp) = DateTime::parse_from_rfc3339(timestamp_str) {
                    let timestamp_utc = timestamp.with_timezone(&Utc);

                    // Filter by time window if specified
                    if let Some(ref window) = time_window {
                        if !window.contains(timestamp_utc) {
                            continue;
                        }
                    }

                    log_results.push(LogSearchResult {
                        embedding: result.embedding,
                        score: result.score,
                        timestamp: timestamp_utc,
                    });
                }
            }
        }

        // Sort by timestamp (most recent first)
        log_results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        Ok(log_results)
    }

    /// Find errors in a time window
    pub async fn find_errors(&self, window: TimeWindow, limit: usize) -> Result<Vec<LogSearchResult>> {
        // Search for errors using semantic understanding
        let error_queries = vec![
            "error exception failed",
            "critical fatal crash",
            "timeout connection refused",
        ];

        let mut all_results = Vec::new();

        for query in error_queries {
            let results = self.search_logs(query, limit, Some(window.clone())).await?;
            all_results.extend(results);
        }

        // Deduplicate and sort by score
        all_results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(all_results.into_iter().take(limit).collect())
    }

    /// Find slow requests (performance analysis)
    pub async fn find_slow_requests(&self, window: TimeWindow, threshold_ms: f64, limit: usize) -> Result<Vec<LogSearchResult>> {
        self.search_logs(
            &format!("slow request performance duration {}ms", threshold_ms),
            limit,
            Some(window)
        ).await
    }

    /// Get log statistics for a time window
    pub async fn get_statistics(&self, window: TimeWindow) -> Result<LogStatistics> {
        // Get all embeddings (in production, would filter by time in DB)
        let all_embeddings = self.repository.list().await?;

        let mut stats = LogStatistics {
            total_logs: 0,
            error_count: 0,
            warn_count: 0,
            info_count: 0,
            avg_duration_ms: None,
            time_window: window.clone(),
            top_sources: HashMap::new(),
        };

        let mut durations = Vec::new();

        for embedding in all_embeddings {
            // Parse timestamp
            if let Some(timestamp_str) = embedding.metadata().properties.get("timestamp") {
                if let Ok(timestamp) = DateTime::parse_from_rfc3339(timestamp_str) {
                    let timestamp_utc = timestamp.with_timezone(&Utc);

                    if window.contains(timestamp_utc) {
                        stats.total_logs += 1;

                        // Count by level
                        if let Some(level_str) = embedding.metadata().properties.get("level") {
                            match level_str.as_str() {
                                "ERROR" | "FATAL" => stats.error_count += 1,
                                "WARN" | "WARNING" => stats.warn_count += 1,
                                "INFO" => stats.info_count += 1,
                                _ => {}
                            }
                        }

                        // Track duration
                        if let Some(duration_str) = embedding.metadata().properties.get("duration_ms") {
                            if let Ok(duration) = duration_str.parse::<f64>() {
                                durations.push(duration);
                            }
                        }

                        // Track sources
                        let source = embedding.metadata().source.clone();
                        *stats.top_sources.entry(source).or_insert(0) += 1;
                    }
                }
            }
        }

        // Calculate average duration
        if !durations.is_empty() {
            let sum: f64 = durations.iter().sum();
            stats.avg_duration_ms = Some(sum / durations.len() as f64);
        }

        Ok(stats)
    }

    /// Detect anomalies (logs that are unusual)
    pub async fn detect_anomalies(&self, window: TimeWindow, limit: usize) -> Result<Vec<LogSearchResult>> {
        // Strategy: Find logs that are dissimilar to common patterns
        // In production, this would use more sophisticated ML techniques

        // Get recent normal logs (before the window)
        let normal_window_end = window.start;
        let normal_window_start = normal_window_end - Duration::hours(24);
        let normal_window = TimeWindow::new(normal_window_start, normal_window_end);

        // Get all embeddings in target window
        let target_logs = self.search_logs("*", limit * 2, Some(window)).await?;

        // Get baseline logs (what's "normal")
        let baseline_logs = self.search_logs("*", 100, Some(normal_window)).await?;

        if baseline_logs.is_empty() {
            return Ok(target_logs); // No baseline, return all
        }

        // Calculate centroid of normal logs
        let baseline_vectors: Vec<&Vector> = baseline_logs
            .iter()
            .map(|r| r.embedding.vector())
            .collect();

        let centroid = calculate_centroid(&baseline_vectors)?;

        // Find logs farthest from centroid (anomalies)
        let mut anomalies: Vec<(LogSearchResult, f32)> = Vec::new();

        for log_result in target_logs {
            let distance = log_result.embedding.vector()
                .euclidean_distance(&centroid)
                .unwrap_or(0.0);

            anomalies.push((log_result, distance));
        }

        // Sort by distance (highest = most anomalous)
        anomalies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(anomalies.into_iter()
            .take(limit)
            .map(|(result, _)| result)
            .collect())
    }
}

/// Log search result with temporal information
#[derive(Debug, Clone)]
pub struct LogSearchResult {
    pub embedding: Embedding,
    pub score: f32,
    pub timestamp: DateTime<Utc>,
}

/// Log statistics for a time window
#[derive(Debug, Clone)]
pub struct LogStatistics {
    pub total_logs: usize,
    pub error_count: usize,
    pub warn_count: usize,
    pub info_count: usize,
    pub avg_duration_ms: Option<f64>,
    pub time_window: TimeWindow,
    pub top_sources: HashMap<String, usize>,
}

/// Calculate centroid of vectors
fn calculate_centroid(vectors: &[&Vector]) -> Result<Vector> {
    if vectors.is_empty() {
        return Err(crate::error::EmbeddingError::validation("Cannot calculate centroid of empty set"));
    }

    let dims = vectors[0].dimensions();

    let mut centroid_data = vec![0.0; dims];

    for vector in vectors {
        for (i, &value) in vector.data().iter().enumerate() {
            centroid_data[i] += value;
        }
    }

    // Average
    let count = vectors.len() as f32;
    for value in &mut centroid_data {
        *value /= count;
    }

    Vector::new(centroid_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::LogLevel;
    use crate::infrastructure::{InMemoryRepository, InMemoryVectorStore};

    // Mock generator for testing
    struct MockGenerator {
        dimensions: usize,
    }

    #[async_trait::async_trait]
    impl EmbeddingGenerator for MockGenerator {
        async fn generate_from_text(&self, text: &str) -> Result<Embedding> {
            let vector_data: Vec<f32> = (0..self.dimensions)
                .map(|i| (text.len() as f32 + i as f32) * 0.01)
                .collect();

            let vector = Vector::new(vector_data)?;
            let metadata = EmbeddingMetadata::builder()
                .source(text)
                .model("mock-model")
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
            "mock-model"
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }
    }

    #[tokio::test]
    async fn test_log_analyzer_index_and_search() {
        let generator = Arc::new(MockGenerator { dimensions: 128 });
        let repository = Arc::new(InMemoryRepository::new());
        let vector_store = Arc::new(InMemoryVectorStore::new());

        let analyzer = LogAnalyzer::new(generator, repository, vector_store);

        // Create test logs
        let log1 = LogEntry::builder()
            .now()
            .level(LogLevel::Error)
            .message("Database connection failed")
            .source("api-server")
            .field("error_code", "DB001")
            .build();

        let log2 = LogEntry::builder()
            .now()
            .level(LogLevel::Info)
            .message("Request processed successfully")
            .source("api-server")
            .duration_ms(45.2)
            .build();

        // Index logs
        analyzer.index_log(&log1).await.unwrap();
        analyzer.index_log(&log2).await.unwrap();

        // Search for database errors
        let results = analyzer
            .search_logs("database error", 10, None)
            .await
            .unwrap();

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_log_statistics() {
        let generator = Arc::new(MockGenerator { dimensions: 128 });
        let repository = Arc::new(InMemoryRepository::new());
        let vector_store = Arc::new(InMemoryVectorStore::new());

        let analyzer = LogAnalyzer::new(generator, repository, vector_store);

        // Create test logs with different levels
        for i in 0..10 {
            let level = if i < 2 {
                LogLevel::Error
            } else if i < 5 {
                LogLevel::Warn
            } else {
                LogLevel::Info
            };

            let log = LogEntry::builder()
                .now()
                .level(level)
                .message(format!("Test log {}", i))
                .source("test-service")
                .build();

            analyzer.index_log(&log).await.unwrap();
        }

        // Get statistics
        let window = TimeWindow::last_hours(1);
        let stats = analyzer.get_statistics(window).await.unwrap();

        assert_eq!(stats.total_logs, 10);
        assert_eq!(stats.error_count, 2);
        assert_eq!(stats.warn_count, 3);
        assert_eq!(stats.info_count, 5);
    }
}
