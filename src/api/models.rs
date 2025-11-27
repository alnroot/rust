//! API Request/Response Models (DTOs)

use serde::{Deserialize, Serialize};
use crate::ports::DistanceMetric;

/// Request to create an embedding from text
#[derive(Debug, Deserialize)]
pub struct CreateEmbeddingRequest {
    /// Text to generate embedding from
    pub text: String,

    /// Optional source identifier
    #[serde(default)]
    pub source: Option<String>,

    /// Optional tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Request to create multiple embeddings
#[derive(Debug, Deserialize)]
pub struct BatchCreateRequest {
    /// Texts to generate embeddings from
    pub texts: Vec<String>,
}

/// Response after creating an embedding
#[derive(Debug, Serialize)]
pub struct CreateEmbeddingResponse {
    /// Unique identifier
    pub id: String,

    /// Embedding vector (high-dimensional)
    pub vector: Vec<f32>,

    /// Dimensions
    pub dimensions: usize,

    /// Source identifier
    pub source: String,

    /// Model used
    pub model: String,

    /// Timestamp (ISO 8601)
    pub created_at: String,
}

/// Response for batch creation
#[derive(Debug, Serialize)]
pub struct BatchCreateResponse {
    /// Created embedding IDs
    pub ids: Vec<String>,

    /// Number of embeddings created
    pub count: usize,
}

/// Request to search for similar embeddings
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// Query text
    pub query: String,

    /// Number of results (default: 10)
    #[serde(default = "default_k")]
    pub k: usize,

    /// Minimum similarity threshold
    #[serde(default)]
    pub threshold: Option<f32>,

    /// Distance metric (default: Cosine)
    #[serde(default)]
    pub metric: DistanceMetric,
}

fn default_k() -> usize {
    10
}

/// Single search result
#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    /// Embedding ID
    pub id: String,

    /// Similarity score
    pub score: f32,

    /// Source
    pub source: String,

    /// Optional text snippet
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Response for search query
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    /// Search results
    pub results: Vec<SearchResultItem>,

    /// Query time in milliseconds
    pub query_time_ms: f64,

    /// Total results found
    pub total: usize,
}

/// Embedding details response
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    /// Unique identifier
    pub id: String,

    /// Embedding vector
    pub vector: Vec<f32>,

    /// Dimensions
    pub dimensions: usize,

    /// Metadata
    pub metadata: EmbeddingMetadata,
}

/// Embedding metadata
#[derive(Debug, Serialize)]
pub struct EmbeddingMetadata {
    pub source: String,
    pub model: String,
    pub tags: Vec<String>,
    pub created_at: String,
}

/// List embeddings response
#[derive(Debug, Serialize)]
pub struct ListEmbeddingsResponse {
    /// Embeddings (without vectors for efficiency)
    pub embeddings: Vec<EmbeddingListItem>,

    /// Total count
    pub total: usize,
}

/// Embedding list item (no vector data)
#[derive(Debug, Serialize)]
pub struct EmbeddingListItem {
    pub id: String,
    pub source: String,
    pub model: String,
    pub dimensions: usize,
    pub created_at: String,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
}

/// Statistics response
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub total_embeddings: usize,
    pub indexed_embeddings: usize,
    pub model: String,
    pub dimensions: usize,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}
