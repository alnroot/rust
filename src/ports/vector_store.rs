//! Vector Store Port
//!
//! Defines the interface for similarity search.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::domain::{Embedding, Vector};
use crate::error::Result;

/// Distance metric for similarity calculation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine distance)
    Cosine,
    /// Euclidean distance (L2)
    Euclidean,
    /// Dot product similarity
    DotProduct,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

/// Parameters for similarity search
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Number of results to return
    pub k: usize,

    /// Minimum similarity threshold (optional)
    pub threshold: Option<f32>,

    /// Distance metric to use
    pub metric: DistanceMetric,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            threshold: None,
            metric: DistanceMetric::Cosine,
        }
    }
}

impl SearchParams {
    pub fn builder() -> SearchParamsBuilder {
        SearchParamsBuilder::default()
    }
}

/// Builder for SearchParams
#[derive(Default)]
pub struct SearchParamsBuilder {
    k: Option<usize>,
    threshold: Option<f32>,
    metric: Option<DistanceMetric>,
}

impl SearchParamsBuilder {
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = Some(metric);
        self
    }

    pub fn build(self) -> SearchParams {
        SearchParams {
            k: self.k.unwrap_or(10),
            threshold: self.threshold,
            metric: self.metric.unwrap_or(DistanceMetric::Cosine),
        }
    }
}

/// Result of a similarity search
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// The matching embedding
    pub embedding: Embedding,

    /// Similarity score
    pub score: f32,
}

/// Trait for vector similarity search
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Index an embedding for search
    async fn index(&self, embedding: &Embedding) -> Result<()>;

    /// Search for similar embeddings
    async fn search(&self, query: &Vector, params: SearchParams) -> Result<Vec<SimilarityResult>>;

    /// Remove an embedding from the index
    async fn remove(&self, id: &str) -> Result<()>;

    /// Get the number of indexed embeddings
    async fn count(&self) -> Result<usize>;
}
