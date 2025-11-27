//! In-Memory Vector Store Implementation

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::domain::{Embedding, Vector};
use crate::error::Result;
use crate::ports::{VectorStore, SearchParams, SimilarityResult, DistanceMetric};

/// In-memory implementation of VectorStore
pub struct InMemoryVectorStore {
    embeddings: Arc<RwLock<Vec<Embedding>>>,
}

impl InMemoryVectorStore {
    /// Create a new in-memory vector store
    pub fn new() -> Self {
        Self {
            embeddings: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Calculate similarity between two vectors based on metric
    fn calculate_similarity(v1: &Vector, v2: &Vector, metric: DistanceMetric) -> Result<f32> {
        match metric {
            DistanceMetric::Cosine => v1.cosine_similarity(v2),
            DistanceMetric::Euclidean => {
                // Convert distance to similarity (inverse)
                let distance = v1.euclidean_distance(v2)?;
                Ok(1.0 / (1.0 + distance))
            }
            DistanceMetric::DotProduct => {
                // Simple dot product
                let dot: f32 = v1.data().iter().zip(v2.data().iter()).map(|(a, b)| a * b).sum();
                Ok(dot)
            }
        }
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn index(&self, embedding: &Embedding) -> Result<()> {
        let mut embeddings = self.embeddings.write().await;
        embeddings.push(embedding.clone());
        Ok(())
    }

    async fn search(&self, query: &Vector, params: SearchParams) -> Result<Vec<SimilarityResult>> {
        let embeddings = self.embeddings.read().await;

        let mut results: Vec<SimilarityResult> = embeddings
            .iter()
            .filter_map(|embedding| {
                let score = Self::calculate_similarity(query, embedding.vector(), params.metric).ok()?;

                // Apply threshold if specified
                if let Some(threshold) = params.threshold {
                    if score < threshold {
                        return None;
                    }
                }

                Some(SimilarityResult {
                    embedding: embedding.clone(),
                    score,
                })
            })
            .collect();

        // Sort by score (descending)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        results.truncate(params.k);

        Ok(results)
    }

    async fn remove(&self, id: &str) -> Result<()> {
        let mut embeddings = self.embeddings.write().await;
        embeddings.retain(|e| e.id().to_string() != id);
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let embeddings = self.embeddings.read().await;
        Ok(embeddings.len())
    }
}
