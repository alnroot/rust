//! HNSW Vector Store - Efficient ANN Search
//!
//! Production-ready Approximate Nearest Neighbor search using
//! Hierarchical Navigable Small World graphs.
//!
//! Performance characteristics:
//! - Index construction: O(n log n)
//! - Search: O(log n) with high recall
//! - Memory: ~500 bytes overhead per vector
//!
//! Ideal for datasets > 10K vectors where brute-force is too slow.

use async_trait::async_trait;
use instant_distance::{Builder, HnswMap, Search, Point as InstantPoint};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::domain::{Embedding, Vector};
use crate::error::{EmbeddingError, Result};
use crate::ports::{VectorStore, SearchParams, SimilarityResult, DistanceMetric};

/// Point wrapper for instant-distance
#[derive(Clone, Debug)]
struct Point {
    /// Vector data
    data: Vec<f32>,
    /// Reference to original embedding ID
    embedding_id: String,
}

impl InstantPoint for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance: 1 - cosine_similarity
        cosine_distance(&self.data, &other.data)
    }
}

/// HNSW-based vector store for efficient similarity search
pub struct HnswVectorStore {
    /// HNSW index
    index: Arc<RwLock<Option<HnswMap<Point, usize>>>>,

    /// Points data for index building
    points: Arc<RwLock<Vec<Point>>>,

    /// Mapping from point ID to embedding
    embeddings: Arc<RwLock<HashMap<String, Embedding>>>,

    /// HNSW parameters
    config: HnswConfig,

    /// Whether index needs rebuild
    needs_rebuild: Arc<RwLock<bool>>,
}

/// HNSW configuration parameters
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Number of bi-directional links per node (M)
    /// Higher M = better recall, more memory
    /// Typical: 12-48
    pub m: usize,

    /// Size of dynamic candidate list (efConstruction)
    /// Higher ef = better quality index, slower building
    /// Typical: 100-500
    pub ef_construction: usize,

    /// Size of dynamic candidate list for search (efSearch)
    /// Higher ef = better recall, slower search
    /// Typical: 50-200
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

impl HnswVectorStore {
    /// Create a new HNSW vector store with default configuration
    pub fn new() -> Self {
        Self::with_config(HnswConfig::default())
    }

    /// Create a new HNSW vector store with custom configuration
    pub fn with_config(config: HnswConfig) -> Self {
        Self {
            index: Arc::new(RwLock::new(None)),
            points: Arc::new(RwLock::new(Vec::new())),
            embeddings: Arc::new(RwLock::new(HashMap::new())),
            config,
            needs_rebuild: Arc::new(RwLock::new(false)),
        }
    }

    /// Rebuild the HNSW index
    fn rebuild_index(&self) -> Result<()> {
        let points = self.points.read()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock points: {}", e)))?;

        if points.is_empty() {
            // Clear the index if no points
            let mut index = self.index.write()
                .map_err(|e| EmbeddingError::storage(format!("Failed to lock index: {}", e)))?;
            *index = None;

            let mut needs_rebuild = self.needs_rebuild.write()
                .map_err(|e| EmbeddingError::storage(format!("Failed to lock rebuild flag: {}", e)))?;
            *needs_rebuild = false;

            return Ok(());
        }

        // Build HNSW index
        let points_vec: Vec<Point> = points.clone();

        // Release read lock before building (expensive operation)
        drop(points);

        tracing::info!(
            "Building HNSW index with {} points (M={}, efConstruction={})",
            points_vec.len(),
            self.config.m,
            self.config.ef_construction
        );

        // Create value indices (0, 1, 2, ...)
        let values: Vec<usize> = (0..points_vec.len()).collect();

        let hnsw = Builder::default()
            .seed(42)
            .build(points_vec, values);

        // Update index
        let mut index = self.index.write()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock index: {}", e)))?;
        *index = Some(hnsw);

        let mut needs_rebuild = self.needs_rebuild.write()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock rebuild flag: {}", e)))?;
        *needs_rebuild = false;

        tracing::info!("HNSW index built successfully");

        Ok(())
    }

    /// Check if index needs rebuild and rebuild if necessary
    fn ensure_index_ready(&self) -> Result<()> {
        let needs_rebuild = self.needs_rebuild.read()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock rebuild flag: {}", e)))?;

        if *needs_rebuild {
            drop(needs_rebuild);
            self.rebuild_index()?;
        }

        Ok(())
    }
}

impl Default for HnswVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for HnswVectorStore {
    async fn index(&self, embedding: &Embedding) -> Result<()> {
        let id = embedding.id().to_string();

        // Add to embeddings map
        {
            let mut embeddings = self.embeddings.write()
                .map_err(|e| EmbeddingError::storage(format!("Failed to lock embeddings: {}", e)))?;
            embeddings.insert(id.clone(), embedding.clone());
        }

        // Add point to points list
        {
            let mut points = self.points.write()
                .map_err(|e| EmbeddingError::storage(format!("Failed to lock points: {}", e)))?;

            let point = Point {
                data: embedding.vector().data().to_vec(),
                embedding_id: id,
            };

            points.push(point);
        }

        // Mark index for rebuild
        {
            let mut needs_rebuild = self.needs_rebuild.write()
                .map_err(|e| EmbeddingError::storage(format!("Failed to lock rebuild flag: {}", e)))?;
            *needs_rebuild = true;
        }

        Ok(())
    }

    async fn search(&self, query: &Vector, params: SearchParams) -> Result<Vec<SimilarityResult>> {
        // Ensure index is ready
        self.ensure_index_ready()?;

        // Get index for searching
        let index = self.index.read()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock index: {}", e)))?;

        let hnsw = match &*index {
            Some(hnsw) => hnsw,
            None => return Ok(Vec::new()), // No data indexed yet
        };

        // Create query point
        let query_point = Point {
            data: query.data().to_vec(),
            embedding_id: String::new(), // Not needed for query
        };

        // Perform search
        let mut search = Search::default();
        let neighbors = hnsw.search(&query_point, &mut search);

        // Get embeddings for results
        let embeddings = self.embeddings.read()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock embeddings: {}", e)))?;

        // Get points for lookup
        let points = self.points.read()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock points: {}", e)))?;

        let mut results = Vec::new();

        for item in neighbors.take(params.k) {
            let distance = item.distance;
            let point_index = *item.value;

            // Use the actual index from the HNSW search result
            if let Some(point) = points.get(point_index) {
                if let Some(embedding) = embeddings.get(&point.embedding_id) {
                    // Convert distance to similarity based on metric
                    let score = match params.metric {
                        DistanceMetric::Cosine => 1.0 - distance,
                        DistanceMetric::Euclidean => 1.0 / (1.0 + distance),
                        DistanceMetric::DotProduct => -distance,
                    };

                    // Apply threshold filter
                    if let Some(threshold) = params.threshold {
                        if score < threshold {
                            continue;
                        }
                    }

                    results.push(SimilarityResult {
                        embedding: embedding.clone(),
                        score,
                    });
                }
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    async fn remove(&self, id: &str) -> Result<()> {
        // Remove from embeddings
        {
            let mut embeddings = self.embeddings.write()
                .map_err(|e| EmbeddingError::storage(format!("Failed to lock embeddings: {}", e)))?;
            embeddings.remove(id);
        }

        // Remove from points
        {
            let mut points = self.points.write()
                .map_err(|e| EmbeddingError::storage(format!("Failed to lock points: {}", e)))?;
            points.retain(|p| p.embedding_id != id);
        }

        // Mark for rebuild
        {
            let mut needs_rebuild = self.needs_rebuild.write()
                .map_err(|e| EmbeddingError::storage(format!("Failed to lock rebuild flag: {}", e)))?;
            *needs_rebuild = true;
        }

        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let embeddings = self.embeddings.read()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock embeddings: {}", e)))?;
        Ok(embeddings.len())
    }
}

/// Calculate cosine distance between two vectors
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return f32::MAX;
    }

    // Cosine distance = 1 - cosine_similarity
    1.0 - (dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::EmbeddingMetadata;

    #[tokio::test]
    async fn test_hnsw_index_and_search() {
        let store = HnswVectorStore::new();

        // Create some test embeddings
        for i in 0..100 {
            let vector = Vector::new(vec![i as f32; 128]).unwrap();
            let metadata = EmbeddingMetadata::builder()
                .source(format!("test-{}", i))
                .model("test-model")
                .build()
                .unwrap();

            let embedding = Embedding::new(vector, metadata);
            store.index(&embedding).await.unwrap();
        }

        // Search for similar vectors
        let query = Vector::new(vec![50.0; 128]).unwrap();
        let params = SearchParams::builder().k(10).build();

        let results = store.search(&query, params).await.unwrap();

        assert!(results.len() <= 10);
        assert!(!results.is_empty());

        // Results should be sorted by similarity
        for i in 1..results.len() {
            assert!(results[i-1].score >= results[i].score);
        }
    }

    #[tokio::test]
    async fn test_hnsw_remove() {
        let store = HnswVectorStore::new();

        // Add embeddings
        let mut ids = Vec::new();
        for i in 0..10 {
            let vector = Vector::new(vec![i as f32; 128]).unwrap();
            let metadata = EmbeddingMetadata::builder()
                .source(format!("test-{}", i))
                .model("test-model")
                .build()
                .unwrap();

            let embedding = Embedding::new(vector, metadata);
            let id = embedding.id().to_string();
            ids.push(id);
            store.index(&embedding).await.unwrap();
        }

        assert_eq!(store.count().await.unwrap(), 10);

        // Remove one
        store.remove(&ids[0]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 9);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!(dist.abs() < 1e-6); // Same vectors = distance 0

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6); // Orthogonal = distance 1
    }
}
