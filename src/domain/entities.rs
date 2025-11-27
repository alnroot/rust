//! Domain entities
//!
//! Entities are objects with a unique identity that persists over time.
//!
//! ## Memory Allocation Patterns
//!
//! ### Embedding Struct
//! - **Size on Stack**: ~56 bytes (metadata pointer + timestamps + version + id)
//! - **Heap Allocations**:
//!   - `Vector` contains Array1<f32> → heap allocated array
//!   - `EmbeddingMetadata` contains String and Vec → heap allocations
//!   - `DateTime<Utc>` → stack allocated (12 bytes each)
//!   - `EmbeddingId(Uuid)` → stack allocated (16 bytes)
//!
//! ### Memory Ownership
//! - When `Embedding` is moved, ownership transfers (no copy)
//! - When cloned, both stack data AND heap data are copied
//! - References (`&Embedding`) only pass a pointer (8 bytes on 64-bit)
//!
//! ### Concurrency Implications
//! - Embeddings are `Clone` → can be shared via `Arc<Embedding>`
//! - Not `Send` by default unless all fields are `Send`
//! - Interior mutability requires `Mutex` or `RwLock`

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::error::{EmbeddingError, Result};
use super::value_objects::{Vector, EmbeddingMetadata};

/// Core entity representing an embedding
///
/// An embedding is a vector representation of data (text, image, etc.)
/// along with metadata and a unique identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Unique identifier
    id: EmbeddingId,

    /// The vector representation
    vector: Vector,

    /// Associated metadata
    metadata: EmbeddingMetadata,

    /// Creation timestamp
    created_at: DateTime<Utc>,

    /// Last update timestamp
    updated_at: DateTime<Utc>,

    /// Version for optimistic locking
    version: u32,
}

impl Embedding {
    /// Create a new embedding
    pub fn new(vector: Vector, metadata: EmbeddingMetadata) -> Self {
        let now = Utc::now();
        Self {
            id: EmbeddingId::new(),
            vector,
            metadata,
            created_at: now,
            updated_at: now,
            version: 1,
        }
    }

    /// Create an embedding with a specific ID (for reconstruction from storage)
    pub fn with_id(
        id: EmbeddingId,
        vector: Vector,
        metadata: EmbeddingMetadata,
        created_at: DateTime<Utc>,
        updated_at: DateTime<Utc>,
        version: u32,
    ) -> Self {
        Self {
            id,
            vector,
            metadata,
            created_at,
            updated_at,
            version,
        }
    }

    /// Get the embedding ID
    pub fn id(&self) -> &EmbeddingId {
        &self.id
    }

    /// Get the vector
    pub fn vector(&self) -> &Vector {
        &self.vector
    }

    /// Get the metadata
    pub fn metadata(&self) -> &EmbeddingMetadata {
        &self.metadata
    }

    /// Get creation timestamp
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Get last update timestamp
    pub fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }

    /// Get version
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Update the metadata
    pub fn update_metadata(&mut self, metadata: EmbeddingMetadata) {
        self.metadata = metadata;
        self.updated_at = Utc::now();
        self.version += 1;
    }

    /// Calculate similarity with another embedding
    pub fn similarity_with(&self, other: &Embedding) -> Result<f32> {
        self.vector.cosine_similarity(&other.vector)
    }

    /// Calculate distance with another embedding
    pub fn distance_from(&self, other: &Embedding) -> Result<f32> {
        self.vector.euclidean_distance(&other.vector)
    }

    /// Check if dimensions match another embedding
    pub fn is_compatible_with(&self, other: &Embedding) -> bool {
        self.vector.dimensions() == other.vector.dimensions()
    }

    /// Validate the embedding
    pub fn validate(&self) -> Result<()> {
        if self.vector.dimensions() == 0 {
            return Err(EmbeddingError::validation("Vector cannot be empty"));
        }

        if self.metadata.source.is_empty() {
            return Err(EmbeddingError::validation("Source cannot be empty"));
        }

        if self.metadata.model.is_empty() {
            return Err(EmbeddingError::validation("Model cannot be empty"));
        }

        Ok(())
    }
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Embedding {}

/// Unique identifier for embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingId(Uuid);

impl EmbeddingId {
    /// Create a new random ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create an ID from a UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Parse an ID from a string
    pub fn parse(s: &str) -> Result<Self> {
        Uuid::parse_str(s)
            .map(Self)
            .map_err(|e| EmbeddingError::validation(format!("Invalid UUID: {}", e)))
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for EmbeddingId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EmbeddingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for EmbeddingId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl From<EmbeddingId> for Uuid {
    fn from(id: EmbeddingId) -> Self {
        id.0
    }
}

/// Collection of embeddings with aggregation capabilities
#[derive(Debug, Clone)]
pub struct EmbeddingCollection {
    embeddings: Vec<Embedding>,
}

impl EmbeddingCollection {
    /// Create a new empty collection
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
        }
    }

    /// Create a collection from a vector of embeddings
    pub fn from_vec(embeddings: Vec<Embedding>) -> Self {
        Self { embeddings }
    }

    /// Add an embedding to the collection
    pub fn add(&mut self, embedding: Embedding) {
        self.embeddings.push(embedding);
    }

    /// Get the number of embeddings
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get embeddings as a slice
    pub fn embeddings(&self) -> &[Embedding] {
        &self.embeddings
    }

    /// Find the most similar embedding to a target
    pub fn find_most_similar(&self, target: &Embedding) -> Result<Option<&Embedding>> {
        let mut best_match: Option<&Embedding> = None;
        let mut best_similarity = f32::NEG_INFINITY;

        for embedding in &self.embeddings {
            if !embedding.is_compatible_with(target) {
                continue;
            }

            let similarity = embedding.similarity_with(target)?;
            if similarity > best_similarity {
                best_similarity = similarity;
                best_match = Some(embedding);
            }
        }

        Ok(best_match)
    }

    /// Find the k most similar embeddings
    pub fn find_k_most_similar(&self, target: &Embedding, k: usize) -> Result<Vec<&Embedding>> {
        let mut similarities: Vec<(&Embedding, f32)> = self
            .embeddings
            .iter()
            .filter(|e| e.is_compatible_with(target))
            .map(|e| {
                let sim = e.similarity_with(target)?;
                Ok((e, sim))
            })
            .collect::<Result<Vec<_>>>()?;

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(similarities.into_iter().take(k).map(|(e, _)| e).collect())
    }
}

impl Default for EmbeddingCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::value_objects::EmbeddingMetadata;

    fn create_test_embedding(data: &[f32]) -> Embedding {
        let vector = Vector::new(data.to_vec()).unwrap();
        let metadata = EmbeddingMetadata::builder()
            .source("test")
            .model("test-model")
            .build()
            .unwrap();
        Embedding::new(vector, metadata)
    }

    #[test]
    fn test_embedding_creation() {
        let embedding = create_test_embedding(&[1.0, 2.0, 3.0]);
        assert_eq!(embedding.vector().dimensions(), 3);
    }

    #[test]
    fn test_embedding_similarity() {
        let emb1 = create_test_embedding(&[1.0, 0.0, 0.0]);
        let emb2 = create_test_embedding(&[1.0, 0.0, 0.0]);
        let similarity = emb1.similarity_with(&emb2).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_id_parsing() {
        let id = EmbeddingId::new();
        let id_str = id.to_string();
        let parsed_id = EmbeddingId::parse(&id_str).unwrap();
        assert_eq!(id, parsed_id);
    }

    #[test]
    fn test_collection_find_most_similar() {
        let mut collection = EmbeddingCollection::new();
        collection.add(create_test_embedding(&[1.0, 0.0, 0.0]));
        collection.add(create_test_embedding(&[0.0, 1.0, 0.0]));

        let target = create_test_embedding(&[0.9, 0.1, 0.0]);
        let most_similar = collection.find_most_similar(&target).unwrap().unwrap();

        let similarity = most_similar.similarity_with(&target).unwrap();
        assert!(similarity > 0.9);
    }
}
