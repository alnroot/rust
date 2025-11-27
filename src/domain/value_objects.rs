//! Domain value objects
//!
//! Value objects are immutable objects defined by their attributes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{EmbeddingError, Result};

/// Vector representation of an embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    /// The embedding vector data
    data: Vec<f32>,
}

impl Vector {
    /// Create a new vector
    pub fn new(data: Vec<f32>) -> Result<Self> {
        if data.is_empty() {
            return Err(EmbeddingError::invalid_vector("Vector cannot be empty"));
        }
        Ok(Self { data })
    }

    /// Get the vector data
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get the dimensions of the vector
    pub fn dimensions(&self) -> usize {
        self.data.len()
    }

    /// Calculate cosine similarity with another vector
    pub fn cosine_similarity(&self, other: &Vector) -> Result<f32> {
        if self.dimensions() != other.dimensions() {
            return Err(EmbeddingError::invalid_dimensions(
                self.dimensions(),
                other.dimensions(),
            ));
        }

        let dot_product: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.data.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Calculate Euclidean distance to another vector
    pub fn euclidean_distance(&self, other: &Vector) -> Result<f32> {
        if self.dimensions() != other.dimensions() {
            return Err(EmbeddingError::invalid_dimensions(
                self.dimensions(),
                other.dimensions(),
            ));
        }

        let sum: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum();

        Ok(sum.sqrt())
    }

    /// Truncate vector to specified dimensions (for Matryoshka embeddings)
    pub fn truncate(&self, dims: usize) -> Result<Self> {
        if dims > self.dimensions() {
            return Err(EmbeddingError::invalid_vector(
                format!("Cannot truncate to {} dims, vector has {} dims", dims, self.dimensions())
            ));
        }
        Ok(Self {
            data: self.data[..dims].to_vec(),
        })
    }
}

/// Metadata associated with an embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    /// Source of the embedding (e.g., filename, URL)
    pub source: String,

    /// Model used to generate the embedding
    pub model: String,

    /// Optional tags for categorization
    pub tags: Vec<String>,

    /// Additional properties
    pub properties: HashMap<String, String>,

    /// Timestamp when the embedding was created
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl EmbeddingMetadata {
    /// Create new metadata with current timestamp
    pub fn new(source: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            model: model.into(),
            tags: Vec::new(),
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
        }
    }

    /// Create a metadata builder
    pub fn builder() -> EmbeddingMetadataBuilder {
        EmbeddingMetadataBuilder::default()
    }
}

/// Builder for EmbeddingMetadata
#[derive(Default)]
pub struct EmbeddingMetadataBuilder {
    source: Option<String>,
    model: Option<String>,
    tags: Vec<String>,
    properties: HashMap<String, String>,
    created_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl EmbeddingMetadataBuilder {
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Alias for property (for convenience)
    pub fn field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Set explicit creation timestamp (useful for deserialization)
    pub fn created_at(mut self, timestamp: chrono::DateTime<chrono::Utc>) -> Self {
        self.created_at = Some(timestamp);
        self
    }

    /// Build the metadata, using current time if timestamp not explicitly set
    pub fn build(self) -> Result<EmbeddingMetadata> {
        Ok(EmbeddingMetadata {
            source: self.source.ok_or_else(|| EmbeddingError::validation("source is required"))?,
            model: self.model.ok_or_else(|| EmbeddingError::validation("model is required"))?,
            tags: self.tags,
            properties: self.properties,
            created_at: self.created_at.unwrap_or_else(chrono::Utc::now),
        })
    }
}
