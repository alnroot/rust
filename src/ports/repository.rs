//! Repository Port
//!
//! Defines the interface for persisting embeddings.

use async_trait::async_trait;
use crate::domain::Embedding;
use crate::error::Result;

/// Trait for embedding persistence
#[async_trait]
pub trait EmbeddingRepository: Send + Sync {
    /// Save an embedding
    async fn save(&self, embedding: &Embedding) -> Result<()>;

    /// Find embedding by ID
    async fn find_by_id(&self, id: &str) -> Result<Option<Embedding>>;

    /// List all embeddings
    async fn list(&self) -> Result<Vec<Embedding>>;

    /// Delete an embedding
    async fn delete(&self, id: &str) -> Result<()>;

    /// Count embeddings
    async fn count(&self) -> Result<usize>;
}
