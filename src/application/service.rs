//! Embedding Service - Main application facade

use std::sync::Arc;

use crate::domain::{Embedding, Vector};
use crate::error::Result;
use crate::ports::{EmbeddingGenerator, EmbeddingRepository, VectorStore, SearchParams, SimilarityResult};

/// Main service for embedding operations
pub struct EmbeddingService {
    generator: Arc<dyn EmbeddingGenerator>,
    repository: Arc<dyn EmbeddingRepository>,
    vector_store: Arc<dyn VectorStore>,
}

impl EmbeddingService {
    /// Create a new embedding service
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

    /// Create an embedding from text
    pub async fn create_embedding(&self, text: &str) -> Result<Embedding> {
        let embedding = self.generator.generate_from_text(text).await?;
        self.repository.save(&embedding).await?;
        self.vector_store.index(&embedding).await?;
        Ok(embedding)
    }

    /// Create embeddings from multiple texts (batch)
    pub async fn create_embeddings_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let embeddings = self.generator.generate_batch(texts).await?;

        for embedding in &embeddings {
            self.repository.save(embedding).await?;
            self.vector_store.index(embedding).await?;
        }

        Ok(embeddings)
    }

    /// Find similar embeddings to a text query
    pub async fn find_similar(&self, query_text: &str, params: SearchParams) -> Result<Vec<SimilarityResult>> {
        let query_embedding = self.generator.generate_from_text(query_text).await?;
        self.vector_store.search(query_embedding.vector(), params).await
    }

    /// Find similar embeddings to a vector
    pub async fn find_similar_vector(&self, query: &Vector, params: SearchParams) -> Result<Vec<SimilarityResult>> {
        self.vector_store.search(query, params).await
    }

    /// Get embedding by ID
    pub async fn get_embedding(&self, id: &str) -> Result<Option<Embedding>> {
        self.repository.find_by_id(id).await
    }

    /// List all embeddings
    pub async fn list_embeddings(&self) -> Result<Vec<Embedding>> {
        self.repository.list().await
    }

    /// Delete an embedding
    pub async fn delete_embedding(&self, id: &str) -> Result<()> {
        self.repository.delete(id).await?;
        self.vector_store.remove(id).await?;
        Ok(())
    }

    /// Get statistics
    pub async fn statistics(&self) -> Result<ServiceStatistics> {
        let total_embeddings = self.repository.count().await?;
        let indexed_embeddings = self.vector_store.count().await?;

        Ok(ServiceStatistics {
            total_embeddings,
            indexed_embeddings,
            generator_model: self.generator.model_name().to_string(),
            dimensions: self.generator.dimensions(),
        })
    }
}

/// Service statistics
#[derive(Debug, Clone)]
pub struct ServiceStatistics {
    pub total_embeddings: usize,
    pub indexed_embeddings: usize,
    pub generator_model: String,
    pub dimensions: usize,
}
