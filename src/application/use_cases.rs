//! Application Use Cases
//!
//! This module implements the Command/Query Separation (CQS) pattern.
//! - Commands: Modify state (create, update, delete)
//! - Queries: Read-only operations (get, list, search)
//!
//! Each use case is a focused operation that orchestrates domain logic
//! and infrastructure concerns.

use std::sync::Arc;
use crate::domain::{Embedding, Vector};
use crate::error::Result;
use crate::ports::{
    EmbeddingGenerator, EmbeddingRepository, VectorStore,
    SearchParams, SimilarityResult,
};

// ============================================================================
// COMMAND USE CASES (State-modifying operations)
// ============================================================================

/// Create a single embedding from text
pub struct CreateEmbeddingUseCase {
    generator: Arc<dyn EmbeddingGenerator>,
    repository: Arc<dyn EmbeddingRepository>,
    vector_store: Arc<dyn VectorStore>,
}

impl CreateEmbeddingUseCase {
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

    pub async fn execute(&self, text: &str) -> Result<Embedding> {
        // Generate embedding
        let embedding = self.generator.generate_from_text(text).await?;

        // Validate
        embedding.validate()?;

        // Persist
        self.repository.save(&embedding).await?;

        // Index for search
        self.vector_store.index(&embedding).await?;

        Ok(embedding)
    }
}

/// Create multiple embeddings in batch
pub struct BatchCreateUseCase {
    generator: Arc<dyn EmbeddingGenerator>,
    repository: Arc<dyn EmbeddingRepository>,
    vector_store: Arc<dyn VectorStore>,
}

impl BatchCreateUseCase {
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

    pub async fn execute(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        // Generate embeddings in batch
        let embeddings = self.generator.generate_batch(texts).await?;

        // Validate all embeddings
        for embedding in &embeddings {
            embedding.validate()?;
        }

        // Persist all
        for embedding in &embeddings {
            self.repository.save(embedding).await?;
            self.vector_store.index(embedding).await?;
        }

        Ok(embeddings)
    }
}

/// Delete an embedding by ID
pub struct DeleteEmbeddingUseCase {
    repository: Arc<dyn EmbeddingRepository>,
    vector_store: Arc<dyn VectorStore>,
}

impl DeleteEmbeddingUseCase {
    pub fn new(
        repository: Arc<dyn EmbeddingRepository>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        Self {
            repository,
            vector_store,
        }
    }

    pub async fn execute(&self, id: &str) -> Result<()> {
        // Check if exists
        let exists = self.repository.find_by_id(id).await?.is_some();
        if !exists {
            return Err(crate::error::EmbeddingError::not_found(
                format!("Embedding with id {} not found", id)
            ));
        }

        // Delete from repository
        self.repository.delete(id).await?;

        // Remove from vector store
        self.vector_store.remove(id).await?;

        Ok(())
    }
}

// ============================================================================
// QUERY USE CASES (Read-only operations)
// ============================================================================

/// Get a single embedding by ID
pub struct GetEmbeddingUseCase {
    repository: Arc<dyn EmbeddingRepository>,
}

impl GetEmbeddingUseCase {
    pub fn new(repository: Arc<dyn EmbeddingRepository>) -> Self {
        Self { repository }
    }

    pub async fn execute(&self, id: &str) -> Result<Option<Embedding>> {
        self.repository.find_by_id(id).await
    }
}

/// List all embeddings
pub struct ListEmbeddingsUseCase {
    repository: Arc<dyn EmbeddingRepository>,
}

impl ListEmbeddingsUseCase {
    pub fn new(repository: Arc<dyn EmbeddingRepository>) -> Self {
        Self { repository }
    }

    pub async fn execute(&self) -> Result<Vec<Embedding>> {
        self.repository.list().await
    }
}

/// Find similar embeddings by text query
pub struct FindSimilarUseCase {
    generator: Arc<dyn EmbeddingGenerator>,
    vector_store: Arc<dyn VectorStore>,
}

impl FindSimilarUseCase {
    pub fn new(
        generator: Arc<dyn EmbeddingGenerator>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        Self {
            generator,
            vector_store,
        }
    }

    /// Find similar embeddings by text query
    pub async fn execute_text(
        &self,
        query_text: &str,
        params: SearchParams,
    ) -> Result<Vec<SimilarityResult>> {
        // Generate embedding for query
        let query_embedding = self.generator.generate_from_text(query_text).await?;

        // Search for similar embeddings
        self.vector_store.search(query_embedding.vector(), params).await
    }

    /// Find similar embeddings by vector directly
    pub async fn execute_vector(
        &self,
        query_vector: &Vector,
        params: SearchParams,
    ) -> Result<Vec<SimilarityResult>> {
        self.vector_store.search(query_vector, params).await
    }
}

/// Get statistics about the embedding system
pub struct GetStatisticsUseCase {
    repository: Arc<dyn EmbeddingRepository>,
    vector_store: Arc<dyn VectorStore>,
    generator: Arc<dyn EmbeddingGenerator>,
}

impl GetStatisticsUseCase {
    pub fn new(
        repository: Arc<dyn EmbeddingRepository>,
        vector_store: Arc<dyn VectorStore>,
        generator: Arc<dyn EmbeddingGenerator>,
    ) -> Self {
        Self {
            repository,
            vector_store,
            generator,
        }
    }

    pub async fn execute(&self) -> Result<Statistics> {
        let total_embeddings = self.repository.count().await?;
        let indexed_embeddings = self.vector_store.count().await?;
        let generator_model = self.generator.model_name().to_string();
        let dimensions = self.generator.dimensions();

        Ok(Statistics {
            total_embeddings,
            indexed_embeddings,
            generator_model,
            dimensions,
        })
    }
}

/// Statistics about the embedding system
#[derive(Debug, Clone)]
pub struct Statistics {
    pub total_embeddings: usize,
    pub indexed_embeddings: usize,
    pub generator_model: String,
    pub dimensions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::EmbeddingMetadata;
    use crate::infrastructure::{
        InMemoryRepository,
        InMemoryVectorStore,
    };

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
    async fn test_create_embedding_use_case() {
        let generator = Arc::new(MockGenerator { dimensions: 128 });
        let repository = Arc::new(InMemoryRepository::new());
        let vector_store = Arc::new(InMemoryVectorStore::new());

        let use_case = CreateEmbeddingUseCase::new(
            generator.clone(),
            repository.clone(),
            vector_store.clone(),
        );

        let embedding = use_case.execute("test text").await.unwrap();
        assert_eq!(embedding.vector().dimensions(), 128);

        // Verify it was saved
        let found = repository.find_by_id(&embedding.id().to_string()).await.unwrap();
        assert!(found.is_some());
    }

    #[tokio::test]
    async fn test_batch_create_use_case() {
        let generator = Arc::new(MockGenerator { dimensions: 128 });
        let repository = Arc::new(InMemoryRepository::new());
        let vector_store = Arc::new(InMemoryVectorStore::new());

        let use_case = BatchCreateUseCase::new(
            generator.clone(),
            repository.clone(),
            vector_store.clone(),
        );

        let texts = vec!["text1".to_string(), "text2".to_string(), "text3".to_string()];
        let embeddings = use_case.execute(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(repository.count().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_find_similar_use_case() {
        let generator = Arc::new(MockGenerator { dimensions: 128 });
        let repository = Arc::new(InMemoryRepository::new());
        let vector_store = Arc::new(InMemoryVectorStore::new());

        // Create some embeddings first
        let create_use_case = CreateEmbeddingUseCase::new(
            generator.clone(),
            repository.clone(),
            vector_store.clone(),
        );

        create_use_case.execute("hello world").await.unwrap();
        create_use_case.execute("goodbye world").await.unwrap();

        // Find similar
        let find_use_case = FindSimilarUseCase::new(generator.clone(), vector_store.clone());

        let results = find_use_case
            .execute_text("hello", SearchParams::builder().k(5).build())
            .await
            .unwrap();

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_get_statistics_use_case() {
        let generator = Arc::new(MockGenerator { dimensions: 128 });
        let repository = Arc::new(InMemoryRepository::new());
        let vector_store = Arc::new(InMemoryVectorStore::new());

        let use_case = GetStatisticsUseCase::new(
            repository.clone(),
            vector_store.clone(),
            generator.clone(),
        );

        let stats = use_case.execute().await.unwrap();
        assert_eq!(stats.dimensions, 128);
        assert_eq!(stats.generator_model, "mock-model");
    }
}
