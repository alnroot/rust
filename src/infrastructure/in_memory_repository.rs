//! In-Memory Repository Implementation

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::domain::Embedding;
use crate::error::Result;
use crate::ports::EmbeddingRepository;

/// In-memory implementation of EmbeddingRepository
pub struct InMemoryRepository {
    storage: Arc<RwLock<HashMap<String, Embedding>>>,
}

impl InMemoryRepository {
    /// Create a new in-memory repository
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EmbeddingRepository for InMemoryRepository {
    async fn save(&self, embedding: &Embedding) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage.insert(embedding.id().to_string(), embedding.clone());
        Ok(())
    }

    async fn find_by_id(&self, id: &str) -> Result<Option<Embedding>> {
        let storage = self.storage.read().await;
        Ok(storage.get(id).cloned())
    }

    async fn list(&self) -> Result<Vec<Embedding>> {
        let storage = self.storage.read().await;
        Ok(storage.values().cloned().collect())
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage.remove(id);
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let storage = self.storage.read().await;
        Ok(storage.len())
    }
}
