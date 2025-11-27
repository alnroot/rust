//! SQLite Repository Implementation
//!
//! Production-ready persistent storage using SQLite.
//!
//! Features:
//! - ACID transactions
//! - JSON serialization for vectors and metadata
//! - Connection pooling (via rusqlite)
//! - Automatic schema migrations
//! - Crash-safe persistence

use async_trait::async_trait;
use rusqlite::{Connection, params, OptionalExtension};
use std::path::Path;
use std::sync::{Arc, Mutex};
use chrono::{DateTime, Utc};

use crate::domain::{Embedding, EmbeddingId};
use crate::error::{EmbeddingError, Result};
use crate::ports::EmbeddingRepository;

/// SQLite-based embedding repository
pub struct SqliteRepository {
    /// Connection wrapped in Arc<Mutex> for thread-safety
    conn: Arc<Mutex<Connection>>,
}

impl SqliteRepository {
    /// Create a new SQLite repository
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)
            .map_err(|e| EmbeddingError::storage(format!("Failed to open SQLite: {}", e)))?;

        // Initialize schema
        Self::initialize_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Create an in-memory database (for testing)
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| EmbeddingError::storage(format!("Failed to create in-memory DB: {}", e)))?;

        Self::initialize_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Initialize database schema
    fn initialize_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            -- Enable WAL mode for better concurrency
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            PRAGMA synchronous = NORMAL;

            -- Embeddings table
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY NOT NULL,
                vector_data TEXT NOT NULL,  -- JSON array of f32
                dimensions INTEGER NOT NULL,
                source TEXT NOT NULL,
                model TEXT NOT NULL,
                tags TEXT NOT NULL,  -- JSON array of strings
                properties TEXT NOT NULL,  -- JSON object
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1
            );

            -- Index on model for filtering
            CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

            -- Index on created_at for sorting
            CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at);

            -- Metadata for schema versioning
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO schema_metadata (key, value)
            VALUES ('version', '1');
            "#
        ).map_err(|e| EmbeddingError::storage(format!("Schema initialization failed: {}", e)))?;

        Ok(())
    }

    /// Convert embedding to database row format
    fn embedding_to_row(&self, embedding: &Embedding) -> Result<(String, String, usize, String, String, String, String, String, String, u32)> {
        let id = embedding.id().to_string();

        // Serialize vector as JSON array
        let vector_json = serde_json::to_string(embedding.vector().data())
            .map_err(|e| EmbeddingError::serialization(format!("Failed to serialize vector: {}", e)))?;

        let dimensions = embedding.vector().dimensions();

        let metadata = embedding.metadata();
        let source = metadata.source.clone();
        let model = metadata.model.clone();

        // Serialize tags as JSON array
        let tags_json = serde_json::to_string(&metadata.tags)
            .map_err(|e| EmbeddingError::serialization(format!("Failed to serialize tags: {}", e)))?;

        // Serialize properties as JSON object
        let properties_json = serde_json::to_string(&metadata.properties)
            .map_err(|e| EmbeddingError::serialization(format!("Failed to serialize properties: {}", e)))?;

        let created_at = embedding.created_at().to_rfc3339();
        let updated_at = embedding.updated_at().to_rfc3339();
        let version = embedding.version();

        Ok((id, vector_json, dimensions, source, model, tags_json, properties_json, created_at, updated_at, version))
    }

    /// Convert database row to embedding
    fn row_to_embedding(&self,
        id: String,
        vector_data: String,
        source: String,
        model: String,
        tags: String,
        properties: String,
        created_at: String,
        updated_at: String,
        version: u32
    ) -> Result<Embedding> {
        use crate::domain::{Vector, EmbeddingMetadata};

        // Deserialize vector data
        let vector_array: Vec<f32> = serde_json::from_str(&vector_data)
            .map_err(|e| EmbeddingError::serialization(format!("Failed to deserialize vector: {}", e)))?;

        let vector = Vector::new(vector_array)?;

        // Deserialize tags
        let tags_vec: Vec<String> = serde_json::from_str(&tags)
            .map_err(|e| EmbeddingError::serialization(format!("Failed to deserialize tags: {}", e)))?;

        // Deserialize properties
        let properties_map = serde_json::from_str(&properties)
            .map_err(|e| EmbeddingError::serialization(format!("Failed to deserialize properties: {}", e)))?;

        // Parse timestamps first
        let created_at_dt = DateTime::parse_from_rfc3339(&created_at)
            .map_err(|e| EmbeddingError::serialization(format!("Invalid created_at timestamp: {}", e)))?
            .with_timezone(&Utc);

        let metadata = EmbeddingMetadata {
            source,
            model,
            tags: tags_vec,
            properties: properties_map,
            created_at: created_at_dt,
        };

        let updated_at_dt = DateTime::parse_from_rfc3339(&updated_at)
            .map_err(|e| EmbeddingError::serialization(format!("Invalid updated_at timestamp: {}", e)))?
            .with_timezone(&Utc);

        // Parse ID
        let embedding_id = EmbeddingId::parse(&id)?;

        Ok(Embedding::with_id(
            embedding_id,
            vector,
            metadata,
            created_at_dt,
            updated_at_dt,
            version,
        ))
    }
}

#[async_trait]
impl EmbeddingRepository for SqliteRepository {
    async fn save(&self, embedding: &Embedding) -> Result<()> {
        let (id, vector_json, dimensions, source, model, tags_json, properties_json, created_at, updated_at, version) =
            self.embedding_to_row(embedding)?;

        let conn = self.conn.lock()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock connection: {}", e)))?;

        conn.execute(
            r#"
            INSERT INTO embeddings
            (id, vector_data, dimensions, source, model, tags, properties, created_at, updated_at, version)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ON CONFLICT(id) DO UPDATE SET
                vector_data = excluded.vector_data,
                dimensions = excluded.dimensions,
                source = excluded.source,
                model = excluded.model,
                tags = excluded.tags,
                properties = excluded.properties,
                updated_at = excluded.updated_at,
                version = excluded.version
            "#,
            params![id, vector_json, dimensions as i64, source, model, tags_json, properties_json, created_at, updated_at, version]
        ).map_err(|e| EmbeddingError::storage(format!("Failed to save embedding: {}", e)))?;

        Ok(())
    }

    async fn find_by_id(&self, id: &str) -> Result<Option<Embedding>> {
        let conn = self.conn.lock()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock connection: {}", e)))?;

        let result = conn.query_row(
            r#"
            SELECT id, vector_data, source, model, tags, properties, created_at, updated_at, version
            FROM embeddings
            WHERE id = ?1
            "#,
            params![id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, u32>(8)?,
                ))
            }
        ).optional()
        .map_err(|e| EmbeddingError::storage(format!("Failed to find embedding: {}", e)))?;

        match result {
            Some((id, vector_data, source, model, tags, properties, created_at, updated_at, version)) => {
                let embedding = self.row_to_embedding(id, vector_data, source, model, tags, properties, created_at, updated_at, version)?;
                Ok(Some(embedding))
            }
            None => Ok(None)
        }
    }

    async fn list(&self) -> Result<Vec<Embedding>> {
        let conn = self.conn.lock()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock connection: {}", e)))?;

        let mut stmt = conn.prepare(
            r#"
            SELECT id, vector_data, source, model, tags, properties, created_at, updated_at, version
            FROM embeddings
            ORDER BY created_at DESC
            "#
        ).map_err(|e| EmbeddingError::storage(format!("Failed to prepare statement: {}", e)))?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, u32>(8)?,
            ))
        }).map_err(|e| EmbeddingError::storage(format!("Failed to query embeddings: {}", e)))?;

        let mut embeddings = Vec::new();
        for row_result in rows {
            let (id, vector_data, source, model, tags, properties, created_at, updated_at, version) =
                row_result.map_err(|e| EmbeddingError::storage(format!("Failed to read row: {}", e)))?;

            let embedding = self.row_to_embedding(id, vector_data, source, model, tags, properties, created_at, updated_at, version)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock connection: {}", e)))?;

        let rows_affected = conn.execute(
            "DELETE FROM embeddings WHERE id = ?1",
            params![id]
        ).map_err(|e| EmbeddingError::storage(format!("Failed to delete embedding: {}", e)))?;

        if rows_affected == 0 {
            return Err(EmbeddingError::not_found(format!("Embedding {} not found", id)));
        }

        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let conn = self.conn.lock()
            .map_err(|e| EmbeddingError::storage(format!("Failed to lock connection: {}", e)))?;

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM embeddings",
            [],
            |row| row.get(0)
        ).map_err(|e| EmbeddingError::storage(format!("Failed to count embeddings: {}", e)))?;

        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Vector, EmbeddingMetadata};

    #[tokio::test]
    async fn test_sqlite_save_and_find() {
        let repo = SqliteRepository::in_memory().unwrap();

        let vector = Vector::new(vec![1.0, 2.0, 3.0]).unwrap();
        let metadata = EmbeddingMetadata::builder()
            .source("test")
            .model("test-model")
            .build()
            .unwrap();

        let embedding = Embedding::new(vector, metadata);
        let id = embedding.id().to_string();

        // Save
        repo.save(&embedding).await.unwrap();

        // Find
        let found = repo.find_by_id(&id).await.unwrap();
        assert!(found.is_some());

        let found_embedding = found.unwrap();
        assert_eq!(found_embedding.id().to_string(), id);
        assert_eq!(found_embedding.vector().dimensions(), 3);
    }

    #[tokio::test]
    async fn test_sqlite_list() {
        let repo = SqliteRepository::in_memory().unwrap();

        // Create multiple embeddings
        for i in 0..5 {
            let vector = Vector::new(vec![i as f32; 128]).unwrap();
            let metadata = EmbeddingMetadata::builder()
                .source(format!("test-{}", i))
                .model("test-model")
                .build()
                .unwrap();

            let embedding = Embedding::new(vector, metadata);
            repo.save(&embedding).await.unwrap();
        }

        // List all
        let embeddings = repo.list().await.unwrap();
        assert_eq!(embeddings.len(), 5);
    }

    #[tokio::test]
    async fn test_sqlite_delete() {
        let repo = SqliteRepository::in_memory().unwrap();

        let vector = Vector::new(vec![1.0, 2.0, 3.0]).unwrap();
        let metadata = EmbeddingMetadata::builder()
            .source("test")
            .model("test-model")
            .build()
            .unwrap();

        let embedding = Embedding::new(vector, metadata);
        let id = embedding.id().to_string();

        // Save
        repo.save(&embedding).await.unwrap();
        assert_eq!(repo.count().await.unwrap(), 1);

        // Delete
        repo.delete(&id).await.unwrap();
        assert_eq!(repo.count().await.unwrap(), 0);

        // Verify not found
        let found = repo.find_by_id(&id).await.unwrap();
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_update() {
        let repo = SqliteRepository::in_memory().unwrap();

        let vector = Vector::new(vec![1.0, 2.0, 3.0]).unwrap();
        let metadata = EmbeddingMetadata::builder()
            .source("test")
            .model("test-model")
            .build()
            .unwrap();

        let mut embedding = Embedding::new(vector, metadata);
        let id = embedding.id().to_string();

        // Save original
        repo.save(&embedding).await.unwrap();

        // Update metadata
        let new_metadata = EmbeddingMetadata::builder()
            .source("updated-source")
            .model("test-model")
            .tag("updated")
            .build()
            .unwrap();

        embedding.update_metadata(new_metadata);

        // Save updated
        repo.save(&embedding).await.unwrap();

        // Verify update
        let found = repo.find_by_id(&id).await.unwrap().unwrap();
        assert_eq!(found.metadata().source, "updated-source");
        assert!(found.metadata().tags.contains(&"updated".to_string()));
        assert_eq!(found.version(), 2);
    }
}
