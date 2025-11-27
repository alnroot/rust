//! Domain layer - Core business logic
//!
//! Contains entities, value objects, and domain events.

pub mod entities;
pub mod value_objects;
pub mod log_entry;

pub use entities::{Embedding, EmbeddingId, EmbeddingCollection};
pub use value_objects::{Vector, EmbeddingMetadata, EmbeddingMetadataBuilder};
pub use log_entry::{LogEntry, LogLevel, TimeWindow, LogEntryBuilder};
