//! # Rust Embeddings Library
//!
//! A clean, well-architected library for managing embeddings with design patterns.
//! This library follows SOLID principles and clean architecture patterns.
//!
//! ## Architecture
//!
//! The library is organized into layers:
//! - **Domain**: Core business logic and entities
//! - **Application**: Use cases and application services
//! - **Infrastructure**: Concrete implementations
//! - **Ports**: Interfaces and abstractions
//!
//! ## Design Patterns
//!
//! - Repository Pattern: For data access abstraction
//! - Strategy Pattern: For different embedding algorithms
//! - Builder Pattern: For complex object construction
//! - Factory Pattern: For object creation
//! - Dependency Injection: For loose coupling

pub mod domain;
pub mod application;
pub mod infrastructure;
pub mod ports;
pub mod error;
pub mod config;
pub mod concurrency;

// Re-export commonly used types
pub use domain::{Embedding, EmbeddingMetadata, Vector};
pub use error::{EmbeddingError, Result};
pub use ports::{EmbeddingRepository, EmbeddingGenerator, VectorStore};
pub use application::services::EmbeddingService;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::domain::*;
    pub use crate::error::{EmbeddingError, Result};
    pub use crate::ports::*;
    pub use crate::application::services::*;
    pub use crate::application::use_cases::*;
}
