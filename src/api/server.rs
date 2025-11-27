//! API Server

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;

use crate::application::EmbeddingService;
use crate::error::Result;
use super::handlers::AppState;
use super::routes::create_router;

/// API Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Host to bind to
    pub host: String,

    /// Port to bind to
    pub port: u16,

    /// Application version
    pub version: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            // Bind to localhost by default for security
            // Use 0.0.0.0 in production to accept external connections
            host: "127.0.0.1".to_string(),
            // Standard port 8080 for HTTP alternative services
            // Override via PORT environment variable
            port: 8080,
            // Version from Cargo.toml, embedded at compile time
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// API Server
pub struct ApiServer {
    pub config: ServerConfig,
    pub service: Arc<EmbeddingService>,
}

impl ApiServer {
    /// Create a new API server
    pub fn new(config: ServerConfig, service: Arc<EmbeddingService>) -> Self {
        Self { config, service }
    }

    /// Start the server
    pub async fn serve(self) -> Result<()> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| crate::error::EmbeddingError::config(format!("Invalid address: {}", e)))?;

        tracing::info!("🚀 Starting API server on http://{}", addr);
        tracing::info!("📚 API documentation: http://{}/api/v1/health", addr);
        tracing::info!("📊 Health check: http://{}/api/v1/health", addr);

        // Create application state
        let state = AppState {
            service: self.service,
            start_time: Instant::now(),
            version: self.config.version.clone(),
        };

        // Build router
        let app = create_router(state);

        // Create TCP listener
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| crate::error::EmbeddingError::storage(format!("Failed to bind: {}", e)))?;

        tracing::info!("✅ Server is ready to accept connections");

        // Serve with graceful shutdown
        axum::serve(listener, app)
            .await
            .map_err(|e| crate::error::EmbeddingError::processing(format!("Server error: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
    }
}
