//! API Authentication
//!
//! Provides API key authentication for protected endpoints.

use axum::{
    async_trait,
    extract::FromRequestParts,
    http::{header::AUTHORIZATION, request::Parts, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use serde::Serialize;
use std::collections::HashSet;
use std::sync::Arc;

/// API key configuration
#[derive(Debug, Clone)]
pub struct ApiKeyConfig {
    /// Valid API keys (in production, use a secure store)
    valid_keys: HashSet<String>,
    /// Whether authentication is enabled
    enabled: bool,
}

impl Default for ApiKeyConfig {
    fn default() -> Self {
        Self {
            valid_keys: HashSet::new(),
            enabled: false,
        }
    }
}

impl ApiKeyConfig {
    /// Create a new config with authentication enabled
    pub fn new(keys: Vec<String>) -> Self {
        Self {
            valid_keys: keys.into_iter().collect(),
            enabled: true,
        }
    }

    /// Create config from environment variable
    /// Expects comma-separated list of API keys in EMBEDDING_API_KEYS
    pub fn from_env() -> Self {
        match std::env::var("EMBEDDING_API_KEYS") {
            Ok(keys) if !keys.is_empty() => {
                let valid_keys: HashSet<String> = keys
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                if valid_keys.is_empty() {
                    tracing::warn!("EMBEDDING_API_KEYS is set but contains no valid keys, auth disabled");
                    Self::default()
                } else {
                    tracing::info!("API key authentication enabled with {} keys", valid_keys.len());
                    Self {
                        valid_keys,
                        enabled: true,
                    }
                }
            }
            _ => {
                tracing::warn!("EMBEDDING_API_KEYS not set, API authentication is DISABLED");
                Self::default()
            }
        }
    }

    /// Check if authentication is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Validate an API key
    pub fn validate(&self, key: &str) -> bool {
        if !self.enabled {
            return true;
        }
        self.valid_keys.contains(key)
    }

    /// Add a key (for testing)
    pub fn add_key(&mut self, key: String) {
        self.valid_keys.insert(key);
        self.enabled = true;
    }
}

/// State extension for API key validation (used via Extension layer)
#[derive(Clone)]
pub struct ApiKeyState(pub Arc<ApiKeyConfig>);

/// API Key extractor - validates requests have a valid API key
/// Add this extractor to any handler that requires authentication
#[derive(Debug, Clone)]
pub struct RequireApiKey;

/// Error response for authentication failures
#[derive(Debug, Serialize)]
pub struct AuthError {
    pub error: String,
    pub message: String,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let status = StatusCode::UNAUTHORIZED;
        let body = Json(self);
        (status, body).into_response()
    }
}

#[async_trait]
impl<S> FromRequestParts<S> for RequireApiKey
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Try to get ApiKeyState from extensions
        let config = parts
            .extensions
            .get::<ApiKeyState>()
            .map(|s| s.0.clone());

        // If no config or auth disabled, allow through
        let config = match config {
            Some(c) if c.is_enabled() => c,
            _ => return Ok(RequireApiKey),
        };

        // Extract Authorization header
        let auth_header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok());

        let api_key = match auth_header {
            Some(header) => {
                // Support both "Bearer <key>" and raw "<key>" formats
                if header.starts_with("Bearer ") {
                    header.trim_start_matches("Bearer ").to_string()
                } else {
                    header.to_string()
                }
            }
            None => {
                return Err(AuthError {
                    error: "missing_api_key".to_string(),
                    message: "Authorization header required. Use 'Authorization: Bearer <api_key>'".to_string(),
                });
            }
        };

        if config.validate(&api_key) {
            Ok(RequireApiKey)
        } else {
            Err(AuthError {
                error: "invalid_api_key".to_string(),
                message: "Invalid API key".to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_config_default() {
        let config = ApiKeyConfig::default();
        assert!(!config.is_enabled());
        assert!(config.validate("any_key")); // Disabled means all pass
    }

    #[test]
    fn test_api_key_config_enabled() {
        let config = ApiKeyConfig::new(vec!["secret123".to_string(), "secret456".to_string()]);
        assert!(config.is_enabled());
        assert!(config.validate("secret123"));
        assert!(config.validate("secret456"));
        assert!(!config.validate("wrong_key"));
    }
}
