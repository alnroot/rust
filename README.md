# Rust Embeddings Library

Una biblioteca Rust bien arquitecturada para gestionar embeddings con clean code y patrones de diseño de software.

## 🎯 Características

- ✅ **Arquitectura Limpia**: Separación de capas (Domain, Application, Infrastructure, Ports)
- ✅ **Principios SOLID**: Código mantenible y extensible
- ✅ **Patrones de Diseño**: Repository, Strategy, Factory, Builder, Dependency Injection
- ✅ **Type-Safe**: Aprovecha el sistema de tipos de Rust
- ✅ **Async/Await**: Soporte completo para operaciones asíncronas
- ✅ **Concurrencia y Paralelismo**: Threads, Channels, Rayon, Arc, RwLock
- ✅ **Gestión de Memoria Explícita**: Stack, Heap, Static memory bien documentados
- ✅ **Testeable**: Interfaces bien definidas para testing
- ✅ **Documentado**: Documentación completa con ejemplos

## 🏗️ Arquitectura

La biblioteca sigue el patrón de **Arquitectura Hexagonal** (Ports and Adapters):

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│  (Use Cases, Services, Orchestration)                    │
└────────────────┬────────────────────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
┌────────▼──────┐  ┌────▼──────────┐
│  Domain Layer │  │  Ports Layer   │
│  (Entities,   │  │  (Interfaces,  │
│   Values,     │  │   Abstractions)│
│   Events)     │  │                │
└───────────────┘  └────┬───────────┘
                        │
              ┌─────────▼────────────┐
              │ Infrastructure Layer │
              │  (Implementations)   │
              └──────────────────────┘
```

### Capas

#### 1. **Domain Layer** (`src/domain/`)
Contiene la lógica de negocio central:
- **Entities**: `Embedding`, `EmbeddingCollection`, `EmbeddingId`
- **Value Objects**: `Vector`, `EmbeddingMetadata`
- **Domain Events**: `EmbeddingCreated`, `EmbeddingUpdated`, etc.

#### 2. **Ports Layer** (`src/ports/`)
Define las abstracciones (traits):
- `EmbeddingRepository`: Abstracción de persistencia
- `EmbeddingGenerator`: Estrategia para generar embeddings
- `VectorStore`: Búsqueda de similitud
- `EventPublisher`: Publicación de eventos

#### 3. **Application Layer** (`src/application/`)
Orquesta los casos de uso:
- **Services**: `EmbeddingService` (Facade Pattern)
- **Use Cases**: `CreateEmbeddingUseCase`, `FindSimilarUseCase`, etc.

#### 4. **Infrastructure Layer** (`src/infrastructure/`)
Implementaciones concretas:
- `InMemoryEmbeddingRepository`
- `RandomEmbeddingGenerator`
- `InMemoryVectorStore`
- `InMemoryEventPublisher`

## 🎨 Patrones de Diseño Implementados

### 1. Repository Pattern
Abstrae el acceso a datos:

```rust
#[async_trait]
pub trait EmbeddingRepository: Send + Sync {
    async fn save(&self, embedding: &Embedding) -> Result<()>;
    async fn find_by_id(&self, id: &EmbeddingId) -> Result<Option<Embedding>>;
    // ...
}
```

### 2. Strategy Pattern
Diferentes algoritmos de generación:

```rust
#[async_trait]
pub trait EmbeddingGenerator: Send + Sync {
    async fn generate_from_text(&self, text: &str) -> Result<Embedding>;
    fn model_name(&self) -> &str;
    fn dimensions(&self) -> usize;
}
```

### 3. Builder Pattern
Construcción fluida de objetos:

```rust
let metadata = EmbeddingMetadata::builder()
    .source("document.txt")
    .model("text-embedding-ada-002")
    .tag("production")
    .property("version", "1.0")
    .build()?;
```

### 4. Factory Pattern
Creación de generadores:

```rust
let config = GeneratorConfig::builder()
    .model("random")
    .dimensions(128)
    .build()?;

let generator = factory.create(config)?;
```

### 5. Facade Pattern
Interfaz unificada:

```rust
let service = EmbeddingService::new(repository, generator, vector_store);
let embedding = service.create_from_text("Hello, world!").await?;
```

### 6. Dependency Injection
Desacoplamiento mediante traits:

```rust
pub struct EmbeddingService {
    repository: Arc<dyn EmbeddingRepository>,
    generator: Arc<dyn EmbeddingGenerator>,
    vector_store: Arc<dyn VectorStore>,
}
```

## 🚀 Uso Rápido

### Instalación

Añade a tu `Cargo.toml`:

```toml
[dependencies]
rust-embeddings = "0.1.0"
```

### Ejemplo Básico

```rust
use std::sync::Arc;
use rust_embeddings::prelude::*;
use rust_embeddings::infrastructure::*;
use rust_embeddings::ports::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Configurar componentes
    let repository = Arc::new(InMemoryEmbeddingRepository::new());
    let generator = Arc::new(RandomEmbeddingGenerator::new(
        "model-v1".to_string(),
        128,
    ));
    let vector_store = Arc::new(InMemoryVectorStore::new_cosine());

    // Crear servicio
    let service = EmbeddingService::new(
        repository,
        generator,
        vector_store,
    );

    // Crear embedding
    let embedding = service.create_from_text("Rust programming").await?;
    println!("Created embedding: {}", embedding.id());

    // Buscar similares
    let params = SearchParams::builder().k(5).build();
    let results = service
        .search_similar_text("programming languages", params)
        .await?;

    for result in results {
        println!("Similarity: {:.4}", result.score);
    }

    Ok(())
}
```

## 📚 Ejemplos

Ejecuta los ejemplos incluidos:

```bash
# Ejemplo básico de uso
cargo run --example basic_usage

# Implementación personalizada
cargo run --example custom_storage

# Memoria y concurrencia (Stack, Heap, Static, Threads, Rayon)
cargo run --example memory_and_concurrency
```

## 🧠 Gestión de Memoria y Concurrencia

Esta biblioteca demuestra explícitamente los conceptos de memoria de Rust y patrones de concurrencia:

### Tipos de Memoria

#### 1. **Static Memory** (Segmento de datos)
Constantes compiladas en el binario:
```rust
pub const DEFAULT_EMBEDDING_DIMENSIONS: usize = 384;
pub const MAX_BATCH_SIZE: usize = 1000;
```
- **Ubicación**: Binario (`.data` / `.rodata`)
- **Lifetime**: `'static` (toda la duración del programa)
- **Costo**: Zero-cost (compile time)

#### 2. **Stack Memory** (Pila)
Variables locales y parámetros:
```rust
let dimensions: usize = 128;  // Stack: 8 bytes
let threshold: f32 = 0.85;     // Stack: 4 bytes
```
- **Ubicación**: Stack de cada thread (~2MB en Linux)
- **Velocidad**: Muy rápida (mover puntero de pila)
- **Limitación**: Tamaño fijo, stack overflow si se excede

#### 3. **Heap Memory** (Montículo)
Datos dinámicos:
```rust
let embeddings = Vec::new();           // Heap allocation
let shared = Arc::new(data);           // Heap + ref counting
let cached = HashMap::new();           // Heap allocation
```
- **Ubicación**: Heap del proceso (puede crecer a GBs)
- **Flexibilidad**: Tamaño dinámico en runtime
- **Costo**: Allocación más lenta que stack

### Patrones de Concurrencia

#### 1. **Thread-based Parallelism** (src/concurrency.rs:62)
Threads explícitos del OS:
```rust
let shared_data = Arc::new(embeddings);  // Heap compartido
thread::spawn(move || {
    // Cada thread tiene su propio stack
    process_data(&shared_data)
});
```

#### 2. **Channel Communication** (src/concurrency.rs:125)
Message passing entre threads:
```rust
let (sender, receiver) = channel::bounded(100);
thread::spawn(move || {
    while let Ok(msg) = receiver.recv() {
        process(msg);  // Ownership transferido
    }
});
```

#### 3. **Data Parallelism con Rayon** (src/concurrency.rs:217)
Paralelización automática:
```rust
embeddings.par_iter()
    .map(|e| process(e))
    .collect()  // Work-stealing thread pool
```

#### 4. **RwLock** (src/infrastructure/repository.rs:14)
Múltiples lectores O un escritor:
```rust
storage: RwLock<HashMap<K, V>>  // Thread-safe

let data = storage.read().unwrap();   // Multiple readers
let mut data = storage.write().unwrap(); // Single writer
```

#### 5. **Async/Await** (src/application/services.rs)
Concurrencia para I/O:
```rust
async fn create(&self, text: &str) -> Result<Embedding> {
    let emb = self.generator.generate(text).await?;
    self.repository.save(&emb).await?;
    Ok(emb)
}
```

### Documentación Detallada

Para una explicación completa de los patrones de memoria y concurrencia, ver:
- **[MEMORY_AND_CONCURRENCY.md](MEMORY_AND_CONCURRENCY.md)**: Guía detallada
- **[src/concurrency.rs](src/concurrency.rs)**: Implementaciones comentadas
- **[src/config.rs](src/config.rs)**: Ejemplos de static memory
- **Ejemplo**: `cargo run --example memory_and_concurrency`

## 🧪 Testing

Ejecuta los tests:

```bash
# Todos los tests
cargo test

# Tests con output
cargo test -- --nocapture

# Tests específicos
cargo test domain::
```

## 🔧 Configuración

La biblioteca soporta configuración mediante archivos JSON o variables de entorno:

```rust
// Desde archivo
let config = AppConfig::from_file("config.json")?;

// Desde variables de entorno
let config = AppConfig::from_env()?;

// Validar configuración
config.validate()?;
```

### Variables de Entorno

- `EMBEDDINGS_STORAGE_TYPE`: Tipo de almacenamiento (memory, file, database)
- `EMBEDDINGS_DEFAULT_MODEL`: Modelo por defecto
- `EMBEDDINGS_DIMENSIONS`: Dimensiones de embeddings
- `EMBEDDINGS_API_KEY`: API key si es necesario
- `EMBEDDINGS_LOG_LEVEL`: Nivel de logging (trace, debug, info, warn, error)

## 🎯 Principios SOLID

### Single Responsibility Principle (SRP)
Cada módulo tiene una única responsabilidad:
- `EmbeddingRepository`: Solo persistencia
- `EmbeddingGenerator`: Solo generación
- `VectorStore`: Solo búsqueda de similitud

### Open/Closed Principle (OCP)
Abierto para extensión, cerrado para modificación:
- Nuevas implementaciones de `EmbeddingGenerator` sin modificar código existente
- Custom repositories implementando el trait

### Liskov Substitution Principle (LSP)
Las implementaciones son intercambiables:
- Cualquier `EmbeddingRepository` funciona con `EmbeddingService`
- Mock y producción son intercambiables

### Interface Segregation Principle (ISP)
Interfaces específicas:
- `EmbeddingRepository` vs `EmbeddingRepositoryExt`
- Traits enfocados en responsabilidades específicas

### Dependency Inversion Principle (DIP)
Dependencias en abstracciones, no en implementaciones:
- `EmbeddingService` depende de traits, no de tipos concretos
- Inyección de dependencias mediante `Arc<dyn Trait>`

## 🛠️ Extensibilidad

### Crear un Generador Personalizado

```rust
use async_trait::async_trait;

struct MyCustomGenerator {
    model: String,
    dimensions: usize,
}

#[async_trait]
impl EmbeddingGenerator for MyCustomGenerator {
    async fn generate_from_text(&self, text: &str) -> Result<Embedding> {
        // Tu implementación aquí
        todo!()
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
```

### Crear un Repository Personalizado

```rust
struct PostgresRepository {
    pool: PgPool,
}

#[async_trait]
impl EmbeddingRepository for PostgresRepository {
    async fn save(&self, embedding: &Embedding) -> Result<()> {
        // Implementación con PostgreSQL
        todo!()
    }
    // ... otros métodos
}
```

## 📊 Métricas y Observabilidad

La biblioteca utiliza `tracing` para logging estructurado:

```rust
use tracing_subscriber;

tracing_subscriber::fmt::init();
```

Los eventos de dominio permiten auditoría y debugging:

```rust
let publisher = Arc::new(InMemoryEventPublisher::new());
let service = EmbeddingService::with_events(
    repository,
    generator,
    vector_store,
    publisher,
);
```

## 🔒 Manejo de Errores

Errores tipados y descriptivos:

```rust
pub enum EmbeddingError {
    NotFound(String),
    InvalidDimensions { expected: usize, actual: usize },
    InvalidVector(String),
    StorageError(String),
    GenerationError(String),
    ValidationError(String),
    // ...
}
```

## 🚀 Roadmap

- [ ] Integración con OpenAI embeddings
- [ ] Integración con Hugging Face
- [ ] Soporte para FAISS
- [ ] Soporte para PostgreSQL con pgvector
- [ ] Cache distribuido
- [ ] Métricas con Prometheus
- [ ] Persistencia en disco

## 📖 Documentación Completa

Genera la documentación completa:

```bash
cargo doc --open
```

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature
3. Sigue los principios de clean code
4. Añade tests
5. Actualiza la documentación
6. Envía un Pull Request

## 📝 Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

## 👥 Autores

- Tu Nombre - [GitHub](https://github.com/tu-usuario)

## 🙏 Agradecimientos

- Inspirado por principios de Clean Architecture (Robert C. Martin)
- Patrones de Domain-Driven Design (Eric Evans)
- La comunidad de Rust

---

**Nota**: Esta biblioteca es un ejemplo educativo de clean code y patrones de diseño en Rust. Para uso en producción, considera integrar con modelos y bases de datos vectoriales reales.
