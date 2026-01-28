# Model Download Progress Integration

This document describes how to use the model download progress tracking feature integrated with `ModelBuilder` and `AIDialoguePlugin`.

## Overview

The system uses `crossbeam-channel` to send `ModelDownloadProgress` events from the model loading process to Bevy's observer system. Progress is tracked with three states: `InProgress`, `Completed`, and `Error`.

## Usage Patterns

### Pattern 1: Pre-built Backend (Synchronous)

Build the model before adding the plugin:

```rust
use rustlicious::prelude::*;
use std::sync::Arc;

let backend = ModelBuilder::new()
    .with_huggingface("model-id".to_string(), "main".to_string(), "model.gguf".to_string())
    .build()?;

app.add_plugins(AIDialoguePlugin::new(Arc::new(backend)));
```

### Pattern 2: Async Loading with Progress Tracking (Recommended)

Use `with_builder()` to load the model asynchronously while tracking progress:

```rust
use rustlicious::prelude::*;

let builder = ModelBuilder::new()
    .with_huggingface("model-id".to_string(), "main".to_string(), "model.gguf".to_string())
    .with_progress();

app.add_plugins(AIDialoguePlugin::with_builder(builder));
```

This approach:
- Starts the app immediately with `MockAi` backend
- Loads the model in the background with progress tracking
- Automatically spawns a `ModelDownloadTracker` component
- Emits `ModelDownloadProgressEvent` events via Bevy's observer system

## Event and Observer Pattern

The system uses Bevy 0.18's observer pattern for event handling:

### ModelDownloadProgressEvent

```rust
#[derive(Event, Clone, Debug)]
pub struct ModelDownloadProgressEvent {
    pub model_name: String,
    pub state: DownloadState,
    pub message: String,
    pub progress: Option<f32>, // 0.0 to 1.0 for InProgress, None otherwise
}
```

### Custom Observer

You can add your own observer to handle progress events:

```rust
app.add_observer(|trigger: On<ModelDownloadProgressEvent>| {
    let event = trigger.event();
    match event.state {
        DownloadState::InProgress => {
            println!("[{}] Progress: {:.1}%", event.model_name, event.progress.unwrap_or(0.0));
        }
        DownloadState::Completed => {
            println!("[{}] ✓ Download completed!", event.model_name);
        }
        DownloadState::Error => {
            eprintln!("[{}] ✗ Error: {}", event.model_name, event.message);
        }
    }
});
```
