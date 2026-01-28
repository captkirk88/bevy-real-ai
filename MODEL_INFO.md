# Model Information

## Custom Models

You can specify a custom model using:

### Local File
```rust
ModelBuilder::new()
    .with_local(PathBuf::from("/path/to/model.gguf"))
    .build()
```

### HuggingFace
```rust
ModelBuilder::new()
    .with_huggingface(
        "model-repo-id".to_string(),
        "main".to_string(),
        "model.gguf".to_string()
    )
    .build()
```