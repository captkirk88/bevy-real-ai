# rustlicious 🔧

Lightweight local-AI utilities and a Bevy dialogue plugin. Includes:

- A simple `DialoguePlugin` for Bevy entities and background AI processing.
- A `CandleBackend` wrapper around `candle-pipelines` for local LLM usage (toy models supported).
- A tiny RAG (retrieval-augmented generation) helper and in-memory context store.

---

## Quick start 🚀

`cargo add `

## Examples

Review tests in (/tests)[/tests] for more detailed examples.

Create a backend from config (may download/build pipeline):

```rust
let cfg = CandleBackendConfig { auto_build_pipeline: true, ..Default::default() };
let backend = CandleBackend::try_new_from_config(cfg)?;
let reply = backend.prompt("Tell me a joke about programming.")?;
```

Use the Dialogue plugin in Bevy and provide a backend directly (or rely on the default mock backend):

```rust
// Provide your backend when creating the plugin:
app.add_plugins(MinimalPlugins).add_plugins(DialoguePlugin::new(Arc::new(my_backend)));

// Or use the plugin default (mock backend) and replace the `LocalAiHandle` later:
app.add_plugins(MinimalPlugins).add_plugins(DialoguePlugin::default());
app.insert_resource(LocalAiHandle::new(Arc::new(my_backend)));
```

---

Contributions welcome — open issues or PRs.

# LICENSE
The MIT License (MIT)
