Lightweight local-AI utilities and a Bevy dialogue plugin for integrating local LLMs into Bevy.

- A `AIDialoguePlugin` for asynchronous AI prompts tied to Bevy entities and contexts.
- `ModelBuilder` helpers for loading models asynchronously.
- A **typed AI action** protocol allowing the AI to emit structured actions (typed JSON that maps to registered handlers).  You could spawn a entity by nicely asking the AI (see examples).
- Utility helpers for parsing and repairing LLM JSON output and a small RAG context helper.

---

## Quick start 🚀

1. Add the crate to your project (example using cargo):

```bash
# from your workspace root
cargo run --example actions --release
```

2. In your Bevy app, add the dialogue plugin with a model builder:

```rust
use bevy_real_ai::models::{ModelBuilder, ModelType};
use bevy_real_ai::prelude::*;

App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(AIDialoguePlugin::with_builder(
        ModelBuilder::new_with(ModelType::Llama).with_seed(42).with_progress_tracking(),
    ))
    .run();
```

## Examples

- `examples/actions.rs` — demonstrates typed actions via `AiAction` and `prompt_typed_action`, and registering action handlers with `AiActionRegistry`.
- [typed_actions.rs](examples/typed_actions.rs) — shows typed parsing flows.
- [basic.rs](examples/basic.rs) and [npc.rs](examples/npc.rs) — minimal dialogue and NPC examples.

Run any example with (always --release):

```bash
cargo run --example actions --release
```

## Key APIs 🔧

- `AiRequest` (Bevy SystemParam)
  - Convenience wrapper around the `DialogueRequestQueue`.
  - Use `AiRequest::ask_text(...)` to queue text prompts.
  - Use `AiRequest::ask_action::<T>(...)` to queue typed action prompts.

- `prompt_typed_action::<T>(backend, prompt, entity, &mut pending)`
  - Synchronously prompts the model with the typed schema derived from `T: AiParsable` (from `#[derive(AiAction)]`), parses the response into `T`, and queues it as an action.

- Typed action handlers
  - Use the `#[derive(AiAction)]` macro on a `struct` to generate parsing and action conversion utilities. Register handlers using the auto-generated `register` method:

```rust
SpawnEntityAction::register(&mut registry, |In(action): In<SpawnEntityAction>, mut cmds: Commands| {
    // spawn or apply changes based on `action`
});
```

- `On<ModelLoadCompleteEvent>` system condition
  - Observe model load completion with `On<ModelLoadCompleteEvent>` to queue startup prompts or handle errors.

## Robust parsing

The library includes helpers that attempt to repair common malformed JSON outputs from LLMs (e.g., missing closing brackets) and tries parsing code-fenced JSON commonly emitted by models.

## Tests & Development

- Tests are in `/tests`. Run unit tests with:

```bash
cargo test --release -- --nocapture
```

- Examples are in `/examples`. Run them with `cargo run --example <name> --release`.

---

Contributions welcome — open an issue or pull request!

# LICENSE
The MIT License (MIT)
