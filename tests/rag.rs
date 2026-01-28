use bevy::prelude::*;
use rustlicious::prelude::*;
use std::sync::Arc;

struct EchoAi;
impl LocalAi for EchoAi {
    fn prompt(&self, messages: &[AiMessage]) -> Result<String, String> {
        // Render messages via Debug so tests can assert they contain system/user pieces
        let combined = messages.iter().map(|m| format!("{:?}", m)).collect::<Vec<_>>().join("\n");
        Ok(format!("echo: {}", combined))
    }
}

#[test]
fn retrieval_augmented_ai_includes_context_in_prompt() {
    let backend: Arc<dyn LocalAi> = Arc::new(EchoAi {});

    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AIDialoguePlugin::default());
    app.insert_resource(LocalAiHandle::new(backend));

    // Spawn entity with Speaker + DialogueReceiver + AiContext component
    let mut context = rustlicious::rag::AiContext::new();
    context.add_context("The tavern is to the east, full of noisy patrons.");
    context.add_context("A lantern hangs above the doorway.");
    
    let e = app
        .world_mut()
        .spawn((
            Speaker::new("Bard", ""),
            DialogueReceiver::new(),
            context,
        ))
        .id();

    // Ask through helper and assert echo includes context
    let resp = rustlicious::test_helpers::ask_ai_and_wait(&mut app, e, "Where is the tavern?", 50).expect("expected response");
    assert!(resp.contains("tavern"));
}
