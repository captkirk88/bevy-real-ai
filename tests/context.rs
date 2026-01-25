use bevy::prelude::*;
use rustlicious::context::{ContextGatherRequest};
use rustlicious::npc_dialogue::{AIDialoguePlugin, DialogueReceiver, Speaker};
use rustlicious::models::Type as ModelType;
use std::sync::Arc;


#[test]
fn gather_on_request_collects_nearby_entity_context() {
    let mut app = App::new();
    let backend = match ModelType::Llama.new().build() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping test: failed to build Llama model: {}", e);
            return;
        }
    };
    app.add_plugins(MinimalPlugins).add_plugins(AIDialoguePlugin::new(Arc::from(backend)));

    // Define test-only component types to avoid leaking test-only types into the public API.
    #[derive(Component, Debug, Clone)]
    struct Character {
        name: String,
        items: Vec<String>,
    }

    // Implement the public trait for a complete character description
    impl rustlicious::AiDescribe for Character {
        fn summarize(&self, _entity: bevy::prelude::Entity, _world: &bevy::prelude::World) -> Option<String> {
            Some(format!("{} has a {} in his bag", self.name, self.items.join(" and ")))
        }
    }

    let mut registry = rustlicious::context::ContextExtractorRegistry::new();
    registry.register_summarizable::<Character>();
    app.insert_resource(registry);

    // Spawn requester and a nearby entity with complete character info
    let requester = app.world_mut().spawn((Transform::from_translation(Vec3::new(0.0,0.0,0.0)),)).id();
    let nearby = app.world_mut().spawn((
        Transform::from_translation(Vec3::new(1.0,0.0,0.0)), 
        Character { name: "Bob".into(), items: vec!["sword".into()] },
        DialogueReceiver::new(), 
        rustlicious::context::AIAware
    )).id();

    eprintln!("Spawned requester: {:?}, nearby: {:?}", requester, nearby);

    // Request a gather for the requester entity
    *app.world_mut().resource_mut::<ContextGatherRequest>() = ContextGatherRequest(Some(requester));

    eprintln!("Requested gather for entity {:?}", requester);

    // Run the gather function directly (on-demand) so we avoid scheduling complexity in tests
    rustlicious::context::gather_on_request_world(app.world_mut());

    eprintln!("Gather completed");

    // The gather system should attach an `AiContext` component to the requester entity.
    if let Some(ctx) = app.world().get::<rustlicious::rag::AiContext>(requester) {
        let joined = ctx.messages().iter().map(|m| format!("{:?}", m)).collect::<Vec<_>>().join(" ").to_lowercase();
        eprintln!("AI context messages: {}", joined);
        assert!(joined.contains("bob"), "Context should contain 'bob', but got: {}", joined);
        assert!(joined.contains("sword"));
    } else {
        panic!("expected AiContext on requester");
    }

    // Test that the dialogue system includes context from gathered entities
    // Add dialogue components to the requester entity that already has context
    app.world_mut().entity_mut(requester).insert((
        Speaker::new("Bard", ""),
        DialogueReceiver::new(),
    ));

    eprintln!("Added dialogue components to requester entity: {:?}", requester);
    eprintln!("Request queue length before: {}", app.world().resource::<rustlicious::npc_dialogue::DialogueRequestQueue>().queue.len());

    // Ask a question that should trigger context gathering and include the sword information
    // Use more updates for the real Llama model which might be slower
    let resp = rustlicious::test_helpers::ask_ai_and_wait(&mut app, requester, "Where is the sword?", 1000).expect("expected response");
    eprintln!("AI response: {}", resp);
}
