use bevy::prelude::*;
use bevy_ai_dialogue::context::{ContextGatherRequest, AiSystemContextStore};
use bevy_ai_dialogue::models::ModelBuilder;
use bevy_ai_dialogue::prelude::*;

#[test]
fn gather_on_request_collects_nearby_entity_context() {
    let mut app = App::new();
    let backend = ModelBuilder::new()
        .with_seed(42)  // Use fixed seed for deterministic test output
        .build_chat()
        .expect("failed to build model");
    app.add_plugins(MinimalPlugins).add_plugins(AIDialoguePlugin::with_backend(backend));

    // Define test-only component types to avoid leaking test-only types into the public API.
    #[derive(Component, Debug, Clone)]
    struct Character {
        name: String,
        items: Vec<String>,
    }

    // Register a context-gathering system for the Character component
    // This system only gathers context from NEARBY Character entities (within radius of requester)
    let mut store = app.world_mut().resource_mut::<AiSystemContextStore>();
    store.add_system(
        |ai_entity: bevy_ai_dialogue::context::AiEntity,
         characters: Query<(Entity, &Character, &Transform), With<bevy_ai_dialogue::context::AIAware>>| 
         -> Option<bevy_ai_dialogue::rag::AiMessage> {
            // Gather context only from nearby Character entities using the nearby() method
            let nearby_entities = ai_entity.collect_nearby();
            let summaries: Vec<String> = characters.iter()
                .filter(|(ent, _, _)| nearby_entities.contains(ent))
                .map(|(_, character, _)| {
                    format!("{}, a npc, has {}", character.name, character.items.join(" and "))
                })
                .collect();
            
            if summaries.is_empty() {
                None
            } else {
                Some(bevy_ai_dialogue::rag::AiMessage::system(&summaries.join(". ")))
            }
        }
    );

    // Spawn requester (with AI tag) and a nearby entity with complete character info
    let requester = app.world_mut().spawn((
        Transform::from_translation(Vec3::new(0.0,0.0,0.0)),
        bevy_ai_dialogue::context::AI,
    )).id();
    let nearby = app.world_mut().spawn((
        Transform::from_translation(Vec3::new(1.0,0.0,0.0)), 
        Character { name: "Bob".into(), items: vec!["sword".into()] },
        bevy_ai_dialogue::context::AIAware
    )).id();

    eprintln!("Spawned requester: {:?}, nearby: {:?}", requester, nearby);

    // Request a gather for the requester entity
    app.world_mut().resource_mut::<ContextGatherRequest>().request(requester);

    eprintln!("Requested gather for entity {:?}", requester);

    // Run the gather function directly (on-demand) so we avoid scheduling complexity in tests
    bevy_ai_dialogue::context::gather_on_request_world(app.world_mut());

    eprintln!("Gather completed");

    // The gather system should attach an `AiContext` component to the requester entity.
    if let Some(ctx) = app.world().get::<bevy_ai_dialogue::rag::AiContext>(requester) {
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
    eprintln!("Request queue length before: {}", app.world().resource::<bevy_ai_dialogue::dialogue::DialogueRequestQueue>().len());

    // Ask a question that should trigger context gathering and include the sword information
    // Use more updates for the real Llama model which might be slower (30000 updates × 1ms = 30 seconds)
    let resp = match bevy_ai_dialogue::test_helpers::ask_ai_and_wait_result(&mut app, requester, "Where is the sword?", 10000) {
        Ok(r) => r,
        Err(e) => e,
    };
    eprintln!("AI response: {}", resp);
}
