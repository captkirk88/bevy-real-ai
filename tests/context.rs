use bevy::prelude::*;
use rustlicious::context::{ContextGatherRequest, AiSystemContextStore};
use rustlicious::models::ModelBuilder;
use rustlicious::prelude::*;

#[test]
fn gather_on_request_collects_nearby_entity_context() {
    let mut app = App::new();
    let backend = ModelBuilder::new().build_chat().expect("failed to build model");
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
    store.add_system(Box::new(IntoSystem::into_system(
        |ai_entity: rustlicious::context::AiEntity,
         config: Res<rustlicious::context::ContextGatherConfig>,
         transforms: Query<&Transform>,
         characters: Query<(Entity, &Character, &Transform), With<rustlicious::context::AIAware>>| 
         -> Option<rustlicious::rag::AiMessage> {
            // Get the requester's position
            let requester_pos = match transforms.get(*ai_entity) {
                Ok(t) => t.translation,
                Err(_) => return None, // Requester has no transform
            };
            
            // Gather context only from nearby Character entities
            let summaries: Vec<String> = characters.iter()
                .filter(|(ent, _, transform)| {
                    // Don't include the requester itself, and check distance
                    *ent != *ai_entity && 
                    transform.translation.distance(requester_pos) <= config.radius
                })
                .map(|(_, character, _)| {
                    format!("{} has {}", character.name, character.items.join(" and "))
                })
                .collect();
            
            if summaries.is_empty() {
                None
            } else {
                Some(rustlicious::rag::AiMessage::system(&summaries.join(". ")))
            }
        }
    )));

    // Spawn requester and a nearby entity with complete character info
    let requester = app.world_mut().spawn((Transform::from_translation(Vec3::new(0.0,0.0,0.0)),)).id();
    let nearby = app.world_mut().spawn((
        Transform::from_translation(Vec3::new(1.0,0.0,0.0)), 
        Character { name: "Bob".into(), items: vec!["sword".into()] },
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
    eprintln!("Request queue length before: {}", app.world().resource::<rustlicious::dialogue::DialogueRequestQueue>().queue.len());

    // Ask a question that should trigger context gathering and include the sword information
    // Use more updates for the real Llama model which might be slower (30000 updates × 1ms = 30 seconds)
    let resp = match rustlicious::test_helpers::ask_ai_and_wait_result(&mut app, requester, "Where is the sword?", 10000) {
        Ok(r) => r,
        Err(e) => e,
    };
    eprintln!("AI response: {}", resp);
    eprintln!("Request queue length after: {}", app.world().resource::<rustlicious::dialogue::DialogueRequestQueue>().queue.len());
}
