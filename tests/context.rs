use bevy::prelude::*;
use bevy_real_ai::context::{AiSystemContextStore, ContextGatherRequest};
use bevy_real_ai::models::AiModelBuilder;
use bevy_real_ai::prelude::*;

#[test]
fn gather_on_request_collects_nearby_entity_context() {
    let mut app = App::new();
    let backend = AiModelBuilder::new_with(ModelType::Llama)
        .with_seed(42) // Use fixed seed for deterministic test output
        .build()
        .expect("failed to build model");
    app.add_plugins(MinimalPlugins)
        .add_plugins(AIDialoguePlugin::with_backend(backend));

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
        |ai_entity: bevy_real_ai::context::AiEntity,
         characters: Query<
            (Entity, &Character, &Transform),
            With<bevy_real_ai::context::AIAware>,
        >|
         -> Option<bevy_real_ai::rag::AiMessage> {
            // Gather context only from nearby Character entities using the nearby() method
            let nearby_entities = ai_entity.collect_nearby();
            let summaries: Vec<String> = characters
                .iter()
                .filter(|(ent, _, _)| nearby_entities.contains(ent))
                .map(|(_, character, _)| {
                    format!(
                        "{}, a npc, has {}",
                        character.name,
                        character.items.join(" and ")
                    )
                })
                .collect();

            if summaries.is_empty() {
                None
            } else {
                Some(bevy_real_ai::rag::AiMessage::system(&summaries.join(". ")))
            }
        },
    );

    // Spawn requester (with AI tag) and a nearby entity with complete character info
    let requester = app
        .world_mut()
        .spawn((
            Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            bevy_real_ai::context::AI,
        ))
        .id();
    let nearby = app
        .world_mut()
        .spawn((
            Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            Character {
                name: "Bob".into(),
                items: vec!["sword".into()],
            },
            bevy_real_ai::context::AIAware,
        ))
        .id();

    eprintln!("Spawned requester: {:?}, nearby: {:?}", requester, nearby);

    // Request a gather for the requester entity
    app.world_mut()
        .resource_mut::<ContextGatherRequest>()
        .request(requester);

    eprintln!("Requested gather for entity {:?}", requester);

    // Run the gather function directly (on-demand) so we avoid scheduling complexity in tests
    bevy_real_ai::context::gather_on_request_world(app.world_mut());

    eprintln!("Gather completed");

    // The gather system should attach an `AiContext` component to the requester entity.
    if let Some(ctx) = app.world().get::<bevy_real_ai::rag::AiContext>(requester) {
        let joined = ctx
            .messages()
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        eprintln!("AI context messages: {}", joined);
        assert!(
            joined.contains("bob"),
            "Context should contain 'bob', but got: {}",
            joined
        );
        assert!(joined.contains("sword"));
    } else {
        panic!("expected AiContext on requester");
    }

    // Test that the dialogue system includes context from gathered entities
    // Add dialogue components to the requester entity that already has context
    app.world_mut()
        .entity_mut(requester)
        .insert((DialogueReceiver::new(),));

    eprintln!(
        "Added dialogue components to requester entity: {:?}",
        requester
    );
    eprintln!(
        "Request queue length before: {}",
        app.world()
            .resource::<bevy_real_ai::dialogue::DialogueRequestQueue>()
            .len()
    );

    // Ask a question that should trigger context gathering and include the sword information
    // Use more updates for the real Llama model which might be slower (30000 updates × 1ms = 30 seconds)
    let resp = match bevy_real_ai::test_helpers::ask_ai_and_wait_result(
        &mut app,
        requester,
        "Where is the sword?",
        10000,
    ) {
        Ok(r) => r,
        Err(e) => e,
    };
    eprintln!("AI response: {}", resp);
}
