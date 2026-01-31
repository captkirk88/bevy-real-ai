use bevy::prelude::*;
use bevy_real_ai::prelude::*;
use std::sync::Arc;


#[test]
fn preprogrammed_response_is_immediate() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AIDialoguePlugin::default());

    // spawn entity with Speaker + DialogueReceiver (preprogrammed response)
    let e = app
        .world_mut()
        .spawn((AI, DialogueReceiver::new_with_preprogrammed("Hello, traveler"),))
        .id();

    // Ask AI and wait for response using test helper
    let resp = bevy_real_ai::test_helpers::ask_ai_and_wait(&mut app, e, "ignored", 10).expect("expected response");
    assert_eq!(resp, "Hello, traveler");
}

#[test]
fn mock_ai_generates_response() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AIDialoguePlugin::default());

    // spawn entity with Speaker + DialogueReceiver (no preprogrammed)
    let e = app
        .world_mut()
        .spawn((AI, DialogueReceiver::new(),))
        .id();

    // Ask AI and assert the response contains the expected text
    let resp = bevy_real_ai::test_helpers::ask_ai_and_wait(&mut app, e, "Say hi", 50).expect("expected response");
assert!(resp.contains("mock: Say hi"));
}

#[test]
fn custom_backend_can_be_used() {
    struct TestAi;
    impl LocalAi for TestAi {
        fn prompt(&self, messages: &[bevy_real_ai::rag::AiMessage]) -> Result<String, String> {
            // Return the first user-like message content when present
            for m in messages.iter() {
                match m {
                    bevy_real_ai::rag::AiMessage::User(content) => {
                        return Ok(format!("custom: {}", content));
                    }
                    _ => continue,
                }
            }
            Ok("custom: no user message found".to_string())
        }

    }

    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AIDialoguePlugin::default());

    // replace backend with test backend
    app.insert_resource(LocalAiHandle::new(Arc::new(TestAi)));

    let e = app.world_mut().spawn((AI, DialogueReceiver::new(),)).id();

    // Ask and wait via helper
    let resp = bevy_real_ai::ask_ai_and_wait(&mut app, e, "Ping", 50).expect("expected response");
    assert_eq!(resp, "custom: Ping");
}

#[test]
fn ai_action_block_is_parsed_and_stored() {
    struct ActionAi;
    impl LocalAi for ActionAi {
        fn prompt(&self, _messages: &[bevy_real_ai::rag::AiMessage]) -> Result<String, String> {
            // Return a raw JSON action object (no fenced blocks)
            let body = r#"{"name": "spawn_entity", "params": {"prefab": "goblin", "x": 2.0}}"#;
            Ok(body.to_string())
        }
    }

    #[derive(Component)]
    struct TestSpawned;

    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AIDialoguePlugin::default());

    // Register a registry handler that spawns TestSpawned so we can assert it ran
    {
        let mut registry = app.world_mut().resource_mut::<bevy_real_ai::actions::AiActionRegistry>();
        registry.register("spawn_entity", move |mut cmds: Commands, _ctx: bevy_real_ai::actions::AiAction| {
            let _ = cmds.spawn((TestSpawned, Name::new("testspawn"), Transform::default()));
            // Optionally re-emit the original event for other systems if desired
            // cmds.trigger(ctx.event().clone());
        });
    }

    // Use the action-generating backend
    app.insert_resource(LocalAiHandle::new(Arc::new(ActionAi)));

    let e = app.world_mut().spawn((AI,DialogueReceiver::new(),)).id();

    // Ask and wait via helper
    let _ = bevy_real_ai::ask_ai_and_wait(&mut app, e, "Do action", 50).expect("expected response");

    // Check receiver has parsed actions and stored last_response (JSON string)
    let receiver = app.world().get::<DialogueReceiver>(e).expect("receiver exists");
    assert_eq!(receiver.last_response.as_deref(), Some(r#"{"name": "spawn_entity", "params": {"prefab": "goblin", "x": 2.0}}"#));
    assert_eq!(receiver.actions.len(), 1);
    let action = &receiver.actions[0];
    assert_eq!(action.name, "spawn_entity");
    assert_eq!(action.params["prefab"], serde_json::Value::String("goblin".to_string()));

    // Ensure the registry handler actually spawned an entity with TestSpawned
    let world = app.world_mut();
    let spawned_count = world.query::<&TestSpawned>().iter(&world).count();
    assert_eq!(spawned_count, 1, "expected a handler to spawn TestSpawned");
}

