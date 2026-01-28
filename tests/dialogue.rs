use bevy::prelude::*;
use rustlicious::prelude::*;
use std::sync::Arc;


#[test]
fn preprogrammed_response_is_immediate() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AIDialoguePlugin::default());

    // spawn entity with Speaker + DialogueReceiver (preprogrammed response)
    let e = app
        .world_mut()
        .spawn((Speaker::new("Bob", ""), DialogueReceiver { preprogrammed: Some("Hello, traveler".into()), last_response: None },))
        .id();

    // Ask AI and wait for response using test helper
    let resp = rustlicious::test_helpers::ask_ai_and_wait(&mut app, e, "ignored", 10).expect("expected response");
    assert_eq!(resp, "Hello, traveler");
}

#[test]
fn mock_ai_generates_response() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AIDialoguePlugin::default());

    // spawn entity with Speaker + DialogueReceiver (no preprogrammed)
    let e = app
        .world_mut()
        .spawn((Speaker::new("Alice", ""), DialogueReceiver::new(),))
        .id();

    // Ask AI and assert the response contains the expected text
    let resp = rustlicious::test_helpers::ask_ai_and_wait(&mut app, e, "Say hi", 50).expect("expected response");
assert!(resp.contains("mock: Say hi"));
}

#[test]
fn custom_backend_can_be_used() {
    struct TestAi;
    impl LocalAi for TestAi {
        fn prompt(&self, messages: &[rustlicious::rag::AiMessage]) -> Result<String, String> {
            // Return the first user-like message content when present
            for m in messages.iter() {
                match m {
                    rustlicious::rag::AiMessage::User(content) => {
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

    let e = app.world_mut().spawn((Speaker::new("Eve", ""), DialogueReceiver::new(),)).id();

    // Ask and wait via helper
    let resp = rustlicious::ask_ai_and_wait(&mut app, e, "Ping", 50).expect("expected response");
    assert_eq!(resp, "custom: Ping");
}
