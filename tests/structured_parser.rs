//! Tests for the structured AI parsing system using a real AI model.
//!
//! Run with: cargo test --release -- --nocapture structured_parser

use bevy::prelude::Entity;
use bevy_real_ai::AiAction;
use bevy_real_ai::actions::{PendingAiActions, prompt_typed_action};
use bevy_real_ai::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A typed struct representing a spawn action that the AI should produce.
/// The `AiAction` derive automatically implements both `AiParsable` and `IntoActionPayload`.
/// The action name is auto-generated as "spawn_action" (struct name in snake_case).
#[derive(Clone, Debug, Serialize, Deserialize, AiAction)]
struct SpawnAction {
    pub name: String,
    pub x: i32,
    pub y: i32,
}

#[test]
fn real_ai_parses_typed_spawn_action() {
    println!("Building AI model (this may take a moment on first run)...");

    // Build a real AI model using the default Llama model
    let backend: Arc<dyn LocalAi> = AiModelBuilder::new_with(ModelType::Llama)
        .include_default_context(true)
        .build()
        .expect("Failed to build AI model");

    println!("Model loaded successfully!");

    // Create a mock pending actions queue
    let mut pending = PendingAiActions::default();

    // Create a dummy entity using placeholder (Bevy 0.18+)
    let entity = Entity::PLACEHOLDER;

    // Ask the AI to create an entity at a specific position
    let user_request = "I want to create an entity named 'player' at position x=10 and y=20";

    println!("Sending request to AI: {}", user_request);

    let result = prompt_typed_action::<SpawnAction>(&backend, user_request, entity, &mut pending);

    match result {
        Ok((parsed, response)) => {
            println!(
                "Successfully parsed response: {:?}\nAI: {}",
                parsed, response
            );

            // Verify the parsed action has reasonable values
            assert!(!parsed.name.is_empty(), "Name should not be empty");

            // Check that an action was queued
            assert_eq!(
                pending.actions.len(),
                1,
                "Should have queued exactly one action"
            );

            let queued = &pending.actions[0];
            assert_eq!(queued.action.name, "spawn_action"); // Auto-generated from struct name
            assert_eq!(queued.entity, entity);

            println!("Action queued successfully: {:?}", queued.action);
        }
        Err(e) => {
            panic!("Failed to parse AI response: {}", e);
        }
    }
}

#[test]
fn schema_description_is_generated_correctly() {
    let schema = SpawnAction::schema_description();
    println!("Generated schema:\n{}", schema);

    // Verify the schema contains our field names and types
    assert!(
        schema.contains("name"),
        "Schema should contain 'name' field"
    );
    assert!(schema.contains("x"), "Schema should contain 'x' field");
    assert!(schema.contains("y"), "Schema should contain 'y' field");
    assert!(
        schema.contains("string"),
        "Schema should contain 'string' type"
    );
    assert!(
        schema.contains("integer"),
        "Schema should contain 'integer' type"
    );
}

#[test]
fn parse_extracts_json_from_various_formats() {
    use bevy_real_ai::parse::extract_and_parse_json;

    // Test pure JSON
    let json = r#"{"name": "test", "x": 5, "y": 10}"#;
    let result: SpawnAction = extract_and_parse_json(json).expect("should parse pure JSON");
    assert_eq!(result.name, "test");
    assert_eq!(result.x, 5);
    assert_eq!(result.y, 10);

    // Test JSON in code block
    let markdown = r#"Here's your action:
```json
{"name": "player", "x": 0, "y": 0}
```
"#;
    let result: SpawnAction =
        extract_and_parse_json(markdown).expect("should parse from code block");
    assert_eq!(result.name, "player");

    // Test JSON embedded in text
    let mixed = r#"Sure! {"name": "enemy", "x": -5, "y": 15} That should work."#;
    let result: SpawnAction = extract_and_parse_json(mixed).expect("should parse from mixed text");
    assert_eq!(result.name, "enemy");
    assert_eq!(result.x, -5);
    assert_eq!(result.y, 15);
}
