//! Example demonstrating typed parsing with our custom AiParse derive macro.
//!
//! This example shows how to use `#[derive(AiParse)]` for a typed `SpawnAction`.
//! The derive macro automatically implements both `AiParsable` and `IntoActionPayload`,
//! converting AI responses to queued actions without any manual trait implementations.
//!
//! Run with: `cargo run --example typed_actions --release`

use bevy::prelude::*;
use bevy_real_ai::prelude::*;
use bevy_real_ai::AiParse;
use serde::{Deserialize, Serialize};

/// A typed action struct that the AI will parse from natural language.
/// For example: "Create an entity at position 0,0" -> SpawnAction { name: "enemy", x: 0, y: 0 }
/// 
/// The `AiParse` derive automatically implements:
/// - `AiParsable` for JSON schema generation and parsing
/// - `IntoActionPayload` with action name "spawn_action" (from struct name in snake_case)
#[derive(Clone, Debug, Serialize, Deserialize, AiParse)]
struct SpawnEntity {
    pub name: String,
    pub x: f32,
    pub y: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(AIDialoguePlugin::with_builder(AiModelBuilder::new_with(ModelType::Llama).with_progress()))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut registry: ResMut<AiActionRegistry>, mut request: AiRequest) {
    commands.spawn(Camera2d::default());

    let player = commands
        .spawn((
            Speaker::new("Player", "Typed action requester"),
            DialogueReceiver::new(),
            AI,
        ))
        .id();

    // Register a handler using the auto-generated register method
    SpawnEntity::register(&mut registry, handle_spawn_entity);

    request.ask_action::<SpawnEntity>(
        player,
        "Create a entity named 'goblin' at position x=5.0 and y=10.0",
    );
}

/// Handler system for spawn_action actions
fn handle_spawn_entity(In(action): In<SpawnEntity>, mut commands: Commands) {
    let name = action.name;
    let x = action.x;
    let y = action.y;

    commands.spawn((
        Name::new(name.clone()),
        Transform::from_translation(Vec3::new(x, y, 0.0)),
    ));
}