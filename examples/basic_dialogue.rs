//! Minimal example showing basic AI dialogue.
//!
//! Run with: `cargo run --example basic_dialogue --release`
//!
//! This is the simplest possible example of using rustlicious for AI dialogue.

use bevy::prelude::*;
use rustlicious::models::ModelBuilder;
use rustlicious::prelude::*;

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .add_plugins(setup_ai_plugin())
        .add_systems(Startup, setup)
        .add_systems(Update, check_response)
        .run();
}

fn setup_ai_plugin() -> AIDialoguePlugin {
    let backend = ModelBuilder::new()
        .with_seed(42)
        .build_chat()
        .expect("Failed to build AI model");

    AIDialoguePlugin::with_backend(backend)
}

fn setup(mut commands: Commands, mut request_queue: ResMut<rustlicious::dialogue::DialogueRequestQueue>) {
    println!("Setting up dialogue entity...");

    // Spawn an entity that can speak and receive dialogue
    let entity = commands.spawn((
        Speaker::new("Player", "A curious person"),
        DialogueReceiver::new(),
        rustlicious::context::AI,
    )).id();

    // Queue a dialogue request
    request_queue.queue.push_back(rustlicious::dialogue::DialogueRequest {
        entity,
        prompt: "Hello! What is 2 + 2?".to_string(),
    });

    println!("Dialogue request queued. Waiting for AI response...");
}

fn check_response(
    mut commands: Commands,
    query: Query<&DialogueReceiver, Changed<DialogueReceiver>>,
) {
    for receiver in query.iter() {
        if let Some(response) = &receiver.last_response {
            println!("\n=== AI Response ===");
            println!("{}", response);
            println!("===================\n");
            
            // Exit after receiving response
            commands.write_message(AppExit::Success);
        }
    }
}
