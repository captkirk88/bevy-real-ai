//! Minimal example showing basic AI dialogue.
//!
//! Run with: `cargo run --example basic --release`
//!
//! This is the simplest possible example of using bevy_ai_dialogue for AI dialogue.
//! Uses async model loading - requests are queued until the model is ready.
//! See `npc_dialogue.rs` for a more complete example with progress bar UI.

use bevy::ecs::observer::On;
use bevy::prelude::*;
use bevy_ai_dialogue::models::ModelBuilder;
use bevy_ai_dialogue::prelude::*;

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .add_plugins(setup_ai_plugin())
        .add_systems(Startup, setup)
        .add_systems(Update, check_response)
        .add_observer(on_model_load_complete)
        .run();
}

fn setup_ai_plugin() -> AIDialoguePlugin {
    // Use async builder with progress tracking (model loads in background)
    let builder = ModelBuilder::new()
        .with_seed(42)
        .with_progress();

    AIDialoguePlugin::with_builder(builder)
}

/// Observer that logs when the model finishes loading
fn on_model_load_complete(trigger: On<ModelLoadCompleteEvent>) {
    let event = trigger.event();
    if event.success {
        println!("✓ Model '{}' loaded successfully!", event.model_name);
    } else if let Some(ref err) = event.error_message {
        println!("✗ Model '{}' failed to load: {}", event.model_name, err);
    }
}

fn setup(mut commands: Commands, mut request_queue: ResMut<DialogueRequestQueue>) {
    println!("Setting up dialogue entity...");
    println!("Waiting for model to load in background...");

    // Spawn an entity that can speak and receive dialogue
    let entity = commands.spawn((
        Speaker::new("Player", "A curious person"),
        DialogueReceiver::new(),
        bevy_ai_dialogue::context::AI,
    )).id();

    // Queue the request immediately - it will be processed once the model loads
    println!("Queuing dialogue request (will be processed when model is ready)...");
    request_queue.push(DialogueRequest {
        entity,
        prompt: "Hello! What is 2 + 2?".to_string(),
    });
}

fn check_response(
    mut commands: Commands,
    query: Query<&DialogueReceiver, Changed<DialogueReceiver>>,
) {
    // Check for AI response
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
