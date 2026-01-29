//! A simple Bevy example demonstrating AI-powered NPC dialogue.
//!
//! Run with: `cargo run --example npc_dialogue --release`
//!
//! This example creates a simple game world with:
//! - A player entity
//! - NPCs with items and descriptions
//! - An AI-powered dialogue system that responds based on NPC context
//!
//! Press SPACE to talk to nearby NPCs.

use bevy::prelude::*;
use rustlicious::context::{AiSystemContextStore, ContextGatherRequest, AI, AIAware, AiEntity};
use rustlicious::models::ModelBuilder;
use rustlicious::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(setup_ai_plugin())
        .add_systems(Startup, setup_world)
        .add_systems(Update, (handle_input, display_dialogue))
        .run();
}

/// Set up the AI dialogue plugin with a local model
fn setup_ai_plugin() -> AIDialoguePlugin {
    let backend = ModelBuilder::new()
        .with_seed(42) // Optional: use fixed seed for consistent responses
        .build_chat()
        .expect("Failed to build AI model");

    AIDialoguePlugin::with_backend(backend)
}

/// Components for our game entities

#[derive(Component)]
struct Player;

#[derive(Component)]
struct Npc {
    name: String,
    description: String,
}

#[derive(Component)]
struct Inventory {
    items: Vec<String>,
}

#[derive(Component)]
struct DialogueUI;

/// Set up the game world with player and NPCs
fn setup_world(mut commands: Commands, mut store: ResMut<AiSystemContextStore>) {
    // Camera
    commands.spawn(Camera2d::default());

    // Spawn player (has AI marker for dialogue capability)
    commands.spawn((
        Player,
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        AI, // This entity can initiate AI dialogue
        Speaker::new("Player", "A curious adventurer"),
        DialogueReceiver::new(),
    ));

    // Spawn NPCs with AIAware marker (they provide context to AI)
    commands.spawn((
        Npc {
            name: "Bob the Blacksmith".into(),
            description: "A burly man with soot-covered arms".into(),
        },
        Inventory {
            items: vec!["iron sword".into(), "steel shield".into(), "hammer".into()],
        },
        Transform::from_translation(Vec3::new(50.0, 0.0, 0.0)),
        AIAware, // This entity's info will be gathered for AI context
    ));

    commands.spawn((
        Npc {
            name: "Elena the Herbalist".into(),
            description: "An elderly woman surrounded by the scent of herbs".into(),
        },
        Inventory {
            items: vec!["healing potion".into(), "antidote".into(), "rare mushroom".into()],
        },
        Transform::from_translation(Vec3::new(-30.0, 20.0, 0.0)),
        AIAware,
    ));

    commands.spawn((
        Npc {
            name: "Marcus the Guard".into(),
            description: "A stern-looking guard in polished armor".into(),
        },
        Inventory {
            items: vec!["spear".into(), "city keys".into()],
        },
        Transform::from_translation(Vec3::new(0.0, -40.0, 0.0)),
        AIAware,
    ));

    // Register context-gathering system for NPCs
    // This tells the AI about nearby NPCs when dialogue is initiated
    store.add_system(gather_npc_context);

    // Spawn UI text for dialogue display
    commands.spawn((
        DialogueUI,
        Text::new("Press SPACE to talk to nearby NPCs\nPress 1-3 to ask different questions"),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(20.0),
            left: Val::Px(20.0),
            ..default()
        },
    ));

    info!("World setup complete! Press SPACE to initiate dialogue.");
}

/// Context gathering system - provides NPC information to the AI
fn gather_npc_context(
    ai_entity: AiEntity,
    npcs: Query<(Entity, &Npc, &Inventory, &Transform), With<AIAware>>,
) -> Option<rustlicious::rag::AiMessage> {
    let summaries: Vec<String> = npcs
        .iter()
        .filter(|(ent, _, _, transform)| ai_entity.is_nearby(*ent, transform.translation))
        .map(|(_, npc, inventory, _)| {
            format!(
                "{}, a npc, is nearby. {}. They have: {}",
                npc.name,
                npc.description,
                inventory.items.join(", ")
            )
        })
        .collect();

    if summaries.is_empty() {
        None
    } else {
        Some(rustlicious::rag::AiMessage::system(&summaries.join("\n")))
    }
}

/// Handle player input for dialogue
fn handle_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    player_query: Query<Entity, With<Player>>,
    mut context_request: ResMut<ContextGatherRequest>,
    mut request_queue: ResMut<rustlicious::dialogue::DialogueRequestQueue>,
) {
    let Ok(player_entity) = player_query.single() else {
        return;
    };

    // Gather context when space is pressed
    if keyboard.just_pressed(KeyCode::Space) {
        info!("Gathering context from nearby NPCs...");
        *context_request = ContextGatherRequest(Some(player_entity));
    }

    // Different questions the player can ask
    let prompt = if keyboard.just_pressed(KeyCode::Digit1) {
        Some("Who is nearby and what do they have?")
    } else if keyboard.just_pressed(KeyCode::Digit2) {
        Some("Where can I find a healing potion?")
    } else if keyboard.just_pressed(KeyCode::Digit3) {
        Some("What weapons are available?")
    } else {
        None
    };

    if let Some(prompt) = prompt {
        info!("Asking: {}", prompt);
        request_queue.queue.push_back(rustlicious::dialogue::DialogueRequest {
            entity: player_entity,
            prompt: prompt.to_string(),
        });
    }
}

/// Display AI responses in the UI
fn display_dialogue(
    player_query: Query<&DialogueReceiver, (With<Player>, Changed<DialogueReceiver>)>,
    mut ui_query: Query<&mut Text, With<DialogueUI>>,
) {
    for receiver in player_query.iter() {
        if let Some(response) = &receiver.last_response {
            info!("AI Response: {}", response);
            if let Ok(mut text) = ui_query.single_mut() {
                **text = format!(
                    "AI: {}\n\nPress 1: Who is nearby?\nPress 2: Where's a healing potion?\nPress 3: What weapons?",
                    response
                );
            }
        }
    }
}
