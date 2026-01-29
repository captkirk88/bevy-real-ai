//! A simple Bevy example demonstrating AI-powered NPC dialogue.
//!
//! Run with: `cargo run --example npc --release`
//!
//! This example creates a simple game world with:
//! - A player entity
//! - NPCs with items and descriptions
//! - An AI-powered dialogue system that responds based on NPC context
//! - A progress bar showing model download status
//!
//! The demo automatically asks all questions once the model loads.
//! You can also press 1-3 to ask questions manually.

use bevy::ecs::observer::On;
use bevy::prelude::*;
use bevy_ai_dialogue::context::{AiSystemContextStore, ContextGatherRequest, AI, AIAware, AiEntity};
use bevy_ai_dialogue::models::{DownloadState, ModelBuilder};
use bevy_ai_dialogue::prelude::*;

/// Questions available for the player to ask (label, prompt)
static QUESTIONS: &[(&str, &str)] = &[
    ("Who is nearby?", "Who is nearby and what do they have?"),
    ("Where's a healing potion?", "Where can I find a healing potion?"),
    ("What weapons?", "What weapons are available?"),
];

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(AIDialoguePlugin::with_builder(
            ModelBuilder::new()
                .with_seed(42) // Optional: use fixed seed for consistent responses
                .with_progress(), // Enable progress tracking for download UI
        ).with_config(AiContextGatherConfig::default().with_radius(50.0)))
        .add_systems(Startup, setup_world)
        .add_systems(Update, (handle_input, display_dialogue, update_progress_bar))
        .add_observer(on_model_download_progress)
        .add_observer(on_model_load_complete)
        .run();
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

/// Progress bar UI components
#[derive(Component)]
struct ProgressBarUI;

#[derive(Component)]
struct ProgressBarFill;

#[derive(Component)]
struct ProgressBarText;

/// Component to track download progress on the progress bar entity
#[derive(Component, Default)]
struct DownloadProgress {
    progress: f32,
}

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
            name: "Bob".into(),
            description: "A strong craftsperson".into(),
        },
        Inventory {
            items: vec!["iron sword".into(), "steel shield".into(), "hammer".into()],
        },
        Transform::from_translation(Vec3::new(50.0, 0.0, 0.0)),
        AIAware, // This entity's info will be gathered for AI context
    ));

    commands.spawn((
        Npc {
            name: "Elena".into(),
            description: "A knowledgeable woman".into(),
        },
        Inventory {
            items: vec!["healing potion".into(), "antidote".into(), "rare mushroom".into()],
        },
        Transform::from_translation(Vec3::new(-30.0, 20.0, 0.0)),
        AIAware,
    ));

    commands.spawn((
        Npc {
            name: "Marcus".into(),
            description: "A skilled fighter".into(),
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
        Text::new("Loading AI model...\nPlease wait for download to complete."),
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

    // Spawn progress bar container (centered near top of screen)
    commands
        .spawn((
            ProgressBarUI,
            DownloadProgress::default(),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(100.0),
                left: Val::Percent(50.0),
                width: Val::Px(400.0),
                height: Val::Px(30.0),
                margin: UiRect::left(Val::Px(-200.0)), // Center the bar
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                ..default()
            },
        ))
        .with_children(|parent| {
            // Progress bar background
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(20.0),
                        border: UiRect::all(Val::Px(2.0)),
                        ..default()
                    },
                    BorderColor::all(Color::WHITE),
                    BackgroundColor(Color::srgba(0.2, 0.2, 0.2, 0.8)),
                ))
                .with_children(|bar_parent| {
                    // Progress bar fill
                    bar_parent.spawn((
                        ProgressBarFill,
                        Node {
                            width: Val::Percent(0.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.3, 0.7, 0.3)),
                    ));
                });

            // Progress text
            parent.spawn((
                ProgressBarText,
                Text::new("Downloading model: 0%"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                Node {
                    margin: UiRect::top(Val::Px(5.0)),
                    ..default()
                },
            ));
        });

    info!("World setup complete! Waiting for model to download...");
}

/// Context gathering system - provides NPC information to the AI
fn gather_npc_context(
    ai_entity: AiEntity,
    npcs: Query<(Entity, &Npc, &Inventory, &Transform), With<AIAware>>,
) -> Option<bevy_ai_dialogue::rag::AiMessage> {
    let nearby_entities = ai_entity.collect_nearby();
    
    let summaries: Vec<String> = npcs
        .iter()
        .filter(|(ent, _, _, _)| nearby_entities.contains(ent))
        .map(|(_, npc, inventory, _)| {
            format!(
                "{}. Description: {}. Possessions: {}.",
                npc.name,
                npc.description,
                inventory.items.join(", ")
            )
        })
        .collect();

    if summaries.is_empty() {
        None
    } else {
        Some(bevy_ai_dialogue::rag::AiMessage::system(&summaries.join("\n")))
    }
}

/// Handle player input for dialogue
fn handle_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    player_query: Query<Entity, With<Player>>,
    mut context_request: ResMut<ContextGatherRequest>,
    mut request_queue: ResMut<bevy_ai_dialogue::dialogue::DialogueRequestQueue>,
) {
    let Ok(player_entity) = player_query.single() else {
        return;
    };

    // Different questions the player can ask with number keys
    let keys = [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3];
    let prompt = keys
        .iter()
        .zip(QUESTIONS.iter())
        .find(|(key, _)| keyboard.just_pressed(**key))
        .map(|(_, (_, prompt))| *prompt);

    if let Some(prompt) = prompt {
        // Gather context before asking
        context_request.request(player_entity);
        
        info!("Asking: {}", prompt);
        request_queue.push(bevy_ai_dialogue::dialogue::DialogueRequest {
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
                let options: String = QUESTIONS
                    .iter()
                    .enumerate()
                    .map(|(i, (label, _))| format!("Press {}: {}", i + 1, label))
                    .collect::<Vec<_>>()
                    .join("\n");
                **text = format!("AI: {}\n\n{}", response, options);
            }
        }
    }
}

/// Observer that handles model download progress events and updates the progress component
fn on_model_download_progress(
    trigger: On<ModelDownloadProgressEvent>,
    mut progress_query: Query<&mut DownloadProgress>,
) {
    let event = trigger.event();
    let Ok(mut progress) = progress_query.single_mut() else {
        return;
    };
    
    match event.state {
        DownloadState::InProgress => {
            if let Some(p) = event.progress {
                progress.progress = p; // Already 0.0-100.0 range
            }
        }
        DownloadState::Completed => {
            progress.progress = 1.0;
        }
        DownloadState::Error => {
            info!("Model download error: {}", event.message);
        }
    }
}

/// Observer that handles model load complete events - despawns progress UI and gathers context
fn on_model_load_complete(
    trigger: On<ModelLoadCompleteEvent>,
    mut commands: Commands,
    progress_ui_query: Query<Entity, With<ProgressBarUI>>,
    mut ui_query: Query<&mut Text, With<DialogueUI>>,
    player_query: Query<Entity, With<Player>>,
    mut context_request: ResMut<ContextGatherRequest>,
) {
    let event = trigger.event();
    if event.success {
        info!("Model '{}' loaded successfully!", event.model_name);
        
        // Despawn the progress bar UI - we no longer need it
        if let Ok(entity) = progress_ui_query.single() {
            commands.entity(entity).despawn();
        }
        
        // Update the dialogue UI with instructions
        if let Ok(mut text) = ui_query.single_mut() {
            let options: String = QUESTIONS
                .iter()
                .enumerate()
                .map(|(i, (label, _))| format!("Press {}: {}", i + 1, label))
                .collect::<Vec<_>>()
                .join("\n");
            **text = format!("Model loaded! Ready to chat.\n\n{}", options);
        }

        // Automatically gather context from nearby NPCs
        if let Ok(player_entity) = player_query.single() {
            info!("Gathering context from nearby NPCs...");
            context_request.request(player_entity);
        }
    } else if let Some(ref err) = event.error_message {
        info!("Model '{}' failed to load: {}", event.model_name, err);
    }
}

/// System to update the progress bar UI based on download progress component
fn update_progress_bar(
    progress_query: Query<&DownloadProgress>,
    mut fill_query: Query<&mut Node, With<ProgressBarFill>>,
    mut text_query: Query<&mut Text, With<ProgressBarText>>,
) {
    let Ok(progress) = progress_query.single() else {
        return; // Progress entity was despawned
    };
    
    // Update progress bar fill width (progress is 0.0-1.0)
    if let Ok(mut fill_node) = fill_query.single_mut() {
        fill_node.width = Val::Percent(progress.progress * 100.0);
    }

    // Update progress text
    if let Ok(mut text) = text_query.single_mut() {
        **text = format!("Downloading model: {:.0}%", progress.progress * 100.0);
    }
}
