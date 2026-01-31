//! Example demonstrating typed AI action parsing with our custom `AiParse` derive macro.
//!
//! Run with: `cargo run --example actions --release`
//!
//! This example uses `prompt_typed_action` to ask the AI to spawn an entity at a
//! specific location. The response is automatically parsed into a typed struct
//! and queued as an action for the registered handler.

use bevy::prelude::*;
use bevy_real_ai::prelude::*;
use serde::{Deserialize, Serialize};

/// Marker component for spawned AI entities (recommended for tracking)
#[derive(Component)]
pub struct AiSpawned;

#[derive(Resource)]
struct ExamplePlayer(Entity);

/// Typed struct for spawn actions - the AI will produce JSON matching this schema.
/// The `AiParse` derive automatically implements both `AiParsable` (for JSON schema
/// generation and parsing) and `IntoActionPayload` (for action conversion).
/// The action name is derived from the struct name in snake_case: "spawn_entity_action"
#[derive(Clone, Debug, Serialize, Deserialize, AiParse)]
struct SpawnEntityAction {
    pub prefab: String,
    pub x: f32,
    pub y: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // Use async model builder (loads in background)
        .add_plugins(AIDialoguePlugin::with_builder(
            AiModelBuilder::new_with(ModelType::Llama)
                .with_seed(42)
                .with_progress(),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, check_response)
        //.add_observer(on_model_load_complete)
        .run();
}

fn setup(mut commands: Commands, mut registry: ResMut<AiActionRegistry>, mut request: AiRequest) {
    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let player = commands
        .spawn((
            DialogueReceiver::new(),
            AI,
        ))
        .id();

    // Store the player so the model-load observer can queue a request later
    commands.insert_resource(ExamplePlayer(player));

    // Register a spawn handler using the auto-generated register method.
    // The handler receives the typed struct as In<SpawnEntityAction> and can use any Bevy system params.
    SpawnEntityAction::register(
        &mut registry,
        |In(action): In<SpawnEntityAction>, mut cmds: Commands,
                mut meshes: ResMut<Assets<Mesh>>,
            mut materials: ResMut<Assets<StandardMaterial>>,
    | {
            cmds.spawn((
                AiSpawned,
                Name::new(format!("spawned:{}", action.prefab)),
                Transform::from_translation(Vec3::new(action.x, action.y, 0.0)),
                Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::linear_rgb(0.0, 1.0, 0.0),
                    ..Default::default()
                })),
            ));
            info!(
                "Registry handler spawned '{}' at ({}, {})",
                action.prefab, action.x, action.y
            );
        },
    );

    request.ask_action::<SpawnEntityAction>(
        player,
        "Spawn a goblin enemy at position x=0.0 and y=0.0",
    );

    request.ask_text(
        player,
        "Also, can you tell me a joke while you're at it?",
    );
}

/// Observer that queues a typed action request when the model finishes loading.
#[allow(unused)]
fn on_model_load_complete(
    trigger: On<ModelLoadCompleteEvent>,
    player: Option<Res<ExamplePlayer>>,
    ai_handle: Option<Res<LocalAiHandle>>,
    mut pending: ResMut<PendingAiActions>,
) {
    let event = trigger.event();
    if event.success {
        if let (Some(p), Some(ai)) = (player, ai_handle) {
            if let Some(backend) = &ai.backend {
                // Use prompt_typed_action to ask the AI for a spawn action.
                // The schema is automatically included in the prompt, and the
                // response is parsed into SpawnEntityAction then queued.
                match prompt_typed_action::<SpawnEntityAction>(
                    backend,
                    "Spawn a goblin enemy at position x=10.0 and y=-5.0",
                    p.0,
                    &mut pending,
                ) {
                    Ok(parsed) => {
                        info!("AI produced typed action: {:?}", parsed);
                    }
                    Err(e) => {
                        error!("Failed to parse AI response: {}", e);
                    }
                }
            }
        }
    } else if let Some(ref err) = event.error_message {
        info!("Model failed to load: {}", err);
    }
}

/// Check for human-visible response and log it
fn check_response(query: Query<&DialogueReceiver, Changed<DialogueReceiver>>) {
    for receiver in query.iter() {
        if let Some(response) = &receiver.last_response {
            info!("AI: {}", response);
        }
    }
}
