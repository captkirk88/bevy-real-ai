use crate::rag::*;
use bevy::prelude::*;
use flume::{Receiver, Sender, unbounded};
use std::sync::Arc;

/// Event fired when model download progress is updated
#[derive(Event, Clone, Debug)]
pub struct ModelDownloadProgressEvent {
    pub model_name: String,
    pub state: crate::models::DownloadState,
    pub message: String,
    pub progress: Option<f32>, // 0.0 to 1.0 for InProgress, None otherwise
}

/// Component to track an active model download and emit progress events
#[derive(Component)]
pub struct ModelDownloadTracker {
    pub model_name: String,
    pub progress_receiver: crossbeam_channel::Receiver<crate::models::ModelDownloadProgress>,
}

/// Component to track completion of model loading
#[derive(Component)]
pub struct ModelLoadComplete {
    pub result_receiver: crossbeam_channel::Receiver<Result<Arc<dyn LocalAi>, String>>,
}

/// Public re-exports and types for users
#[derive(Component, Debug, Clone)]
pub struct Speaker {
    pub name: String,
    /// Prompt template; eg. "{name}: {message}" or use directly as prompt
    pub prompt_template: String,
}

impl Speaker {
    /// Create new Speaker
    /// # Arguments
    /// * `name` - Name of the speaker
    /// * `prompt_template` - Prompt template to use for this speaker
    pub fn new(name: &str, prompt_template: &str) -> Self {
        Self {
            name: name.to_string(),
            prompt_template: prompt_template.to_string(),
        }
    }
}

/// Component for entities that can receive dialogue responses
#[derive(Component, Debug, Clone)]
pub struct DialogueReceiver {
    /// Optional preprogrammed response (immediate, no AI call)
    pub preprogrammed: Option<String>,
    /// Last response received (for testing / display)
    pub last_response: Option<String>,
}

impl DialogueReceiver {
    pub fn new() -> Self {
        Self {
            preprogrammed: None,
            last_response: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DialogueRequest {
    pub entity: Entity,
    pub prompt: String,
}

#[derive(Debug, Clone)]
pub struct DialogueResponse {
    pub entity: Entity,
    pub response: String,
}

use std::collections::VecDeque;

/// Resource holding the queue of outgoing dialogue requests
#[derive(Resource, Default)]
pub struct DialogueRequestQueue {
    pub queue: VecDeque<DialogueRequest>,
}

/// Result of a prompt with session, containing the response and the updated session.
pub struct PromptResult {
    pub response: String,
    pub session: kalosm::language::BoxedChatSession,
}

/// Trait to abstract local AI backends. Implementors should be quick to return or be used from a background thread.
pub trait LocalAi: Send + Sync + 'static {
    /// Accepts an iterator of `Message` so backends can distinguish
    /// between system/context and user messages without string parsing.
    fn prompt(&self, messages: &[AiMessage]) -> Result<String, String>;

    /// Prompt with an optional existing session, returning the response and updated session.
    /// This allows conversation history to be preserved across calls.
    /// The default implementation creates a new session each time.
    fn prompt_with_session(
        &self,
        messages: &[AiMessage],
        _session: Option<kalosm::language::BoxedChatSession>,
    ) -> Result<PromptResult, String> {
        // Default implementation ignores session and just calls prompt
        let _ = self.prompt(messages)?;
        // Return a dummy error since we can't create a session without the model
        Err("prompt_with_session not supported by this backend".to_string())
    }
}

/// A handle resource that holds the backend and a channel for responses.
#[derive(Resource)]
pub struct LocalAiHandle {
    pub backend: Arc<dyn LocalAi>,
    pub tx: Sender<DialogueResponse>,
    pub rx: Receiver<DialogueResponse>,
}

impl LocalAiHandle {
    pub fn new(backend: Arc<dyn LocalAi>) -> Self {
        let (tx, rx) = unbounded();
        Self { backend, tx, rx }
    }
}

use crate::context::{ContextGatherConfig, ContextGatherRequest, AiSystemContextStore};

/// Plugin that adds NPC dialogue capabilities with the provided LocalAi backend.
///
/// Resources:
/// - `LocalAiHandle`: holds the backend and response channel
/// - `DialogueRequestQueue`: queue of outgoing dialogue requests
/// - `AiSystemContextStore`: registry for context-gathering systems (used by RAG)
pub struct AIDialoguePlugin {
    pub backend: Option<Arc<dyn LocalAi>>,
    pub builder: Option<crate::models::ModelBuilder>,
    pub gather_config: ContextGatherConfig,
}

impl AIDialoguePlugin {
    /// Create a plugin with a prebuilt backend.
    pub fn with_backend(backend: Arc<dyn LocalAi>) -> Self {
        Self {
            backend: Some(backend.clone()),
            builder: None,
            ..Default::default()
        }
    }

    /// Create a plugin with a model builder for async loading.
    /// Starts with MockAi while the model loads in the background.
    pub fn with_builder(builder: crate::models::ModelBuilder) -> Self {
        Self {
            backend: None,
            builder: Some(builder),
            ..Default::default()
        }
    }
}

impl Default for AIDialoguePlugin {
    fn default() -> Self {
        // Default plugin uses the built-in mock backend so tests and examples can work without
        // external dependencies.
        Self {
            backend: Some(Arc::new(MockAi {})),
            builder: None,
            gather_config: ContextGatherConfig {
                radius: 5.0,
                max_docs: 8,
            },
        }
    }
}

impl Plugin for AIDialoguePlugin {
    fn build(&self, app: &mut App) {
        if let Some(_) = app.world().get_resource::<LocalAiHandle>() {
            // Already added; skip
            return;
        }

        let backend = if let Some(builder) = &self.builder {
            match builder.build_chat() {
                Ok(arc_model) => arc_model,
                Err(e) => {
                    eprintln!("AIDialoguePlugin: Failed to build model from builder: {}. Falling back to MockAi.", e);
                    Arc::new(MockAi {})
                }
            }
        } else {
            self.backend.clone().unwrap_or_else(|| {
                eprintln!("AIDialoguePlugin: No backend or builder provided. Using MockAi.");
                Arc::new(MockAi {})
            })
        };
        
        // Insert the provided backend into a LocalAiHandle and add the request queue and systems.
        app.insert_resource(LocalAiHandle::new(backend))
            .insert_resource(DialogueRequestQueue::default())
            .insert_resource(AiSystemContextStore::new())
            .insert_resource(self.gather_config.clone())
            .insert_resource(ContextGatherRequest::default());

        // Schedule dialogue request handling first, then gather (which may have been triggered by dialogue),
        // then response polling. This ensures context is gathered in the same frame as the request is made.
        app.add_systems(
            Update,
            (handle_dialogue_requests, crate::context::gather_on_request_world, poll_responses_receiver, poll_model_download_progress, apply_loaded_model)
                .chain(),
        );

        // Register observer for model download progress events
        app.add_observer(handle_model_download_progress);

        // If a builder was provided, spawn the model loading task
        if let Some(builder) = self.builder.clone() {
            app.add_systems(Startup, move |commands: Commands| {
                spawn_model_load_tracker(commands, "Model".to_string(), builder.clone());
            });
        }
    }
}

/// System that polls for completed model loads and updates LocalAiHandle
fn apply_loaded_model(
    mut query: Query<(Entity, &ModelLoadComplete)>,
    mut ai_handle: ResMut<LocalAiHandle>,
    mut commands: Commands,
) {
    for (entity, loader) in query.iter_mut() {
        if let Ok(result) = loader.result_receiver.try_recv() {
            match result {
                Ok(new_backend) => {
                    eprintln!("Model loading complete, updating backend");
                    ai_handle.backend = new_backend;
                    commands.entity(entity).despawn();
                }
                Err(e) => {
                    eprintln!("Model loading failed: {}", e);
                    commands.entity(entity).despawn();
                }
            }
        }
    }
}

/// System that polls model download progress and triggers events
fn poll_model_download_progress(
    query: Query<&ModelDownloadTracker>,
    mut commands: Commands,
) {
    for tracker in query.iter() {
        while let Ok(progress) = tracker.progress_receiver.try_recv() {
            commands.trigger(ModelDownloadProgressEvent {
                model_name: tracker.model_name.clone(),
                state: progress.state,
                message: progress.message,
                progress: progress.progress,
            });
        }
    }
}


/// Observer that handles model download progress events and logs them
fn handle_model_download_progress(
    trigger: On<ModelDownloadProgressEvent>,
) {
    let event = trigger.event();
    match event.state {
        crate::models::DownloadState::InProgress => {
            if let Some(p) = event.progress {
                eprintln!("[{}] Downloading... {:.1}% - {}", event.model_name, p, event.message);
            } else {
                eprintln!("[{}] {}", event.model_name, event.message);
            }
        }
        crate::models::DownloadState::Completed => {
            eprintln!("[{}] ✓ Download completed", event.model_name);
        }
        crate::models::DownloadState::Error => {
            eprintln!("[{}] ✗ Download error: {}", event.model_name, event.message);
        }
    }
}

/// Helper to build a model and track its progress with component spawning
/// Use this in a system to load a model asynchronously while tracking progress
pub fn spawn_model_load_tracker(
    mut commands: Commands,
    model_name: String,
    mut builder: crate::models::ModelBuilder,
) -> Entity {
    // Extract progress receiver before building
    let progress_receiver = builder.take_progress_receiver();

    // Create a channel for the built model result
    let (result_tx, result_rx) = crossbeam_channel::unbounded::<Result<Arc<dyn LocalAi>, String>>();

    // Spawn a thread that builds the model and sends the result
    std::thread::spawn(move || {
        match builder.build_chat() {
            Ok(arc_model) => {
                let _ = result_tx.send(Ok(arc_model));
            }
            Err(e) => {
                let _ = result_tx.send(Err(e));
            }
        }
    });

    // Spawn an entity with both progress and result trackers
    let entity = commands
        .spawn_empty()
        .insert(ModelLoadComplete {
            result_receiver: result_rx,
        })
        .id();

    if let Some(rx) = progress_receiver {
        // Add progress tracker if progress reporting was enabled
        commands.entity(entity).insert(ModelDownloadTracker {
            model_name: model_name.clone(),
            progress_receiver: rx,
        });
    }

    entity
}


/// System that handles outgoing requests: if NPC has preprogrammed response, respond immediately; else, spawn a thread to call the backend and send result to the response channel.
fn handle_dialogue_requests(
    mut queue: ResMut<DialogueRequestQueue>,
    ai_handle: Res<LocalAiHandle>,
    query: Query<&DialogueReceiver>,
    mut gather_req: Option<ResMut<crate::context::ContextGatherRequest>>,
    ctx_query: Query<&crate::rag::AiContext>,
) {
    while let Some(req) = queue.queue.pop_front() {
        // If receiver has a preprogrammed response, short-circuit and send directly
        if let Ok(receiver) = query.get(req.entity) {
            if let Some(pre) = &receiver.preprogrammed {
                let _ = ai_handle.tx.send(DialogueResponse {
                    entity: req.entity,
                    response: pre.clone(),
                });
                continue;
            }
        }

        // Signal an on-demand gather for the requester; the exclusive gather system will run
        if let Some(gr) = gather_req.as_mut() {
            gr.0 = Some(req.entity);
        }

        // Build message vector: if context docs are available, include them as system messages
        let mut messages: Vec<AiMessage> = Vec::new();
        if let Ok(ctx) = ctx_query.get(req.entity) {
            messages.extend_from_slice(ctx.messages());
        }
        messages.push(AiMessage::user(req.prompt.as_str()));

        // Call backend on a background thread with message-style prompt and send result to the response channel
        let backend = ai_handle.backend.clone();
        let tx = ai_handle.tx.clone();
        let msgs = messages.clone();
        let entity = req.entity;
        std::thread::spawn(move || {
            let result = backend
                .prompt(&msgs)
                .unwrap_or_else(|e| format!("(ai error: {})", e));
            let _ = tx.send(DialogueResponse {
                entity,
                response: result,
            });
        });
    }
}

/// Poll channel and apply responses to receivers.
fn poll_responses_receiver(mut query: Query<&mut DialogueReceiver>, ai_handle: Res<LocalAiHandle>) {
    // Drain all available responses without blocking
    while let Ok(resp) = ai_handle.rx.try_recv() {
        if let Ok(mut receiver) = query.get_mut(resp.entity) {
            receiver.last_response = Some(resp.response.trim().to_string());
        }
    }
}

/// A very small mock AI backend used by default and for tests.
pub struct MockAi {}

impl LocalAi for MockAi {
    fn prompt(&self, messages: &[AiMessage]) -> Result<String, String> {
        // Return the first user message content when present, else debug-join messages.
        for m in messages.iter() {
            let dbg = format!("{:?}", m);
            // crude heuristic: find quoted content in debug output
            if let Some(start) = dbg.find('"') {
                if let Some(end) = dbg[start + 1..].find('"') {
                    let content = &dbg[start + 1..start + 1 + end];
                    return Ok(format!("mock: {}", content));
                }
            }
        }
        let combined = messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(format!("mock: {}", combined))
    }
}
