use crate::{parse::AiParsable, rag::*};
use bevy::prelude::*;
use flume::{Receiver, Sender, unbounded};
use kalosm::language::BoxedChatModel;
use std::sync::Arc;

/// Event fired when model download progress is updated
#[derive(Event, Clone, Debug)]
pub struct ModelDownloadProgressEvent {
    pub model_name: String,
    pub state: crate::models::DownloadState,
    pub message: String,
    pub progress: Option<f32>, // 0.0 to 100.0 for InProgress, None otherwise
}

/// Event fired when model loading completes (success or failure)
#[derive(Event, Clone, Debug)]
pub struct ModelLoadCompleteEvent {
    pub model_name: String,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Resource to track pending model loads via channels
#[derive(Resource, Default)]
pub struct PendingModelLoads {
    /// List of pending model load operations
    pub loaders: Vec<PendingModelLoad>,
}

/// A pending model load operation with result and progress channels
pub struct PendingModelLoad {
    /// Name of the model being loaded
    pub model_name: String,
    /// Channel receiver for the built model result
    pub result_receiver: crossbeam_channel::Receiver<Result<Arc<dyn LocalAi>, String>>,
    /// Optional channel receiver for download progress updates
    pub progress_receiver:
        Option<crossbeam_channel::Receiver<crate::models::ModelDownloadProgress>>,
}

/// Component for entities that can receive dialogue responses
use crate::actions::{ActionPayload, AiActionEvent};

/// Component for entities that can receive dialogue responses
#[derive(Component, Debug, Clone)]
pub struct DialogueReceiver {
    /// Optional preprogrammed response (immediate, no AI call)
    pub preprogrammed: Option<String>,
    /// Last response received (for testing / display)
    pub last_response: Option<String>,
    /// Actions parsed from the last AI response (if any)
    pub actions: Vec<ActionPayload>,
}

impl DialogueReceiver {
    pub fn new() -> Self {
        Self {
            preprogrammed: None,
            last_response: None,
            actions: Vec::new(),
        }
    }

    pub fn new_with_preprogrammed(response: impl ToString) -> Self {
        Self {
            preprogrammed: Some(response.to_string()),
            last_response: None,
            actions: Vec::new(),
        }
    }
}

impl Default for DialogueReceiver {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum DialogueRequestKind {
    /// Text message with a flag controlling whether gathered context should be included.
    Text {
        message: String,
        include_context: bool,
    },
    Typed {
        user_message: String,
        schema_description: String,
        action_name: String,
    },
}

impl DialogueRequestKind {
    /// Create a text request that includes gathered context.
    pub fn text(message: String) -> Self {
        Self::Text {
            message,
            include_context: true,
        }
    }

    /// Create a typed request from an AiParsable type.
    pub fn typed<Action>(user_msg: String) -> Self
    where
        Action: AiParsable,
    {
        Self::Typed {
            user_message: user_msg, // Placeholder; should be set when creating the request
            schema_description: Action::schema_description(),
            action_name: Action::action_name().to_string(),
        }
    }

    pub fn as_user_message(&self) -> &str {
        match self {
            DialogueRequestKind::Text { message, .. } => message.as_str(),
            DialogueRequestKind::Typed { user_message, .. } => user_message.as_str(),
        }
    }

    pub fn include_context(&self) -> bool {
        match self {
            DialogueRequestKind::Text {
                include_context, ..
            } => *include_context,
            DialogueRequestKind::Typed { .. } => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DialogueRequest {
    pub entity: Entity,
    pub kind: DialogueRequestKind,
}

impl DialogueRequest {
    pub fn text(entity: Entity, prompt: impl Into<String>) -> Self {
        Self {
            entity,
            kind: DialogueRequestKind::Text {
                message: prompt.into(),
                include_context: true,
            },
        }
    }

    /// Create a text request that will *not* include gathered context when sent to the model.
    pub fn text_no_context(entity: Entity, prompt: impl Into<String>) -> Self {
        Self {
            entity,
            kind: DialogueRequestKind::Text {
                message: prompt.into(),
                include_context: false,
            },
        }
    }

    /// Create a typed request with schema description.
    pub fn typed<Action>(entity: Entity, user_message: impl ToString) -> Self
    where
        Action: AiParsable,
    {
        Self {
            entity,
            kind: DialogueRequestKind::typed::<Action>(user_message.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DialogueResponse {
    pub entity: Entity,
    pub response: String,
    pub kind: DialogueRequestKind,
    /// Optional pre-parsed actions (when the response was produced as structured actions).
    pub actions: Option<Vec<ActionPayload>>,
}

use std::collections::VecDeque;

/// Resource holding the queue of outgoing dialogue requests
#[derive(Resource, Default)]
pub struct DialogueRequestQueue {
    queue: VecDeque<DialogueRequest>,
    mutex: std::sync::Mutex<()>,
}

impl DialogueRequestQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            mutex: std::sync::Mutex::new(()),
        }
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn push(&mut self, request: DialogueRequest) {
        let _lock = self.mutex.lock().unwrap();
        debug!(
            "Queued DialogueRequest for entity {:?}: {:?}",
            request.entity, request.kind
        );
        self.queue.push_back(request);
    }

    pub fn pop(&mut self) -> Option<DialogueRequest> {
        let _lock = self.mutex.lock().unwrap();
        self.queue.pop_front()
    }
}

/// System parameter for enqueueing AI requests
#[derive(bevy::ecs::system::SystemParam)]
pub struct AiRequest<'w, 's> {
    queue: ResMut<'w, DialogueRequestQueue>,
    // Second lifetime to satisfy SystemParam signature requirements.
    _marker: std::marker::PhantomData<&'s ()>,
}

impl<'w, 's> AiRequest<'w, 's> {
    /// Convenience method to push a simple text prompt for an entity.
    ///
    /// Adds a short instruction to the prompt to encourage a plain, human-readable
    /// response (no JSON, code blocks, or structured action output).
    pub fn ask_text(&mut self, ai_entity: Entity, prompt: impl ToString) {
        let user_message = format!(
            "{}\n\nPlease respond in plain text only (no JSON or code blocks).",
            prompt.to_string()
        );
        self.queue
            .push(DialogueRequest::text_no_context(ai_entity, user_message));
    }

    /// Inquire with context gathering.
    pub fn inquire(&mut self, ai_entity: Entity, prompt: impl ToString) {
        let user_message = format!(
            "{}\n\nPlease respond in plain text only (no JSON or code blocks).",
            prompt.to_string()
        );
        self.queue
            .push(DialogueRequest::text(ai_entity, user_message));
    }

    /// Ask for a typed [AiParsable] according to the schema of the provided `Action` type.
    pub fn ask_action<Action>(&mut self, ai_entity: Entity, prompt: impl ToString)
    where
        Action: AiParsable,
    {
        let schema_description = Action::schema_description();
        let user_message = format!(
            "{}\nProvide a JSON action matching the following schema:\n{}",
            prompt.to_string(),
            schema_description
        );
        self.queue
            .push(DialogueRequest::typed::<Action>(ai_entity, user_message));
    }
}

/// Result of a prompt with session, containing the response and the updated session.
pub struct PromptResult {
    pub response: String,
    pub session: Option<kalosm::language::BoxedChatSession>,
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
        match self.prompt(messages) {
            Ok(response) => Ok(PromptResult {
                response,
                session: None,
            }),
            Err(e) => Err(e),
        }
    }

    fn get_model(&self) -> BoxedChatModel {
        unimplemented!("get_model is not implemented for this LocalAi backend");
    }

    /// Attempt to produce a typed JSON value according to the provided schema description.
    ///
    /// Default implementation performs post-generation parsing by calling
    /// `prompt_with_session` and extracting JSON from the returned text.
    fn prompt_typed(
        &self,
        messages: &[AiMessage],
        session: Option<kalosm::language::BoxedChatSession>,
        _schema_description: &str,
    ) -> Result<
        (
            serde_json::Value,
            Option<kalosm::language::BoxedChatSession>,
        ),
        String,
    > {
        let prompt_res = self.prompt_with_session(messages, session)?;
        match crate::parse::extract_and_parse_json::<serde_json::Value>(&prompt_res.response) {
            Ok(v) => Ok((v, prompt_res.session)),
            Err(e) => Err(e),
        }
    }
}

/// A handle resource that holds the backend and a channel for responses.
#[derive(Resource)]
pub struct LocalAiHandle {
    /// The AI backend. None until a model is loaded when using async loading.
    pub backend: Option<Arc<dyn LocalAi>>,
    pub tx: Sender<DialogueResponse>,
    pub rx: Receiver<DialogueResponse>,
}

impl LocalAiHandle {
    /// Create a new handle with a backend ready to use.
    pub fn new(backend: Arc<dyn LocalAi>) -> Self {
        let (tx, rx) = unbounded();
        Self {
            backend: Some(backend),
            tx,
            rx,
        }
    }

    /// Create a new handle without a backend (for async loading).
    pub fn new_empty() -> Self {
        let (tx, rx) = unbounded();
        Self {
            backend: None,
            tx,
            rx,
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.backend.is_some()
    }

    pub fn get_backend(&self) -> Option<Arc<dyn LocalAi>> {
        self.backend.clone()
    }
}

use crate::context::{AiContextGatherConfig, AiSystemContextStore, ContextGatherRequest};

/// Plugin that adds NPC dialogue capabilities with the provided LocalAi backend.
pub struct AIDialoguePlugin {
    backend: Option<Arc<dyn LocalAi>>,
    builder: Option<crate::models::AiModelBuilder>,
    pub gather_config: AiContextGatherConfig,
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
    pub fn with_builder(builder: crate::models::AiModelBuilder) -> Self {
        Self {
            backend: None,
            builder: Some(builder),
            ..Default::default()
        }
    }

    pub fn with_config(&mut self, gather_config: AiContextGatherConfig) -> Self {
        Self {
            backend: self.backend.clone(),
            builder: self.builder.clone(),
            gather_config,
        }
    }
}

impl Default for AIDialoguePlugin {
    fn default() -> Self {
        // Default plugin uses the built-in mock backend so tests and examples can work without
        // external dependencies.
        Self {
            backend: None,
            builder: None,
            gather_config: AiContextGatherConfig {
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

        // Determine the initial backend:
        // - If a builder is provided, start with no backend (async load happens in Startup)
        // - If a direct backend is provided, use it immediately
        // - Otherwise fall back to MockAi
        let ai_handle = if self.builder.is_some() {
            // No backend yet - model will be loaded async in Startup
            LocalAiHandle::new_empty()
        } else if let Some(backend) = &self.backend {
            LocalAiHandle::new(backend.clone())
        } else {
            warn!("AIDialoguePlugin: No backend or builder provided. Using MockAi.");
            LocalAiHandle::new(Arc::new(MockAi {}))
        };

        // Insert the AI handle and other resources.
        app.insert_resource(ai_handle)
            .insert_resource(DialogueRequestQueue::default())
            .insert_resource(AiSystemContextStore::new())
            .insert_resource(self.gather_config.clone())
            .insert_resource(ContextGatherRequest::default())
            .insert_resource(PendingModelLoads::default())
            // Register the AiActionEvent and registry for handlers
            .insert_resource(crate::actions::AiActionRegistry::default())
            .insert_resource(crate::actions::PendingAiActions::default());

        // Schedule dialogue request handling first, then gather (which may have been triggered by dialogue),
        // then response polling. This ensures context is gathered in the same frame as the request is made.
        app.add_systems(
            Update,
            (
                handle_dialogue_requests,
                crate::context::gather_on_request_world,
                poll_responses_receiver,
                crate::actions::run_registered_actions_world,
                poll_pending_model_loads,
            )
                .chain(),
        );

        // If a builder was provided, spawn the model loading task asynchronously
        if let Some(builder) = self.builder.clone() {
            app.add_systems(Startup, move |mut pending: ResMut<PendingModelLoads>| {
                start_model_load(&mut pending, "Model".to_string(), builder.clone());
            });
        }
    }

    fn cleanup(&self, app: &mut App) {
        // Clear pending model loads to stop polling for results
        // Resources are dropped naturally when the App is dropped
        if let Some(mut pending) = app.world_mut().get_resource_mut::<PendingModelLoads>() {
            pending.loaders.clear();
        }
    }
}

/// System that polls for completed model loads and triggers events
fn poll_pending_model_loads(
    mut pending: ResMut<PendingModelLoads>,
    mut ai_handle: ResMut<LocalAiHandle>,
    mut commands: Commands,
) {
    // Poll progress receivers and trigger progress events
    for loader in pending.loaders.iter() {
        if let Some(ref progress_rx) = loader.progress_receiver {
            while let Ok(progress) = progress_rx.try_recv() {
                commands.trigger(ModelDownloadProgressEvent {
                    model_name: loader.model_name.clone(),
                    state: progress.state,
                    message: progress.message,
                    progress: progress.progress,
                });
            }
        }
    }

    // Poll result receivers and trigger completion events
    let mut completed_indices = Vec::new();
    for (idx, loader) in pending.loaders.iter().enumerate() {
        if let Ok(result) = loader.result_receiver.try_recv() {
            match result {
                Ok(new_backend) => {
                    ai_handle.backend = Some(new_backend);
                    commands.trigger(ModelLoadCompleteEvent {
                        model_name: loader.model_name.clone(),
                        success: true,
                        error_message: None,
                    });
                }
                Err(e) => {
                    commands.trigger(ModelLoadCompleteEvent {
                        model_name: loader.model_name.clone(),
                        success: false,
                        error_message: Some(e),
                    });
                }
            }
            completed_indices.push(idx);
        }
    }

    // Remove completed loaders (in reverse order to preserve indices)
    for idx in completed_indices.into_iter().rev() {
        pending.loaders.remove(idx);
    }
}

/// System condition used with `run_if` to determine if the model has finished loading.
/// Returns `true` when the `LocalAiHandle` has a backend (model loaded), otherwise `false`.
pub fn on_model_load_complete(ai_handle: Option<Res<LocalAiHandle>>) -> bool {
    ai_handle.map(|h| h.is_loaded()).unwrap_or(false)
}

/// Helper to build a model and track its progress using the resource-based system.
/// Call this from a startup system or any system that needs to load a model asynchronously.
pub fn start_model_load(
    pending: &mut ResMut<PendingModelLoads>,
    model_name: String,
    mut builder: crate::models::AiModelBuilder,
) {
    // Extract progress receiver before building
    let progress_receiver = builder.take_progress_receiver();

    // Create a channel for the built model result
    let (result_tx, result_rx) = crossbeam_channel::unbounded::<Result<Arc<dyn LocalAi>, String>>();

    // Spawn a thread that builds the model and sends the result
    std::thread::spawn(move || match builder.build() {
        Ok(arc_model) => {
            let _ = result_tx.send(Ok(arc_model));
        }
        Err(e) => {
            let _ = result_tx.send(Err(e));
        }
    });

    // Add to pending model loads resource
    pending.loaders.push(PendingModelLoad {
        model_name,
        result_receiver: result_rx,
        progress_receiver,
    });
}

/// System that handles outgoing requests: if NPC has preprogrammed response, respond immediately; else, spawn a thread to call the backend and send result to the response channel.
/// Requests are kept in the queue until the model is loaded.
fn handle_dialogue_requests(
    mut queue: ResMut<DialogueRequestQueue>,
    ai_handle: Res<LocalAiHandle>,
    query: Query<&DialogueReceiver>,
    mut gather_req: Option<ResMut<crate::context::ContextGatherRequest>>,
    gather_store: Option<Res<crate::context::AiSystemContextStore>>,
    ctx_query: Query<&crate::rag::AiContext>,
) {
    // Get the backend, or return early if not loaded yet (requests stay queued)
    let Some(backend) = &ai_handle.backend else {
        return;
    };

    while let Some(req) = queue.pop() {
        // If receiver has a preprogrammed response, short-circuit and send directly
        if let Ok(receiver) = query.get(req.entity) {
            if let Some(pre) = &receiver.preprogrammed {
                let _ = ai_handle.tx.send(DialogueResponse {
                    entity: req.entity,
                    response: pre.clone(),
                    kind: req.kind.clone(),
                    actions: None,
                });
                continue;
            }
        }

        // Signal an on-demand gather for the requester only if the request needs context,
        // there are context-gathering systems registered, and the entity doesn't already have
        // collected context.
        if req.kind.include_context() {
            if let (Some(gr), Some(store)) = (gather_req.as_mut(), gather_store.as_ref()) {
                if !store.systems().is_empty() {
                    // Avoid re-gathering if the entity already has an `AiContext` component
                    if ctx_query.get(req.entity).is_err() {
                        gr.request(req.entity);
                    }
                }
            }
        }

        // Build message vector: include a sentinel System message to suppress the
        // default system context if the request opted out of context.
        let mut messages: Vec<AiMessage> = Vec::new();
        if !req.kind.include_context() {
            messages.push(crate::rag::AiMessage::no_default_system_context());
        }
        if let Ok(ctx) = ctx_query.get(req.entity) {
            // Include gathered context only when the request indicates it should be included.
            if req.kind.include_context() {
                messages.extend_from_slice(ctx.messages());
            }
        }
        // Add the user message from the request kind
        messages.push(AiMessage::user(req.kind.as_user_message()));

        // Call backend on a background task and send result to the response channel
        let backend = backend.clone();
        let tx = ai_handle.tx.clone();
        let msgs = messages.clone();
        let entity = req.entity;
        let kind = req.kind.clone();

        crate::models::TOKIO_RUNTIME.spawn(async move {
            // Compute both the textual response and any pre-parsed actions for typed requests
            let (result, actions_opt) = match &kind {
                DialogueRequestKind::Text { .. } => {
                    let r = backend
                        .prompt(&msgs)
                        .unwrap_or_else(|e| format!("(ai error: {})", e));
                    (r, None)
                }
                DialogueRequestKind::Typed {
                    schema_description,
                    action_name,
                    ..
                } => match backend.prompt_typed(&msgs, None, schema_description) {
                    Ok((val, _)) => {
                        let mut actions: Vec<ActionPayload> = Vec::new();
                        match &val {
                            serde_json::Value::Object(map) => {
                                actions.push(crate::actions::ActionPayload {
                                    name: action_name.clone(),
                                    params: serde_json::Value::Object(map.clone()),
                                });
                            }
                            serde_json::Value::Array(arr) => {
                                for v in arr.iter().cloned() {
                                    actions.push(crate::actions::ActionPayload {
                                        name: action_name.clone(),
                                        params: v,
                                    });
                                }
                            }
                            _ => {}
                        }
                        let s = serde_json::to_string(&val).unwrap_or_else(|_| {
                            "(ai error: failed to serialize typed response)".to_string()
                        });
                        (s, Some(actions))
                    }
                    Err(e) => (format!("(ai error: {})", e), None),
                },
            };

            let _ = tx
                .send_async(DialogueResponse {
                    entity,
                    response: result,
                    kind,
                    actions: actions_opt,
                })
                .await;
        });
    }
}

/// Poll channel and apply responses to receivers.
fn poll_responses_receiver(
    mut query: Query<&mut DialogueReceiver>,
    ai_handle: Res<LocalAiHandle>,
    mut pending: Option<ResMut<crate::actions::PendingAiActions>>,
    mut commands: Commands,
) {
    // Drain all available responses without blocking
    while let Ok(resp) = ai_handle.rx.try_recv() {
        if let Ok(mut receiver) = query.get_mut(resp.entity) {
            // Prefer any pre-parsed actions provided on the response (set for typed requests), otherwise try to interpret the response text as JSON actions.
            let mut actions: Vec<ActionPayload> = Vec::new();

            if let Some(pre) = resp.actions.clone() {
                actions = pre;
            } else if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&resp.response) {
                match json_val {
                    serde_json::Value::Object(map) => {
                        match &resp.kind {
                            DialogueRequestKind::Typed { action_name, .. } => {
                                // Wrap the object as an action with the typed action name
                                actions.push(crate::actions::ActionPayload {
                                    name: action_name.clone(),
                                    params: serde_json::Value::Object(map),
                                });
                            }
                            _ => {
                                if let Some(serde_json::Value::String(_)) = map.get("name") {
                                    if let Some(action) = crate::actions::value_to_action(
                                        serde_json::Value::Object(map),
                                    ) {
                                        actions.push(action);
                                    }
                                }
                            }
                        }
                    }
                    serde_json::Value::Array(arr) => match &resp.kind {
                        DialogueRequestKind::Typed { action_name, .. } => {
                            for v in arr.into_iter() {
                                actions.push(crate::actions::ActionPayload {
                                    name: action_name.clone(),
                                    params: v,
                                });
                            }
                        }
                        _ => {
                            for v in arr.into_iter() {
                                if let Some(action) = crate::actions::value_to_action(v) {
                                    actions.push(action);
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }

            for action in actions.iter() {
                let event = AiActionEvent {
                    entity: resp.entity,
                    action: action.clone(),
                };

                // Log for debugging so we can see what was parsed and enqueued
                debug!(
                    "Enqueuing AI action '{}' for entity {:?} with params: {}",
                    action.name, resp.entity, action.params
                );

                // Push into pending actions resource so the world-runner can execute handlers
                if let Some(p) = pending.as_mut() {
                    p.actions.push(event.clone());
                }

                // Also emit an event so other systems can react if they want
                commands.trigger(event);
            }

            // Store parsed actions
            receiver.actions = actions;

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
