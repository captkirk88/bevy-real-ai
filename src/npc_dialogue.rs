use bevy::prelude::*;
use crate::rag::*;
use flume::{Receiver, Sender, unbounded};
use std::sync::Arc;

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

#[derive(Resource, Default)]
pub struct DialogueRequestQueue {
    pub queue: VecDeque<DialogueRequest>,
}

/// Trait to abstract local AI backends. Implementors should be quick to return or be used from a background thread.
pub trait LocalAi: Send + Sync + 'static {
    /// Accepts an iterator of `Message` so backends can distinguish
    /// between system/context and user messages without string parsing.
    fn prompt(&self, messages: &[AiMessage]) -> Result<String, String>;
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

use crate::context::ContextExtractorRegistry;
use crate::{ContextGatherConfig, ContextGatherRequest};


/// Plugin that adds NPC dialogue capabilities with the provided LocalAi backend.
///
/// Resources:
/// - `LocalAiHandle`: holds the backend and response channel
/// - `DialogueRequestQueue`: queue of outgoing dialogue requests
/// - `ContextExtractorRegistry`: registry for context extractors (used by RAG)
pub struct AIDialoguePlugin {
    pub backend: Arc<dyn LocalAi>,
    pub gather_config: ContextGatherConfig,
}

impl AIDialoguePlugin {
    /// Create a new plugin with the provided backend.
    pub fn new(backend: Arc<dyn LocalAi>) -> Self {
        Self { backend, ..Default::default() }
    }
}

impl Default for AIDialoguePlugin {
    fn default() -> Self {
        // Default plugin uses the built-in mock backend so tests and examples can work without
        // external dependencies.
        Self {
            backend: Arc::new(MockAi {}),
            gather_config: ContextGatherConfig {
                radius: 5.0,
                max_docs: 8,
            },
        }
    }
}


impl Plugin for AIDialoguePlugin {
    fn build(&self, app: &mut App) {
        // Insert the provided backend into a LocalAiHandle and add the request queue and systems.
        app.insert_resource(LocalAiHandle::new(self.backend.clone()))
            .insert_resource(DialogueRequestQueue::default())
            .insert_resource(ContextExtractorRegistry::new())
            .insert_resource(self.gather_config.clone())
            .insert_resource(ContextGatherRequest::default());

        // Schedule the exclusive gather system first, then the normal systems that use it.
        app.add_systems(Update, crate::context::gather_on_request_world);
        app.add_systems(Update, (handle_dialogue_requests, poll_responses_receiver));
    }
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

        // Debug: print all messages being sent to AI
        #[cfg(debug_assertions)]
        {
            eprintln!("DEBUG: Messages to AI:");
            for (i, msg) in messages.iter().enumerate() {
                eprintln!("  {}: {:?}", i, msg);
            }
        }

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
    // Drain available responses without blocking
    while let Ok(resp) = ai_handle.rx.try_recv() {
        if let Ok(mut receiver) = query.get_mut(resp.entity) {
            receiver.last_response = Some(resp.response.clone());
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
                if let Some(end) = dbg[start+1..].find('"') {
                    let content = &dbg[start+1..start+1+end];
                    return Ok(format!("mock: {}", content));
                }
            }
        }
        let combined = messages.iter().map(|m| format!("{:?}", m)).collect::<Vec<_>>().join(" ");
        Ok(format!("mock: {}", combined))
    }
}
