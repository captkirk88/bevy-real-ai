#![allow(dead_code)]

use crate::dialogue::{DialogueReceiver, DialogueRequest, DialogueRequestQueue};
use bevy::prelude::*;
use std::time::Duration;

/// Ask the AI (by queueing a `DialogueRequest`) and wait for a response to
/// appear on the provided `entity`'s `DialogueReceiver`.
///
/// Returns the response if it arrived within `max_updates` update iterations,
/// otherwise returns `None`.
pub fn ask_ai_and_wait(
    app: &mut App,
    entity: Entity,
    prompt: &str,
    max_updates: usize,
) -> Option<String> {
    // Push request onto the request queue
    let mut req_queue = app.world_mut().resource_mut::<DialogueRequestQueue>();
    req_queue.push(DialogueRequest::text(entity, prompt.to_string()));

    for _ in 0..max_updates {
        app.update();
        if let Some(receiver) = app.world().get::<DialogueReceiver>(entity) {
            if let Some(resp) = &receiver.last_response {
                return Some(resp.clone());
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    }

    None
}

/// Ask the AI and wait for a response, returning a Result.
///
/// Returns `Ok(response)` if successful, `Err(msg)` if timeout or empty response.
pub fn ask_ai_and_wait_result(
    app: &mut App,
    entity: Entity,
    prompt: &str,
    max_updates: usize,
) -> Result<String, String> {
    ask_ai_and_wait(app, entity, prompt, max_updates)
        .ok_or_else(|| format!("AI response timeout after {} updates", max_updates))
}

/// Convenience helper that asserts the AI response matches the provided predicate.
///
/// Panics if no response arrives in time or if the predicate returns false.
pub fn assert_ai_response<F: Fn(&str) -> bool>(
    app: &mut App,
    entity: Entity,
    prompt: &str,
    max_updates: usize,
    predicate: F,
) {
    match ask_ai_and_wait(app, entity, prompt, max_updates) {
        Some(resp) => assert!(
            predicate(&resp),
            "AI response did not match predicate: {}",
            resp
        ),
        None => panic!("No AI response within time limit"),
    }
}

/// Set the given `contexts` (each a short text) as context entries on `entity`.
pub fn set_ai_context(app: &mut App, entity: Entity, contexts: &[&str]) {
    use crate::rag::AiContext;
    let mut context = AiContext::new();
    for s in contexts.iter() {
        context.add_context(*s);
    }
    // Insert or replace the component
    app.world_mut().entity_mut(entity).insert(context);
}

/// Convenience that sets `AiContext` and then asks the AI, waiting for response.
pub fn ask_ai_with_context(
    app: &mut App,
    entity: Entity,
    contexts: &[&str],
    prompt: &str,
    max_updates: usize,
) -> Option<String> {
    set_ai_context(app, entity, contexts);
    ask_ai_and_wait(app, entity, prompt, max_updates)
}
