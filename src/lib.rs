//! Dialogue plugin for Bevy: lightweight speaker/receiver abstraction + pluggable local AI (gpt4all backend optional)
pub mod dialogue;

pub mod rag;

pub mod models;

pub mod actions;

pub mod parse;

// Test helpers (exposed to tests & dev tooling)
pub mod test_helpers;

pub use crate::test_helpers::{ask_ai_and_wait, assert_ai_response};

pub mod context;

// Re-export the derive macro
pub use bevy_real_ai_derive::AiParse;

pub mod prelude {
    pub use crate::dialogue::{AIDialoguePlugin, Speaker, DialogueReceiver, DialogueRequest, DialogueResponse, LocalAiHandle, LocalAi,AiRequest, ModelDownloadProgressEvent, ModelLoadCompleteEvent, PendingModelLoads, PendingModelLoad, start_model_load, on_model_load_complete};
    pub use crate::rag::{AiMessage, AiContext, ChatHistory};
    pub use crate::context::{AiSystemContextStore, AiContextGatherConfig, AiEntity, AI, AIAware};
    pub use crate::models::{ModelType, AiModelBuilder, AIModel, SecureString};
    pub use crate::actions::{AiAction, ActionPayload, AiActionEvent, AiActionRegistry,  PendingAiActions, prompt_typed_action};
    pub use crate::parse::{AiParsable, AiSchemaType, extract_and_parse_json, build_typed_prompt};
    pub use crate::AiParse;
    // Keep kalosm exports for backward compatibility
    pub use kalosm::language::{Parser, Schema, Parse};
}
