//! Dialogue plugin for Bevy: lightweight speaker/receiver abstraction + pluggable local AI (gpt4all backend optional)
pub mod dialogue;

pub mod rag;

pub mod models;

pub mod actions;

pub mod parse;

mod app_ext;

// Test helpers (exposed to tests & dev tooling)
pub mod test_helpers;

pub use crate::test_helpers::{ask_ai_and_wait, assert_ai_response};

pub mod context;

// Re-export the derive macro
pub use bevy_real_ai_derive::AiAction;

pub mod prelude {
    pub use crate::AiAction;
    pub use crate::actions::{
        ActionPayload, AiActionEvent, AiActionRegistry, PendingAiActions,
        prompt_typed_action,
    };
    pub use crate::app_ext::AiAppExt;
    pub use crate::context::{
        AI, AIAware, AiContextGatherConfig, AiEntity, AiSystemContextStore, ContextGatherRequest,
    };
    pub use crate::dialogue::{
        AIDialoguePlugin, AiRequest, DialogueReceiver, DialogueRequest, DialogueResponse, LocalAi,
        LocalAiHandle, ModelDownloadProgressEvent, ModelLoadCompleteEvent, PendingModelLoad,
        PendingModelLoads, on_model_load_complete, start_model_load,
    };
    pub use crate::models::{AIModel, AiModelBuilder, DownloadState, ModelType, SecureString};
    pub use crate::parse::{AiParsable, build_typed_prompt, extract_and_parse_json};
    pub use crate::rag::{AiContext, AiMessage, ChatHistory};
    // Keep kalosm exports for backward compatibility
    pub use kalosm::language::{Parse, Parser, Schema};
}
