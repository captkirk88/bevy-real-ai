//! Dialogue plugin for Bevy: lightweight speaker/receiver abstraction + pluggable local AI (gpt4all backend optional)
pub mod dialogue;

pub mod rag;

pub mod models;

// Test helpers (exposed to tests & dev tooling)
pub mod test_helpers;

pub use crate::test_helpers::{ask_ai_and_wait, assert_ai_response};

pub mod context;

pub mod prelude {
    pub use crate::dialogue::{AIDialoguePlugin, Speaker, DialogueReceiver, DialogueRequest, DialogueResponse, LocalAiHandle, DialogueRequestQueue, LocalAi, ModelDownloadProgressEvent, ModelLoadCompleteEvent, PendingModelLoads, PendingModelLoad, start_model_load};
    pub use crate::rag::{AiMessage, AiContext, ChatHistory};
    pub use crate::context::{AiSystemContextStore, AiContextGatherConfig, AiEntity, AI, AIAware};
    pub use crate::models::{ModelType, ModelBuilder, AIModel, SecureString};
}
