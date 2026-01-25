//! Dialogue plugin for Bevy: lightweight speaker/receiver abstraction + pluggable local AI (gpt4all backend optional)
pub mod npc_dialogue;

pub mod rag;

pub mod models;

// Test helpers (exposed to tests & dev tooling)
pub mod test_helpers;

pub use crate::test_helpers::{ask_ai_and_wait, assert_ai_response};

pub use crate::context::{ContextGatherConfig, ContextGatherRequest, gather_on_request_world, AiDescribe};
pub use npc_dialogue::LocalAi;

pub mod context;

pub mod prelude {
    pub use crate::npc_dialogue::{AIDialoguePlugin, Speaker, DialogueReceiver, DialogueRequest, DialogueResponse, LocalAiHandle, DialogueRequestQueue, LocalAi};
    pub use crate::rag::{AiContext,AiMessage};
    pub mod context {
        pub use crate::context::{ContextGatherConfig, ContextGatherRequest, gather_on_request_world, AiDescribe};
    }
}
