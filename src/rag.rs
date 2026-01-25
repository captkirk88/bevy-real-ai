use bevy::prelude::Component;


#[derive(Debug, Clone, PartialEq)]
pub enum AiMessage {
    /// System-level message (context, instructions, etc)
    System(String),
    /// User message (from human/user)
    User(String),
    /// Assistant message (from AI)
    #[deprecated(note = "Assistant messages are not currently used in prompt construction")]
    Assistant(String),
}

impl AiMessage {
    pub fn system(text: &str) -> Self {
        AiMessage::System(text.to_string())
    }

    pub fn user(text: &str) -> Self {
        AiMessage::User(text.to_string())
    }

    #[allow(deprecated)]
    pub fn assistant(text: &str) -> Self {
        AiMessage::Assistant(text.to_string())
    }
}

impl From<String> for AiMessage {
    fn from(s: String) -> Self {
        AiMessage::User(s)
    }
}

impl std::fmt::Display for AiMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AiMessage::System(text) => write!(f, "System: {}", text),
            AiMessage::User(text) => write!(f, "User: {}", text),
            #[allow(deprecated)] AiMessage::Assistant(text) => write!(f, "Assistant: {}", text),
        }
    }
}



/// Component storing AI context messages for an entity.
#[derive(Debug, Clone, Component)]
pub struct AiContext {
    messages: Vec<AiMessage>,
}

impl AiContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self { messages: Vec::new() }
    }

    /// Add context as a system message from an opaque text string.
    pub fn add_context(&mut self, text: impl Into<String>) {
        self.messages.push(AiMessage::system(text.into().as_str()));
    }

    /// Access internal messages (primarily for backend/internal framework use).
    /// This exposes the raw Message type but hides the direct public field.
    pub fn messages(&self) -> &[AiMessage] {
        &self.messages
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

