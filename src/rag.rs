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
    /// A pre-parsed action payload (used to pass actions without reparsing text)
    Payload(crate::actions::ActionPayload),
}

impl AiMessage {
    pub fn system(text: &str) -> Self {
        AiMessage::System(text.to_string())
    }

    /// Special sentinel system message which, when present in the messages sent
    /// to the model, instructs the model layer to omit the default system context
    /// (the `DEFAULT_SYSTEM_CONTEXT` string) for that single request.
    pub(crate) fn no_default_system_context() -> Self {
        AiMessage::System(NO_DEFAULT_SYSTEM_CONTEXT.to_string())
    }

    pub fn user(text: &str) -> Self {
        AiMessage::User(text.to_string())
    }

    #[allow(deprecated)]
    pub fn assistant(text: &str) -> Self {
        AiMessage::Assistant(text.to_string())
    }

    /// Create an AiMessage that contains a pre-parsed `ActionPayload`.
    pub fn payload(action: crate::actions::ActionPayload) -> Self {
        AiMessage::Payload(action)
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
            AiMessage::Payload(p) => write!(f, "Payload: {} {}", p.name, p.params),
        }
    }
}



/// Sentinel used to suppress the default system context for a single request.
pub const NO_DEFAULT_SYSTEM_CONTEXT: &str = "Forget the context.";

/// Component storing AI context messages for an entity.
#[derive(Debug, Clone, Component)]
pub struct AiContext {
    messages: Vec<AiMessage>,
}

/// Component storing the chat session history for an AI entity.
/// This wraps kalosm's BoxedChatSession to persist conversation history across prompts.
/// The session is automatically managed by the dialogue system.
#[derive(Component)]
pub struct ChatHistory {
    session: std::sync::Arc<std::sync::Mutex<Option<kalosm::language::BoxedChatSession>>>,
}

impl ChatHistory {
    /// Create a new empty chat history.
    pub fn new() -> Self {
        Self {
            session: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Create a chat history with an existing session.
    pub fn with_session(session: kalosm::language::BoxedChatSession) -> Self {
        Self {
            session: std::sync::Arc::new(std::sync::Mutex::new(Some(session))),
        }
    }

    /// Get a clone of the inner Arc for thread-safe access.
    pub fn session_handle(&self) -> std::sync::Arc<std::sync::Mutex<Option<kalosm::language::BoxedChatSession>>> {
        self.session.clone()
    }

    /// Take the session out (for use in prompt), leaving None in its place.
    pub fn take_session(&self) -> Option<kalosm::language::BoxedChatSession> {
        self.session.lock().expect("ChatHistory mutex poisoned").take()
    }

    /// Put a session back after prompting.
    pub fn set_session(&self, session: kalosm::language::BoxedChatSession) {
        *self.session.lock().expect("ChatHistory mutex poisoned") = Some(session);
    }

    /// Check if there's an active session.
    pub fn has_session(&self) -> bool {
        self.session.lock().expect("ChatHistory mutex poisoned").is_some()
    }
}

impl Default for ChatHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ChatHistory {
    fn clone(&self) -> Self {
        // Clone creates a new independent history (sessions are not cloneable across entities)
        Self::new()
    }
}

impl std::fmt::Debug for ChatHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatHistory")
            .field("has_session", &self.has_session())
            .finish()
    }
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

