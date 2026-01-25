use std::path::PathBuf;

use kalosm::language::*;

use crate::{LocalAi, rag::AiMessage};

#[derive(Clone, Copy)]
pub enum Type {
    Llama,
}

impl Type {
    pub fn new(&self) -> ModelBuilder {
        ModelBuilder::new(*self)
    }
}

#[derive(Clone)]
pub struct ModelBuilder {
    model_type: Type,
    file_source: Option<FileSource>,
}

impl ModelBuilder {
    pub fn new(model_type: Type) -> Self {
        Self {
            model_type,
            file_source: None,
        }
    }

    pub fn with_local(mut self, path: PathBuf) -> Self {
        self.file_source = Some(FileSource::Local(path));
        self
    }

    pub fn with_huggingface(mut self, repo_id: String, revision: String, file: String) -> Self {
        self.file_source = Some(FileSource::HuggingFace{ model_id: repo_id, revision: revision, file: file });
        self
    }

    pub fn build(&self) -> Result<Box<dyn LocalAi + 'static>, String> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;
        rt.block_on(async {
            match self.model_type {
                Type::Llama => {
                    let source = match self.file_source.clone() {
                        Some(path) => LlamaSource::new(path),
                        None => LlamaSource::qwen_2_5_1_5b_instruct(),
                    };
                    let model = Llama::builder()
                        .with_source(LlamaSource::llama_3_2_1b_chat())
                        .build()
                        .await
                        .map_err(|e| format!("Failed to load Llama model: {}", e))?;
                    Ok(
                        Box::new(
                            AIModel::new(model.boxed_chat_model()).include_default_prompt(true),
                        ) as Box<dyn LocalAi>,
                    )
                }
            }
        })
    }
}

#[derive(Clone)]
pub struct AIModel {
    model: kalosm::language::BoxedChatModel,
    session: Option<kalosm::language::BoxedChatSession>,
    include_default_prompt: bool,
}

impl AIModel {
    pub fn new(model: kalosm::language::BoxedChatModel) -> Self {
        Self {
            model: model,
            session: None,
            include_default_prompt: true,
        }
    }

    pub fn with_session(mut self, session: kalosm::language::BoxedChatSession) -> Self {
        self.session = Some(session);
        self
    }

    pub fn include_default_prompt(mut self, include: bool) -> Self {
        self.include_default_prompt = include;
        self
    }
}


impl LocalAi for AIModel {
    #[allow(deprecated)]
    fn prompt(&self, messages: &[AiMessage]) -> Result<String, String> {
        // Create a tokio runtime to run the async kalosm code
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        rt.block_on(async {
            let chat_session = match &self.session {
                Some(session) => session.clone(),
                None => match self.model.new_chat_session() {
                    Ok(s) => s,
                    Err(e) => return Err(format!("Failed to create chat session: {}", e)),
                },
            };
            let mut chat = self.model.chat().with_session(chat_session);

            // Combine system prompt with context information
            let mut system_parts = if self.include_default_prompt {
                vec![DEFAULT_SYSTEM_PROMPT.to_string()]
            } else {
                Vec::new()
            };
            for message in messages {
                if let AiMessage::System(text) = message {
                    system_parts.push(text.clone());
                }
            }
            let combined_system_prompt = system_parts.join("\n\n");
            chat = chat.with_system_prompt(&combined_system_prompt);

            // Build user/assistant conversation history
            let mut conversation_parts = Vec::new();
            for message in messages {
                match message {
                    AiMessage::User(text) => {
                        conversation_parts.push(format!("User: {}", text));
                    }
                    AiMessage::Assistant(_) => {
                        // Assistant messages are deprecated and not used in prompt construction
                    }
                    AiMessage::System(_) => {
                        // Already handled above
                    }
                }
            }

            // Combine all conversation parts
            let full_prompt = conversation_parts.join("\n");

            // Generate response from the combined conversation
            let response = chat.add_message(&full_prompt).all_text().await;
            Ok(response)
        })
    }
}

const DEFAULT_SYSTEM_PROMPT: &str = "You are an NPC.

Answer questions using only these facts. Do not invent anything. Do not explain. Do not add details.

Keep responses short and factual.";
