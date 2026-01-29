use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use crossbeam_channel;
use kalosm::language::*;

use crate::dialogue::LocalAi;
use crate::rag::AiMessage;

/// Global tokio runtime for async operations - creating a runtime per call is very expensive
static TOKIO_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Runtime::new().expect("Failed to create global tokio runtime")
});

/// Represents the state of a model download operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadState {
    InProgress,
    Completed,
    Error,
}

/// Progress information for model downloads sent via crossbeam-channel
#[derive(Debug, Clone)]
pub struct ModelDownloadProgress {
    pub state: DownloadState,
    pub message: String,
    pub progress: Option<f32>, // 0.0 to 1.0 for InProgress, None otherwise
}

#[derive(Clone)]
pub struct ModelBuilder {
    file_source: Option<FileSource>,
    progress_chan_tx: Option<crossbeam_channel::Sender<ModelDownloadProgress>>,
    progress_chan_rx: Option<crossbeam_channel::Receiver<ModelDownloadProgress>>,
    include_default_context: bool,
    seed: Option<u64>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            file_source: None,
            progress_chan_tx: None,
            progress_chan_rx: None,
            include_default_context: true,
            seed: None,
        }
    }

    pub fn include_default_context(mut self, include: bool) -> Self {
        self.include_default_context = include;
        self
    }

    /// Set a seed for deterministic generation.
    /// When set, the model will produce the same output for the same input.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Specify a local file path as the source for the AI model. Recommend .gguf files.
    pub fn with_local(mut self, path: PathBuf) -> Self {
        self.file_source = Some(FileSource::Local(path));
        self
    }

    /// Specify a HuggingFace model as the source for the AI model.
    pub fn with_huggingface(mut self, repo_id: &str, revision: &str, file: &str) -> Self {
        self.file_source = Some(FileSource::HuggingFace {
            model_id: repo_id.to_string(),
            revision: revision.to_string(),
            file: file.to_string(),
        });
        self
    }

    /// Enable progress tracking for model downloads. Returns self for chaining.
    ///
    /// If not called, progress updates will be viewed on the terminal only.
    pub fn with_progress(mut self) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded::<ModelDownloadProgress>();
        self.progress_chan_tx = Some(tx);
        self.progress_chan_rx = Some(rx);
        self
    }

    /// Extract the progress receiver after enabling with_progress().
    /// Returns None if with_progress() was not called.
    pub fn take_progress_receiver(
        &mut self,
    ) -> Option<crossbeam_channel::Receiver<ModelDownloadProgress>> {
        self.progress_chan_rx.take()
    }

    pub fn build_chat(&self) -> Result<Arc<dyn LocalAi>, String> {
        // Use global runtime instead of creating a new one
        TOKIO_RUNTIME.block_on(async {
            let source = match self.file_source.clone() {
                Some(path) => LlamaSource::new(path),
                // Use Llama 3.2 1B as default - fast on CPU, good instruction following, less restrictive
                None => LlamaSource::llama_3_2_1b_chat(),
            };

            let builder = Llama::builder().with_source(source).with_flash_attn(true);

            // If a progress channel is provided, use build_with_loading_handler
            if let Some(tx) = &self.progress_chan_tx {
                let closure_tx = tx.clone();
                match builder
                    .build_with_loading_handler(move |handler| {
                        match closure_tx.send(ModelDownloadProgress {
                            state: if handler.progress() >= 100.0 {
                                DownloadState::Completed
                            } else {
                                DownloadState::InProgress
                            },
                            message: format!("{:.0}", handler.progress()),
                            progress: Some(handler.progress()),
                        }) {
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!("Failed to send model download progress: {}", e);
                            }
                        }
                    })
                    .await
                {
                    Ok(m) => {
                        let mut model = AIModel::new(m.boxed_chat_model())
                            .include_default_context(self.include_default_context);
                        if let Some(seed) = self.seed {
                            model = model.with_seed(seed);
                        }
                        let arc_model: Arc<dyn LocalAi> = Arc::new(model);
                        return Ok(arc_model);
                    }
                    Err(e) => {
                        if let Some(tx) = &self.progress_chan_tx {
                            let _ = tx.send(ModelDownloadProgress {
                                state: DownloadState::Error,
                                message: format!("Failed to load Llama model: {}", e),
                                progress: None,
                            });
                        }
                        return Err(format!("Failed to load Llama model: {}", e));
                    }
                }
            }

            let model = builder.build().await.map_err(|e| {
                if let Some(tx) = &self.progress_chan_tx {
                    let _ = tx.send(ModelDownloadProgress {
                        state: DownloadState::Error,
                        message: format!("Failed to load AI model: {}", e),
                        progress: None,
                    });
                }
                format!("Failed to load AI model: {}", e)
            })?;

            if let Some(tx) = &self.progress_chan_tx {
                let _ = tx.send(ModelDownloadProgress {
                    state: DownloadState::Completed,
                    message: "Model loaded successfully".to_string(),
                    progress: Some(1.0),
                });
            }

            let mut ai_model = AIModel::new(model.boxed_chat_model())
                .include_default_context(self.include_default_context);
            if let Some(seed) = self.seed {
                ai_model = ai_model.with_seed(seed);
            }
            let arc_model: Arc<dyn LocalAi> = Arc::new(ai_model);
            Ok(arc_model)
        })
    }
}

#[derive(Clone)]
pub struct AIModel {
    model: kalosm::language::BoxedChatModel,
    session: Option<kalosm::language::BoxedChatSession>,
    include_default_prompt: Option<String>,
    seed: Option<u64>,
}

impl AIModel {
    pub fn new(model: kalosm::language::BoxedChatModel) -> Self {
        Self {
            model: model,
            session: None,
            include_default_prompt: Some(DEFAULT_SYSTEM_PROMPT.to_string()),
            seed: None,
        }
    }

    pub fn with_session(mut self, session: kalosm::language::BoxedChatSession) -> Self {
        self.session = Some(session);
        self
    }

    pub fn include_default_context(mut self, include: bool) -> Self {
        if include {
            self.include_default_prompt = Some(DEFAULT_SYSTEM_PROMPT.to_string());
        } else {
            self.include_default_prompt = None;
        };
        self
    }

    /// Set a seed for deterministic generation.
    /// When set, the model will produce the same output for the same input.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl LocalAi for AIModel {
    fn prompt(&self, messages: &[AiMessage]) -> Result<String, String> {
        // Delegate to prompt_with_session without an existing session
        self.prompt_with_session(messages, None).map(|r| r.response)
    }

    fn prompt_with_session(
        &self,
        messages: &[AiMessage],
        session: Option<kalosm::language::BoxedChatSession>,
    ) -> Result<crate::dialogue::PromptResult, String> {
        // Use global runtime instead of creating a new one each call
        TOKIO_RUNTIME.block_on(async {
            let chat_session = match session {
                Some(s) => s,
                None => match &self.session {
                    Some(session) => session.clone(),
                    None => match self.model.new_chat_session() {
                        Ok(s) => s,
                        Err(e) => return Err(format!("Failed to create chat session: {}", e)),
                    },
                },
            };
            let mut chat = self.model.chat().with_session(chat_session.clone());

            // Combine system prompt with context information
            let mut system_parts = if let Some(context) = &self.include_default_prompt {
                vec![context.clone()]
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

            // Build user prompt from User messages only (history is in the session)
            let mut conversation_parts = Vec::new();
            for message in messages {
                match message {
                    AiMessage::User(text) => {
                        conversation_parts.push(format!("{}", text));
                    }
                    AiMessage::System(_) => {
                        // Already handled above
                    }
                    _ => {
                        // Ignore Assistant messages in prompt
                    }
                }
            }

            // Combine all conversation parts
            let full_prompt = conversation_parts.join("\n");

            // Generate response with optional seed for deterministic output
            let response = if let Some(seed) = self.seed {
                let sampler = GenerationParameters::default().with_seed(seed);
                chat.add_message(&full_prompt).with_sampler(sampler).all_text().await
            } else {
                chat.add_message(&full_prompt).all_text().await
            };
            
            let updated_session = match chat.session() {
                Ok(s) => Some(s.clone()),
                Err(_) => None,
            };

            if let None = updated_session {
                eprintln!("Warning: Failed to retrieve updated chat session after prompt.");
            }
            Ok(crate::dialogue::PromptResult {
                response,
                session: updated_session.unwrap_or(chat_session),
            })
        })
    }
}

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are an NPC in a game. Answer ONLY using the facts given below.

RULES:
- Use ONLY the information provided in the context
- If the answer is not in the context, say: I don't know
- Keep answers very short (1-2 sentences max)
- Never make up information
- Never explain or add details";
