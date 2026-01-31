use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use crossbeam_channel;
use kalosm::language::*;

use crate::dialogue::LocalAi;
use crate::rag::AiMessage;

/// Global tokio runtime for async operations - creating a runtime per call is very expensive
static TOKIO_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .event_interval(15)
        .thread_name("ai-runtime-worker")
        .build()
        .expect("Failed to create global tokio runtime")
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

pub type SecureString = zeroize::Zeroizing<String>;

#[derive(Clone)]
pub enum ModelType {
    Llama,
    GPT(SecureString),
    Phi,
}

enum ModelSource {
    Llama(Llama),
    GPT(OpenAICompatibleChatModel),
    Phi(Llama),
}

#[derive(Clone)]
pub struct ModelBuilder {
    model_type: ModelType,
    model_file_source: Option<FileSource>,
    progress_chan_tx: Option<crossbeam_channel::Sender<ModelDownloadProgress>>,
    progress_chan_rx: Option<crossbeam_channel::Receiver<ModelDownloadProgress>>,
    include_default_context: bool,
    seed: Option<u64>,
}

impl ModelBuilder {

    /// Create a new ModelBuilder with the default model type (Llama)
    pub fn new() -> Self {
        Self::new_with(ModelType::Llama)
    }

    /// Create a new ModelBuilder with the specified model type
    pub fn new_with(model_type: ModelType) -> Self {
        Self {
            model_type,
            model_file_source: None,
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
        self.model_file_source = Some(FileSource::Local(path));
        self.model_type = ModelType::Llama; // Default to Llama for local files
        self
    }

    /// Specify a HuggingFace model as the source for the AI model.
    pub fn with_huggingface(mut self, repo_id: &str, revision: &str, file: &str) -> Self {
        self.model_file_source = Some(FileSource::HuggingFace {
            model_id: repo_id.to_string(),
            revision: revision.to_string(),
            file: file.to_string(),
        });
        self.model_type = ModelType::Llama; // Default to Llama for HuggingFace
        self
    }

    /// Enable progress tracking for model downloads.
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

    fn model_loading_handler(
        progress_chan_tx: Option<crossbeam_channel::Sender<ModelDownloadProgress>>,
        handler: ModelLoadingProgress,
    ) {
        let message = match handler.clone() {
            ModelLoadingProgress::Downloading { source, .. } => {
                format!(
                    "Downloading model from {}: {:.0}%",
                    source,
                    handler.progress() * 100.0
                )
            }
            ModelLoadingProgress::Loading { progress } => {
                format!("Loading model: {:.0}%", progress * 100.0)
            }
        };

        let progress_value = handler.progress();

        match progress_chan_tx {
            Some(tx) => {
                match tx.send(ModelDownloadProgress {
                    state: DownloadState::InProgress,
                    message,
                    progress: Some(progress_value),
                }) {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("Failed to send model loading progress: {}", e);
                    }
                }
            }
            None => {
                // Print progress to terminal, reusing the same line
                print!("\r{}", message);
                use std::io::{self, Write};
                let _ = io::stdout().flush();
            }
        }
    }

    pub fn build_chat(&self) -> Result<Arc<dyn LocalAi>, String> {
        // Use global runtime instead of creating a new one
        TOKIO_RUNTIME.block_on(async {
            let source = match self.model_type.clone() {
                ModelType::Llama => match &self.model_file_source {
                    Some(s) => {
                        let progress_tx = self.progress_chan_tx.clone();
                        let model = Llama::builder()
                            .with_source(LlamaSource::new(s.clone()))
                            .build_with_loading_handler(move |handler| {
                                Self::model_loading_handler(progress_tx.clone(), handler.clone());
                            })
                            .await
                            .map_err(|e| format!("Failed to create Llama model source: {}", e))?;
                        ModelSource::Llama(model)
                    }
                    None => {
                        let progress_tx = self.progress_chan_tx.clone();
                        let model = Llama::builder()
                            .with_source(LlamaSource::llama_3_2_3b_chat())
                            .build_with_loading_handler(move |handler| {
                                Self::model_loading_handler(progress_tx.clone(), handler.clone());
                            })
                            .await
                            .map_err(|e| format!("Failed to create Llama model source: {}", e))?;
                        ModelSource::Llama(model)
                    }
                },
                ModelType::GPT(api_key) => {
                    let model = OpenAICompatibleChatModelBuilder::new()
                        .with_gpt_4o_mini()
                        .with_client(OpenAICompatibleClient::new().with_api_key(api_key.as_str()))
                        .build();
                    ModelSource::GPT(model)
                },
                ModelType::Phi => match &self.model_file_source {
                    Some(s) => {
                        let progress_tx = self.progress_chan_tx.clone();
                        let model = Llama::builder()
                            .with_source(LlamaSource::new(s.clone()))
                            .build_with_loading_handler(move |handler| {
                                Self::model_loading_handler(progress_tx.clone(), handler.clone());
                            })
                            .await
                            .map_err(|e| format!("Failed to create Phi model source: {}", e))?;
                        ModelSource::Phi(model)
                    }
                    None => {
                        let progress_tx = self.progress_chan_tx.clone();
                        let model = Llama::builder()
                            .with_source(LlamaSource::phi_3_1_mini_4k_instruct())
                            .build_with_loading_handler(move |handler| {
                                Self::model_loading_handler(progress_tx.clone(), handler.clone());
                            })
                            .await
                            .map_err(|e| format!("Failed to create Phi model source: {}", e))?;
                        ModelSource::Phi(model)
                    }
                },
            };

            let model = match source {
                ModelSource::Llama(m) => m.boxed_chat_model(),
                ModelSource::GPT(m) => m.boxed_chat_model(),
                ModelSource::Phi(m) => m.boxed_chat_model(),
            };

            let mut ai_model = AIModel::new(model)
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
    include_default_context: Option<String>,
    seed: Option<u64>,
}

impl AIModel {
    pub fn new(model: kalosm::language::BoxedChatModel) -> Self {
        Self {
            model: model,
            session: None,
            include_default_context: Some(DEFAULT_SYSTEM_CONTEXT.trim().to_string()),
            seed: None,
        }
    }

    pub fn with_session(mut self, session: kalosm::language::BoxedChatSession) -> Self {
        self.session = Some(session);
        self
    }

    pub fn include_default_context(mut self, include: bool) -> Self {
        if include {
            self.include_default_context = Some(DEFAULT_SYSTEM_CONTEXT.trim().to_string());
        } else {
            self.include_default_context = None;
        };
        self
    }

    pub fn with_default_context(mut self, context: &str) -> Self {
        self.include_default_context = Some(context.to_string());
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
            let mut system_parts = if let Some(context) = &self.include_default_context {
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
                chat.add_message(&full_prompt)
                    .with_sampler(sampler)
                    .all_text()
                    .await
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
                session: updated_session.or(Some(chat_session)),
            })
        })
    }
}

const DEFAULT_SYSTEM_CONTEXT: &str = "
You are in a game world.

Rules:
- The context lists ALL people, places, items and information in this world
- No other people, places, items or information exist beyond those listed
- Do not add details, inferences, or explanations
";
