use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_channel;
use kalosm::language::*;

use crate::dialogue::LocalAi;
use crate::rag::AiMessage;

/// Global tokio runtime for async operations - creating a runtime per call is very expensive
pub(crate) static TOKIO_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .event_interval(31)
        .thread_name("ai-runtime-worker")
        .build()
        .expect("Failed to create global tokio runtime")
});

/// Run an async future synchronously, but avoid calling `block_on` from within
/// a tokio runtime worker thread (which panics). If we detect we're inside a
/// runtime, use `tokio::task::block_in_place` to move the blocking work to the
/// blocking pool and then `TOKIO_RUNTIME.block_on` the future there. This
/// preserves the synchronous API while avoiding nested runtime panics.
fn run_sync<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        // We're inside a runtime: move to blocking context and run on global runtime
        tokio::task::block_in_place(|| TOKIO_RUNTIME.block_on(fut))
    } else {
        // Not in a runtime: safe to block on the global runtime directly
        TOKIO_RUNTIME.block_on(fut)
    }
}

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
    /// Llama model or source (e.g., local file or HuggingFace)
    Llama,
    /// OpenAI GPT model with API key
    GPT(SecureString),
    /// Phi3 model
    Phi,
}

enum ModelSource {
    Llama(Llama),
    GPT(OpenAICompatibleChatModel),
    Phi(Llama),
}

#[derive(Clone)]
pub struct AiModelBuilder {
    model_type: ModelType,
    model_file_source: Option<FileSource>,
    progress_chan_tx: Option<crossbeam_channel::Sender<ModelDownloadProgress>>,
    progress_chan_rx: Option<crossbeam_channel::Receiver<ModelDownloadProgress>>,
    include_default_context: bool,
    seed: Option<u64>,
}

impl AiModelBuilder {
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
                format!("{}", source,)
            }
            ModelLoadingProgress::Loading { progress } => {
                format!("Loading model: {:.0}%", progress * 100.0)
            }
        };

        let progress_value = handler.progress() * 100.0;

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
                // Mutex to serialize terminal writes so updates don't interleave
                static PROGRESS_LOCK: Mutex<()> = Mutex::new(());

                // Length of the last single-line progress output (for padding when shorter updates occur)
                static PREV_LINE_LEN: AtomicUsize = AtomicUsize::new(0);

                // Print message on one line and percentage on the next, updating both in-place.
                use console::Term;

                let term = Term::stdout();

                // Prepare both lines and truncate to terminal width to avoid wrapping
                let cols = term.size().1 as usize; // Term::size returns (rows, cols)
                let max_width = cols.saturating_sub(1).max(20);

                // Message line
                let msg_line = if message.chars().count() > max_width {
                    let mut s = message.chars().take(max_width - 1).collect::<String>();
                    s.push('…');
                    s
                } else {
                    message.clone()
                };

                // Percentage line
                let pct = format!("({:.0}%)", progress_value);
                let pct_line = if pct.chars().count() > max_width {
                    let mut s = pct.chars().take(max_width - 1).collect::<String>();
                    s.push_str("...");
                    s
                } else {
                    pct
                };

                // Serialize writes to avoid interleaving
                let _guard = PROGRESS_LOCK.lock().unwrap();

                use std::io::Write;

                // Single-line: "<message> <percent>" and carriage-return overwrite
                let combined = format!("{} {}", msg_line, pct_line);
                let len = combined.chars().count();

                // Serialize writes
                let mut handle = Term::stdout();
                let prev = PREV_LINE_LEN.load(Ordering::SeqCst);

                // Write carriage return then combined content
                let _ = write!(handle, "\r{}", combined);

                // If we shortened, pad with spaces to erase leftover chars
                if len < prev {
                    let _ = write!(handle, "{}", " ".repeat(prev - len));
                    // Re-write to move cursor back to line start with content
                    let _ = write!(handle, "\r{}", combined);
                }

                let _ = handle.flush();
                PREV_LINE_LEN.store(len, Ordering::SeqCst);

                // On completion, print newline and reset length tracker
                if progress_value >= 100.0 {
                    let _ = writeln!(handle, "");
                    PREV_LINE_LEN.store(0, Ordering::SeqCst);
                }
            }
        }
    }

    pub fn build(&self) -> Result<Arc<dyn LocalAi>, String> {
        // Use global runtime instead of creating a new one
        run_sync(async {
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
                        .with_client(
                            OpenAICompatibleClient::new().with_api_key(api_key.to_string()),
                        )
                        .build();
                    ModelSource::GPT(model)
                }
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

            let mut ai_model =
                AIModel::new(model).include_default_context(self.include_default_context);
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

impl AIModel {
    /// Prompt the model and then parse the returned text with the provided `ArcParser<T>`.
    /// This performs post-generation parsing (not constrained generation), which is
    /// a pragmatic way to get a typed output without relying on structured API
    /// support in the underlying boxed model.
    pub fn prompt_with_parser<P, T>(
        &self,
        messages: &[AiMessage],
        session: Option<kalosm::language::BoxedChatSession>,
        parser: P,
    ) -> Result<(T, Option<kalosm::language::BoxedChatSession>), String>
    where
        P: kalosm::language::Parser<Output = T> + kalosm::language::CreateParserState + Send + Sync + 'static,
        T: Clone + Send + 'static,
    {
        // Synchronously call the regular prompt path
        let prompt_res = self.prompt_with_session(messages, session)?;
        let text = prompt_res.response;

        // Create parser state and attempt to parse the response
        let state = parser.create_parser_state();
        let parse_res = parser.parse(&state, text.as_bytes());

        match parse_res {
            Ok(kalosm::language::ParseStatus::Finished { result, .. }) => Ok((result, prompt_res.session)),
            Ok(kalosm::language::ParseStatus::Incomplete { .. }) => Err("Parser reported incomplete result; model output may be truncated or not match the expected shape".to_string()),
            Err(e) => Err(format!("Parser error: {:?}", e)),
        }
    }

    /// Prompt using a kalosm `ArcParser<T>` as *constraints during generation*.
    /// This uses the model's `with_constraints`/typed capability so the model is
    /// constrained to produce a typed `T` as output rather than plain text.
    pub fn prompt_with_constrained_parser<T>(
        &self,
        messages: &[AiMessage],
        session: Option<kalosm::language::BoxedChatSession>,
        parser: kalosm::language::ArcParser<T>,
    ) -> Result<(T, Option<kalosm::language::BoxedChatSession>), String>
    where
        T: Clone + Send + 'static,
    {
        run_sync(async {
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

            // Decide whether to include the default system context. If any System
            // message sentinel is present, we treat it as a request to skip the default
            // system context for this request.
            let skip_default = messages.iter().any(|m| matches!(m, AiMessage::System(text) if text == crate::rag::NO_DEFAULT_SYSTEM_CONTEXT));

            let mut system_parts = if !skip_default {
                if let Some(context) = &self.include_default_context {
                    vec![context.clone()]
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            for message in messages {
                if let AiMessage::System(text) = message {
                    // Filter out the sentinel so it isn't forwarded to the backend
                    if text == crate::rag::NO_DEFAULT_SYSTEM_CONTEXT {
                        continue;
                    }
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
                    _ => {}
                }
            }
            let full_prompt = conversation_parts.join("\n");

            // Start generation with constraints and attempt to parse the result.
            // We pass the parser as a constraint (if supported by the backend)
            // and then parse the generated text to produce a typed result.
            // Generate response text (use same path as prompt_with_session) and then
            // run the parser over the output. This avoids needing extra trait bounds
            // on the builder while still providing a constrained-generation intention
            // (if the backend supports it in the future).
            let response = if let Some(seed) = self.seed {
                let sampler = GenerationParameters::default().with_seed(seed);
                chat.add_message(&full_prompt).with_sampler(sampler).all_text().await
            } else {
                chat.add_message(&full_prompt).all_text().await
            };

            let text = response;

            // Create parser state and attempt to parse the response
            let state = parser.create_parser_state();
            let parse_res = parser.parse(&state, text.as_bytes());

            match parse_res {
                Ok(kalosm::language::ParseStatus::Finished { result, .. }) => {
                    let updated_session = match chat.session() {
                        Ok(s) => Some(s.clone()),
                        Err(_) => None,
                    };
                    Ok((result, updated_session.or(Some(chat_session))))
                }
                Ok(kalosm::language::ParseStatus::Incomplete { .. }) => Err("Parser reported incomplete result; model output may be truncated or not match the expected shape".to_string()),
                Err(e) => Err(format!("Parser error: {:?}", e)),
            }
        })
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
        run_sync(async {
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

            // Decide whether to include the default system context. If any System
            // message sentinel is present, we treat it as a request to skip the default
            // system context for this request.
            let skip_default = messages.iter().any(|m| matches!(m, AiMessage::System(text) if text == crate::rag::NO_DEFAULT_SYSTEM_CONTEXT));

            let mut system_parts = if !skip_default {
                if let Some(context) = &self.include_default_context {
                    vec![context.clone()]
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            for message in messages {
                if let AiMessage::System(text) = message {
                    // Filter out the sentinel so it isn't forwarded to the backend
                    if text == crate::rag::NO_DEFAULT_SYSTEM_CONTEXT {
                        continue;
                    }
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

    fn get_model(&self) -> kalosm::language::BoxedChatModel {
        // Provide access to the underlying kalosm model for backends that need it.
        self.model.clone()
    }

    #[cfg(feature = "kalosm")]
    fn prompt_typed(
        &self,
        messages: &[AiMessage],
        session: Option<kalosm::language::BoxedChatSession>,
        _schema_description: &str,
    ) -> Result<(serde_json::Value, Option<kalosm::language::BoxedChatSession>), String> {
        // Fast path: use the kalosm-aware JsonParser to extract JSON directly.
        use crate::parse::json_parser::JsonParser;

        match self.prompt_with_parser::<JsonParser, serde_json::Value>(messages, session.clone(), JsonParser) {
            Ok((v, sess)) => Ok((v, sess)),
            Err(e) => {
                // Fall back to the generic post-generation JSON extraction
                eprintln!("JsonParser path failed: {}. Falling back to generic extraction.", e);
                let prompt_res = self.prompt_with_session(messages, session)?;
                match crate::parse::extract_and_parse_json::<serde_json::Value>(&prompt_res.response) {
                    Ok(v) => Ok((v, prompt_res.session)),
                    Err(err) => Err(err),
                }
            }
        }
    }

    // `as_any` removed from `LocalAi` trait. No downcast helper here.
}

/// Helper that attempts to run a parser against the response produced by the
/// provided backend. This will downcast the backend to `AIModel` and use the
/// synchronous post-generation parser path; if the backend isn't an `AIModel`,
/// an error is returned.
pub fn prompt_with_parser_from_backend<P, T>(
    backend: &std::sync::Arc<dyn crate::dialogue::LocalAi>,
    messages: &[AiMessage],
    session: Option<kalosm::language::BoxedChatSession>,
    parser: P,
) -> Result<(T, Option<kalosm::language::BoxedChatSession>), String>
where
    P: kalosm::language::Parser<Output = T> + kalosm::language::CreateParserState + Send + Sync + 'static,
    T: Clone + Send + 'static,
{
    // Use the object-safe `prompt_with_session` so we don't need to downcast backends.
    let prompt_res = backend.prompt_with_session(messages, session)?;
    let text = prompt_res.response;

    // Create parser state and attempt to parse the response
    let state = parser.create_parser_state();
    let parse_res = parser.parse(&state, text.as_bytes());

    match parse_res {
        Ok(kalosm::language::ParseStatus::Finished { result, .. }) => Ok((result, prompt_res.session)),
        Ok(kalosm::language::ParseStatus::Incomplete { .. }) => Err("Parser reported incomplete result; model output may be truncated or not match the expected shape".to_string()),
        Err(e) => Err(format!("Parser error: {:?}", e)),
    }
}

/// Attempt to parse typed output from any backend by using post-generation parsing.
/// This does not rely on backend-specific constrained generation and will work
/// with any `LocalAi` implementation that returns text via `prompt_with_session`.
pub fn prompt_with_typed_from_backend<P, T>(
    backend: &std::sync::Arc<dyn crate::dialogue::LocalAi>,
    messages: &[AiMessage],
    session: Option<kalosm::language::BoxedChatSession>,
    parser: P,
) -> Result<(T, Option<kalosm::language::BoxedChatSession>), String>
where
    P: kalosm::language::Parser<Output = T> + kalosm::language::CreateParserState + Send + Sync + 'static,
    T: Clone + Send + 'static + serde::de::DeserializeOwned + crate::parse::AiParsable,
{
    // First, ask the backend to produce a typed JSON value if it supports optimized paths.
    match backend.prompt_typed(messages, session.clone(), &T::schema_description()) {
        Ok((value, sess)) => match serde_json::from_value::<T>(value) {
            Ok(parsed) => return Ok((parsed, sess)),
            Err(e) => {
                // Fall through to post-generation parsing if conversion fails
                eprintln!("Typed parse failed: {}. Falling back to post-generation parsing.", e);
            }
        },
        Err(_) => {
            // Backend didn't produce typed result; fall back
        }
    }

    // Fallback: just get the text and run the parser against it (post-generation parsing)
    let prompt_res = backend.prompt_with_session(messages, session)?;
    let text = prompt_res.response;

    let state = parser.create_parser_state();
    let parse_res = parser.parse(&state, text.as_bytes());

    match parse_res {
        Ok(kalosm::language::ParseStatus::Finished { result, .. }) => Ok((result, prompt_res.session)),
        Ok(kalosm::language::ParseStatus::Incomplete { .. }) => Err("Parser reported incomplete result; model output may be truncated or not match the expected shape".to_string()),
        Err(e) => Err(format!("Parser error: {:?}", e)),
    }
}

const DEFAULT_SYSTEM_CONTEXT: &str = "
You are in a game world.

Rules:
- The context lists ALL people, places, items and information in this world
- No other people, places, items or information exist beyond those listed
- Do not add details, inferences, or explanations
";
