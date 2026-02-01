//! Bevy App extension traits for easier AI integration.
//!
//! Provides convenient methods for setting up AI in your Bevy app without
//! needing to manually configure plugins and resources.
//!
//! # Example
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_real_ai::prelude::*;
//!
//! App::new()
//!     .add_plugins(DefaultPlugins)
//!     .use_ai(ModelType::Llama)
//!     .register_ai_action::<SpawnAction, _, _>(|In(action): In<SpawnAction>, mut commands: Commands| {
//!         // Handle spawn action
//!     })
//!     .run();
//! ```

use bevy::prelude::*;
use crate::actions::{AiActionRegistry, IntoActionPayload};
use crate::dialogue::AIDialoguePlugin;
use crate::models::{AiModelBuilder, ModelType};

/// Extension trait for `App` that provides convenient AI setup methods.
pub trait AiAppExt {
    /// Initialize AI with the specified model type.
    ///
    /// This is a convenience method that adds the `AIDialoguePlugin` with an
    /// `AiModelBuilder` configured for the given model type. The model will be
    /// loaded asynchronously in the background.
    ///
    /// # Example
    /// ```ignore
    /// App::new()
    ///     .add_plugins(DefaultPlugins)
    ///     .use_ai(ModelType::Llama)
    ///     .run();
    /// ```
    fn use_ai(&mut self, model_type: ModelType) -> &mut Self;

    /// Initialize AI with a custom model builder.
    ///
    /// Use this for more control over model configuration (seed, progress tracking, etc.)
    ///
    /// # Example
    /// ```ignore
    /// App::new()
    ///     .add_plugins(DefaultPlugins)
    ///     .use_ai_with_builder(
    ///         AiModelBuilder::new_with(ModelType::Llama)
    ///             .with_seed(42)
    ///             .with_progress_tracking()
    ///     )
    ///     .run();
    /// ```
    fn use_ai_with_builder(&mut self, builder: AiModelBuilder) -> &mut Self;

    /// Register a typed AI action handler.
    ///
    /// The action name is automatically derived from the type (via `IntoActionPayload::action_name()`).
    /// The handler receives the deserialized action as `In<T>` plus any other system parameters.
    ///
    /// # Example
    /// ```ignore
    /// #[derive(Clone, Debug, Serialize, Deserialize, AiAction)]
    /// struct SpawnAction { name: String, x: f32, y: f32 }
    ///
    /// app.register_ai_action::<SpawnAction, _, _>(|In(action): In<SpawnAction>, mut commands: Commands| {
    ///     commands.spawn(/* ... */);
    /// });
    /// ```
    fn register_ai_action<T, S, M>(&mut self, system: S) -> &mut Self
    where
        T: 'static + Send + Sync + serde::de::DeserializeOwned + IntoActionPayload,
        S: bevy::ecs::system::IntoSystem<In<T>, (), M> + 'static;

    /// Register a raw AI action handler by name.
    ///
    /// The handler receives the full `AiActionEvent` as `In<AiActionEvent>`.
    ///
    /// # Example
    /// ```ignore
    /// app.register_ai_action_raw("custom_action", |In(event): In<AiActionEvent>, mut commands: Commands| {
    ///     // Handle custom action
    /// });
    /// ```
    fn register_ai_action_raw<S, M>(&mut self, name: &str, system: S) -> &mut Self
    where
        S: bevy::ecs::system::IntoSystem<In<crate::actions::AiActionEvent>, (), M> + 'static;
}

impl AiAppExt for App {
    fn use_ai(&mut self, model_type: ModelType) -> &mut Self {
        let builder = AiModelBuilder::new_with(model_type).with_progress_tracking();
        self.add_plugins(AIDialoguePlugin::with_builder(builder));
        self
    }

    fn use_ai_with_builder(&mut self, builder: AiModelBuilder) -> &mut Self {
        self.add_plugins(AIDialoguePlugin::with_builder(builder));
        self
    }

    fn register_ai_action<T, S, M>(&mut self, system: S) -> &mut Self
    where
        T: 'static + Send + Sync + serde::de::DeserializeOwned + IntoActionPayload,
        S: bevy::ecs::system::IntoSystem<In<T>, (), M> + 'static,
    {
        let action_name = T::action_name();
        
        // Ensure registry exists (it should if AIDialoguePlugin was added)
        self.world_mut().get_resource_or_init::<AiActionRegistry>()
            .register_typed::<T, S, M>(action_name, system);
        
        self
    }

    fn register_ai_action_raw<S, M>(&mut self, name: &str, system: S) -> &mut Self
    where
        S: bevy::ecs::system::IntoSystem<In<crate::actions::AiActionEvent>, (), M> + 'static,
    {
        // Ensure registry exists
        if self.world().get_resource::<AiActionRegistry>().is_none() {
            self.insert_resource(AiActionRegistry::default());
        }
        
        self.world_mut()
            .resource_mut::<AiActionRegistry>()
            .register(name, system);
        
        self
    }
}
