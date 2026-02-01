use serde_json::Value;
use bevy::prelude::*;
use std::collections::HashMap;

/// A generic action produced by the AI. `name` is the action identifier, and
/// `params` contains arbitrary JSON parameters for the action.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionPayload {
    pub name: String,
    pub params: Value,
}

impl ActionPayload {
    pub fn new(name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            params: Value::Null,
        }
    }

    pub fn with_param(mut self, key: impl ToString, value: Value) -> Self {
        if let Value::Object(ref mut map) = self.params {
            map.insert(key.to_string(), value);
        } else {
            let mut map = serde_json::Map::new();
            map.insert(key.to_string(), value);
            self.params = Value::Object(map);
        }
        self
    }

    pub fn get_raw(&self, key: &str) -> Option<&Value> {
        match &self.params {
            Value::Object(map) => map.get(key),
            Value::Null => None,
            _ => None,
        }
    }

    pub fn get<T>(&self, key: &str) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.get_raw(key).and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}

/// Trait for typed structs that can be converted into an `ActionPayload`.
pub trait IntoActionPayload {
    /// Returns the action name used to match registered handlers.
    fn action_name() -> &'static str;

    /// Convert the typed struct into an `ActionPayload`.
    fn into_action_payload(self) -> ActionPayload;
}

/// Event emitted when an AI response contains an action for an entity to handle.
#[derive(Event, Clone, Debug)]
pub struct AiActionEvent {
    pub entity: Entity,
    pub action: ActionPayload,
}

pub(crate) fn value_to_action(v: Value) -> Option<ActionPayload> {
    if let Value::Object(map) = v {
        if let Some(Value::String(name)) = map.get("name") {
            let params = map.get("params").cloned().unwrap_or(Value::Null);
            return Some(ActionPayload { name: name.clone(), params });
        }
    }
    None
}

/// A boxed, type-erased handler that can be fed an `AiActionEvent` and then run.
/// This trait allows handlers to receive action data directly without needing
/// a temporary resource.
pub trait AiActionHandlerDyn: Send + Sync {
    /// Run the handler with the given action event.
    fn run_with_action(&mut self, event: AiActionEvent, world: &mut World);
}

/// Boxed handler type for the registry.
pub type AiActionHandler = Box<dyn AiActionHandlerDyn>;

/// Pending actions that have been parsed and await processing by registered handlers.
#[derive(Resource, Default)]
pub struct PendingAiActions {
    pub actions: Vec<AiActionEvent>,
}

/// Registry mapping action names to boxed handlers.
#[derive(Resource, Default)]
pub struct AiActionRegistry {
    handlers: HashMap<String, AiActionHandler>,
}

impl AiActionRegistry {
    pub fn new() -> Self {
        Self { handlers: HashMap::new() }
    }

    /// Register a handler that receives the full `AiActionEvent` as input.
    ///
    /// The handler function receives `In<AiActionEvent>` plus any other system parameters.
    ///
    /// # Example
    /// ```ignore
    /// registry.register("my_action", |In(event): In<AiActionEvent>, mut commands: Commands| {
    ///     // Handle the action
    /// });
    /// ```
    pub fn register<S, M>(&mut self, name: &str, system: S)
    where
        S: bevy::ecs::system::IntoSystem<In<AiActionEvent>, (), M> + 'static,
    {
        let inner_system = bevy::ecs::system::IntoSystem::into_system(system);
        let name_owned = name.to_string();
        
        // Create a wrapper that implements AiActionHandlerDyn
        struct SystemWrapper<Sys> {
            system: Sys,
            initialized: bool,
        }
        
        impl<Sys> AiActionHandlerDyn for SystemWrapper<Sys>
        where
            Sys: bevy::ecs::system::System<In = In<AiActionEvent>, Out = ()> + Send + Sync,
        {
            fn run_with_action(&mut self, event: AiActionEvent, world: &mut World) {
                if !self.initialized {
                    let _ = self.system.initialize(world);
                    self.initialized = true;
                }
                let _ = self.system.run(event, world);
                self.system.apply_deferred(world);
            }
        }
        
        self.handlers.insert(name_owned, Box::new(SystemWrapper {
            system: inner_system,
            initialized: false,
        }));
    }

    /// Register a typed handler system for an action name.
    ///
    /// The provided `system` must be convertible to a Bevy `System` that accepts
    /// `In<T>` input where `T` is deserializable from the action's params.
    /// The handler receives the deserialized typed struct directly.
    ///
    /// # Example
    /// ```ignore
    /// #[derive(Deserialize)]
    /// struct SpawnAction { name: String, x: f32, y: f32 }
    ///
    /// registry.register_typed::<SpawnAction, _, _>("spawn_action", |In(action): In<SpawnAction>, mut commands: Commands| {
    ///     commands.spawn(/* ... */);
    /// });
    /// ```
    pub fn register_typed<T, S, M>(&mut self, name: &str, system: S)
    where
        T: 'static + Send + Sync + serde::de::DeserializeOwned,
        S: bevy::ecs::system::IntoSystem<In<T>, (), M> + 'static,
    {
        let inner_system = bevy::ecs::system::IntoSystem::into_system(system);
        let name_owned = name.to_string();
        let name_for_error = name.to_string();
        
        // Create a wrapper that deserializes T and runs the inner system
        struct TypedSystemWrapper<T, Sys> {
            system: Sys,
            initialized: bool,
            name: String,
            _marker: std::marker::PhantomData<T>,
        }
        
        impl<T, Sys> AiActionHandlerDyn for TypedSystemWrapper<T, Sys>
        where
            T: 'static + Send + Sync + serde::de::DeserializeOwned,
            Sys: bevy::ecs::system::System<In = In<T>, Out = ()> + Send + Sync,
        {
            fn run_with_action(&mut self, event: AiActionEvent, world: &mut World) {
                match serde_json::from_value::<T>(event.action.params.clone()) {
                    Ok(typed) => {
                        if !self.initialized {
                            let _ = self.system.initialize(world);
                            self.initialized = true;
                        }
                        let _ = self.system.run(typed, world);
                        self.system.apply_deferred(world);
                    }
                    Err(e) => {
                        error!("Failed to deserialize typed action for {}: {}", self.name, e);
                    }
                }
            }
        }
        
        self.handlers.insert(name_owned, Box::new(TypedSystemWrapper {
            system: inner_system,
            initialized: false,
            name: name_for_error,
            _marker: std::marker::PhantomData::<T>,
        }));
    }

    /// Get a mutable reference to a handler by name, if any.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut AiActionHandler> {
        self.handlers.get_mut(name)
    }
}

/// World-exclusive runner that executes handler systems for pending actions.
/// This should be scheduled as an exclusive system (`fn(&mut World)`) each frame.
pub fn run_registered_actions_world(world: &mut World) {
    // Drain pending actions resource
    let pending = match world.get_resource_mut::<PendingAiActions>() {
        Some(mut p) => std::mem::take(&mut p.actions),
        None => Vec::new(),
    };

    if pending.is_empty() {
        return;
    }

    // For each action event, run any registered handler
    for evt in pending.into_iter() {
        world.resource_scope::<AiActionRegistry, _>(|world, mut registry| {
            if let Some(handler) = registry.get_mut(&evt.action.name) {
                debug!("Executing handler '{}' for entity {:?}", evt.action.name, evt.entity);
                handler.run_with_action(evt, world);
            }
        });
    }
}

/// Prompt the AI and parse the response using our custom `AiParsable` trait.
/// This version uses our own derive macro instead of kalosm's Parse/Schema.
///
/// # Arguments
/// * `backend` - The AI backend
/// * `user_message` - The user's request (will be formatted with schema instructions)
/// * `entity` - The entity that will receive the action event
/// * `pending` - The pending actions queue to add the action to
///
/// # Example
/// ```ignore
/// use bevy_real_ai::actions::prompt_typed_action;
/// use bevy_real_ai::AiAction;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Clone, Debug, Serialize, Deserialize, AiAction)]
/// struct SpawnAction {
///     pub name: String,
///     pub x: i32,
///     pub y: i32,
/// }
///
/// // Then use:
/// let result = prompt_typed_action::<SpawnAction>(
///     &backend,
///     "Create an entity named 'player' at position 5, 10",
///     entity,
///     &mut pending,
/// );
/// ```
pub fn prompt_typed_action<T>(
    backend: &std::sync::Arc<dyn crate::dialogue::LocalAi>,
    user_message: &str,
    entity: Entity,
    pending: &mut PendingAiActions,
) -> Result<(T, String), String>
where
    T: crate::parse::AiParsable + serde::de::DeserializeOwned,
{
    // Build the prompt with schema instructions
    let formatted_prompt = crate::parse::build_typed_prompt::<T>(user_message);
    let messages = vec![crate::rag::AiMessage::user(&formatted_prompt)];

    // Get response from AI
    let response = backend.prompt(&messages)?;

    // Parse the response
    let parsed = T::parse_from_ai_response(&response)?;

    // Queue the action
    let action = parsed.clone().into_action_payload();
    pending.actions.push(AiActionEvent { entity, action });

    Ok((parsed, response))
}
