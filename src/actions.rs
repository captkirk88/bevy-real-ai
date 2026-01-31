use serde_json::Value;
use bevy::prelude::*;
use bevy::ecs::system::SystemParam;
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
        if let Value::Object(ref map) = self.params {
            map.get(key)
        } else {
            None
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

/// A boxed system invoked for a named AI action. Handlers run with `()` input
/// and may read a temporary `CurrentAiAction` resource containing the action event.
pub type AiActionHandler = Box<dyn System<In = (), Out = ()>>;

/// Temporary resource made available while a handler system runs so it can access
/// the action that triggered it.
#[derive(Resource, Clone)]
pub struct CurrentAiAction(pub AiActionEvent);

/// System parameter providing easy access to the current action. Handlers can accept
/// SystemParam providing easy access to the current action. Handlers can accept
/// this parameter to access the triggering `AiActionEvent` and its `ActionPayload`.
#[derive(SystemParam)]
pub struct AiAction<'w, 's> {
    pub current: Res<'w, CurrentAiAction>,
    // use `'s` to satisfy SystemParam signature
    _marker: std::marker::PhantomData<&'s ()>,
}

impl<'w, 's> AiAction<'w, 's> {
    pub fn event(&self) -> &AiActionEvent {
        &self.current.0
    }

    pub fn payload(&self) -> &ActionPayload {
        &self.current.0.action
    }

    pub fn entity(&self) -> Entity {
        self.current.0.entity
    }
}

/// Pending actions that have been parsed and await processing by registered handlers.
#[derive(Resource, Default)]
pub struct PendingAiActions {
    pub actions: Vec<AiActionEvent>,
}

/// Registry mapping action names to boxed `System`s.
#[derive(Resource, Default)]
pub struct AiActionRegistry {
    handlers: HashMap<String, AiActionHandler>,
}

impl AiActionRegistry {
    pub fn new() -> Self {
        Self { handlers: HashMap::new() }
    }

    /// Register a handler system for an action name.
    ///
    /// The provided `system` must be convertible to a Bevy `System` that accepts
    /// `()` input. The handler can then read `CurrentAiAction` as a `Res<CurrentAiAction>`.
    pub fn register<S, M>(&mut self, name: &str, system: S)
    where
        S: IntoSystem<(), (), M> + 'static,
    {
        self.handlers.insert(name.to_string(), Box::new(IntoSystem::into_system(system)));
    }

    /// Register a typed handler system for an action name.
    ///
    /// The provided `system` must be convertible to a Bevy `System` that accepts
    /// `In<T>` input. The handler can then operate directly on the typed struct.
    pub fn register_typed<T, S, M>(&mut self, name: &str, system: S)
    where
        T: 'static + Send + Sync + serde::de::DeserializeOwned,
        S: bevy::ecs::system::IntoSystem<bevy::ecs::system::In<T>, (), M> + 'static,
    {
        use bevy::ecs::system::IntoSystem;
        // Convert the user system into a concrete System type and capture it.
        let mut user_system = IntoSystem::into_system(system);

        // Own the name so it can be captured by the wrapper
        let name_owned = name.to_string();
        let name_for_register = name_owned.clone();

        // Wrapper is an exclusive system that reads CurrentAiAction, deserializes T,
        // and runs the user system with T.
        let wrapper = move |world: &mut World| {
            if let Some(current) = world.get_resource::<CurrentAiAction>() {
                let payload = &current.0.action.params;
                match serde_json::from_value::<T>(payload.clone()) {
                    Ok(typed) => {
                        user_system.initialize(world);
                        let _ = user_system.run(typed, world);
                        user_system.apply_deferred(world);
                    }
                    Err(e) => bevy::log::error!("Failed to deserialize typed action for {}: {}", name_owned, e),
                }
            }
        };

        // Register the wrapper as a normal handler (it matches In=(), Out=()).
        self.register(&name_for_register, wrapper);
    }

    /// Get a mutable reference to a handler system by name, if any.
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

    // For each action event, run any registered handler system with a temporary CurrentAiAction
    for evt in pending.into_iter() {
        world.resource_scope::<AiActionRegistry, _>(|world, mut registry| {
            // Insert the current action as a temporary resource so handlers can read it
            world.insert_resource(CurrentAiAction(evt.clone()));

            if let Some(handler) = registry.get_mut(&evt.action.name) {
                // Log the action execution for debugging
                debug!("Executing handler '{}' for entity {:?}", evt.action.name, evt.entity);
                // Initialize and run the handler (it operates with `()` input)
                handler.initialize(world);
                let _ = handler.run((), world);
                handler.apply_deferred(world);
            }

            // Remove the temporary resource
            world.remove_resource::<CurrentAiAction>();
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
/// use bevy_real_ai::AiParse;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Clone, Debug, Serialize, Deserialize, AiParse)]
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
) -> Result<(T,String), String>
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