use bevy::prelude::*;
use bevy::ecs::system::{SystemParam};

/// Marker component that indicates an entity should be considered for AI context gathering.
/// Only entities with this component will be scanned for nearby context information.
#[derive(Component, Debug, Clone, Copy)]
pub struct AIAware;

/// Marker component that tags an entity as an AI entity.
/// Entities with this component can use the `AiEntity` system parameter to access their transform.
#[derive(Component, Debug, Clone, Copy, Default)]
#[component(storage = "SparseSet")]
pub struct AI;

/// Configuration for on-demand context gathering.
#[derive(Resource, Debug, Clone)]
pub struct ContextGatherConfig {
    /// Search radius (world units) around the requester entity.
    pub radius: f32,
    /// Maximum number of documents to collect per gather request.
    pub max_docs: usize,
}

impl Default for ContextGatherConfig {
    fn default() -> Self {
        Self { radius: 10.0, max_docs: 8 }
    }
}

/// Resource used to request a single on-demand gather for an entity.
/// Set this to Some(entity) to trigger one gather run; the system will clear it after running.
#[derive(Resource, Default, Debug)]
pub struct ContextGatherRequest(pub Option<Entity>);

/// Temporary resource holding the entity being processed by context gathering systems.
/// Systems read this to know which entity they're gathering context for.
#[derive(Resource, Debug, Clone, Copy)]
pub struct CurrentContextEntity(pub Entity);

/// Custom system parameter providing easy access to the current AI context entity,
/// the context gathering configuration, and spatial queries.
/// Systems can use this parameter to get the entity being processed by the gather function
/// and to check spatial relationships with other entities.
#[derive(SystemParam)]
pub struct AiEntity<'w, 's> {
    current: Res<'w, CurrentContextEntity>,
    config: Res<'w, ContextGatherConfig>,
    transforms: Query<'w, 's, &'static Transform, With<AI>>,
}

impl<'w, 's> AiEntity<'w, 's> {
    /// Get the entity being processed for AI context gathering
    pub fn entity(&self) -> Entity {
        self.current.0
    }

    /// Get the Transform component of the AI entity being processed.
    /// Returns None if the entity has no Transform.
    pub fn transform(&self) -> Option<&Transform> {
        self.transforms.get(self.current.0).ok()
    }

    /// Get the position (translation) of the AI entity being processed.
    /// Returns None if the entity has no Transform.
    pub fn position(&self) -> Option<Vec3> {
        self.transform().map(|t| t.translation)
    }

    /// Get the context gathering configuration
    pub fn config(&self) -> &ContextGatherConfig {
        &self.config
    }

    /// Get the configured gather radius
    pub fn radius(&self) -> f32 {
        self.config.radius
    }

    /// Get the configured max documents limit
    pub fn max_docs(&self) -> usize {
        self.config.max_docs
    }

    /// Check if a position is within the gather radius of a given origin position.
    /// Returns true if the distance between origin and other_pos is within the configured radius.
    pub fn aware_of_pos(&self, origin: Vec3, other_pos: Vec3) -> bool {
        origin.distance(other_pos) <= self.config.radius
    }

    /// Check if a position is within the gather radius of the AI entity.
    /// Returns false if the AI entity has no Transform.
    pub fn aware_of(&self, other_pos: Vec3) -> bool {
        self.position()
            .map(|pos| pos.distance(other_pos) <= self.config.radius)
            .unwrap_or(false)
    }

    /// Check if another entity should be considered for context gathering.
    /// Returns true if the entity is not the requester itself and is within the gather radius.
    /// Returns false if the AI entity has no Transform.
    pub fn is_nearby(&self, other_entity: Entity, other_pos: Vec3) -> bool {
        other_entity != self.current.0 && self.aware_of(other_pos)
    }
}

impl<'w, 's> std::ops::Deref for AiEntity<'w, 's> {
    type Target = Entity;
    
    fn deref(&self) -> &Self::Target {
        &self.current.0
    }
}

/// Type alias for a context-gathering Bevy System.
/// Systems are stored as boxed systems that read CurrentContextEntity resource
/// and return an optional `AiMessage` via world.run_system_with((), &mut system).
pub type AiContextSystem = Box<dyn System<In = (), Out = Option<crate::rag::AiMessage>>>;

/// Registry of Bevy Systems that gather and build AI context for entities.
/// Systems are stored as boxed dyn System and invoked via world.run_system_with with () input.
#[derive(Resource, Default)]
pub struct AiSystemContextStore {
    systems: Vec<AiContextSystem>,
}

impl AiSystemContextStore {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
        }
    }

    /// Add a context-gathering system to the store.
    /// 
    /// The system should return an `Option<AiMessage>` and can use any valid Bevy system parameters.
    /// 
    /// # Example
    /// ```ignore
    /// store.add_system(|ai_entity: AiEntity, query: Query<&MyComponent>| {
    ///     // gather context...
    ///     Some(AiMessage::system("context"))
    /// });
    /// ```
    pub fn add_system<M>(&mut self, system: impl IntoSystem<(), Option<crate::rag::AiMessage>, M> + 'static) {
        self.systems.push(Box::new(IntoSystem::into_system(system)));
    }

    /// Get a reference to all registered systems.
    pub fn systems(&self) -> &[AiContextSystem] {
        &self.systems
    }

    /// Get a mutable reference to all registered systems (for internal use).
    pub fn systems_mut(&mut self) -> &mut [AiContextSystem] {
        &mut self.systems
    }
}

/// On-demand gather system: when `ContextGatherRequest` contains an entity, run all context-gathering
/// systems from `AiSystemContextStore` in order. Systems read CurrentContextEntity resource to know
/// which entity they're gathering context for. Returned messages are collected and added to the entity's `AiContext`.
pub fn gather_on_request_world(world: &mut World) {
    // Take and release the request resource to avoid long mutable borrow
    let ent_opt = {
        let mut req = match world.get_resource_mut::<ContextGatherRequest>() {
            Some(r) => r,
            None => return,
        };
        req.0.take()
    };
    let Some(ent) = ent_opt else { return };

    // Insert the temporary resource so systems can read which entity they're processing
    world.insert_resource(CurrentContextEntity(ent));

    // Get the number of systems to run
    let num_systems = {
        match world.get_resource::<AiSystemContextStore>() {
            Some(store) => store.systems.len(),
            None => return,
        }
    };

    // Collect messages from all systems
    let mut messages = Vec::new();

    // Run each system with () input - systems read CurrentContextEntity from world
    for i in 0..num_systems {
        if let Some(mut store) = world.get_resource_mut::<AiSystemContextStore>() {
            if i < store.systems.len() {
                // Take ownership of the system
                let mut system = store.systems.remove(i);
                drop(store); // Release the borrow

                // Initialize the system
                system.initialize(world);
                
                // Run the system directly with &mut World
                let result = system.run((), world);

                // Apply any deferred commands
                system.apply_deferred(world);

                // Collect the returned message if present
                if let Ok(Some(msg)) = result {
                    messages.push(msg);
                }

                // Put the system back
                if let Some(mut store) = world.get_resource_mut::<AiSystemContextStore>() {
                    store.systems.insert(i, system);
                }
            }
        }
    }

    // Remove the temporary resource
    world.remove_resource::<CurrentContextEntity>();

    // Attach collected messages as an `AiContext` component on the requester entity if any were returned
    use crate::rag::AiContext;
    if !messages.is_empty() {
        let mut context = AiContext::new();
        for msg in messages {
            // Messages from systems should be converted to system context
            if let crate::rag::AiMessage::System(text) = msg {
                context.add_context(text);
            } else {
                // If a system returns a user/assistant message, convert to system context
                context.add_context(format!("{:?}", msg));
            }
        }
        // Safe to insert component even if present; replace existing context
        world.entity_mut(ent).insert(context);
    }
}

