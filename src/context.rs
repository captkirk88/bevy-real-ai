use bevy::prelude::*;

/// Marker component that indicates an entity should be considered for AI context gathering.
/// Only entities with this component will be scanned for nearby context information.
#[derive(Component, Debug, Clone, Copy)]
pub struct AIAware;

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

/// Trait for components that can summarize themselves into a short text snippet
/// suitable for inclusion in AI context. Implement this on any component that should
/// be easy to summarize (e.g. inventories, simple status components).
pub trait AiDescribe: Send + Sync + 'static {
    /// Return a short summary or None if nothing useful to report.
    /// Takes the entity ID and world reference for richer context-aware summarization.
    fn summarize(&self, entity: Entity, world: &World) -> Option<String>;
}

use bevy::prelude::World;
use std::sync::Arc;

/// Registry mapping component types to extractor functions that can summarize them.
#[derive(Resource, Default)]
pub struct ContextExtractorRegistry {
    extractors: Vec<Arc<dyn Fn(Entity, &World) -> Option<String> + Send + Sync>>,
}

impl ContextExtractorRegistry {
    pub fn new() -> Self { Self { extractors: Vec::new() } }

    /// Register an extractor for a concrete component type `T`.
    pub fn register<T: Component + 'static, F>(&mut self, f: F)
    where
        F: Fn(&T, Entity, &World) -> Option<String> + Send + Sync + 'static,
    {
        let cb = Arc::new(move |ent: Entity, world: &World| {
            world.get::<T>(ent).and_then(|c| f(c, ent, world))
        }) as Arc<dyn Fn(Entity, &World) -> Option<String> + Send + Sync>;
        self.extractors.push(cb);
    }

    /// Convenience helper to register an extractor for types that implement `AiDescribe`.
    pub fn register_summarizable<T: Component + AiDescribe + 'static>(&mut self) {
        self.register::<T, _>(|c: &T, ent: Entity, world: &World| c.summarize(ent, world));
    }
}

/// On-demand gather system: when `ContextGatherRequest` contains an entity, collect nearby
/// entities by iterating world entities and calling registered extractors.
pub fn gather_on_request_world(world: &mut World) {
    // Take and release the request resource to avoid long mutable borrow
    let ent_opt = {
        let mut req = match world.get_resource_mut::<ContextGatherRequest>() { Some(r) => r, None => return };
        req.0.take()
    };
    let Some(ent) = ent_opt else { return };

    let cfg = if let Some(cfg) = world.get_resource::<ContextGatherConfig>() { cfg.clone() } else { return };
    // Clone extractors out of the registry to avoid holding the registry borrow during world queries
    let extractors = {
        let reg = match world.get_resource::<ContextExtractorRegistry>() { Some(r) => r, None => return };
        reg.extractors.clone()
    };

    let mut added = 0usize;

    // Find the requester's transform first (requester doesn't need to be AI-aware to request context)
    let requester_t = {
        let mut q_requester = world.query::<(Entity, &Transform)>();
        let mut requester_t_opt = None;
        for (e, t) in q_requester.iter(world) {
            if e == ent { 
                requester_t_opt = Some(t.clone()); 
                break; 
            }
        }
        match requester_t_opt {
            Some(t) => t,
            None => return, // Requester has no transform
        }
    };

    // iterate entities with transforms and AI awareness to compute distances compactly
    let mut q = world.query::<(Entity, &Transform, &AIAware)>();

    // Collect summaries per entity (grouped) to produce a single natural-language
    // context message per nearby entity, which models typically handle better.
    use std::collections::HashMap;
    let mut per_entity: HashMap<Entity, Vec<String>> = HashMap::new();

    for (e, t, _) in q.iter(world) {
        if e == ent { continue; }
        let dist_sq = requester_t.translation.distance_squared(t.translation);
        if dist_sq > cfg.radius * cfg.radius { continue; }

        // Call each registered extractor; accumulate any returned summaries for this entity
        for (_i, ex) in extractors.iter().enumerate() {
            if let Some(txt) = ex(e, world) {
                per_entity.entry(e).or_insert_with(Vec::new).push(txt);
                added += 1;
                if added >= cfg.max_docs { break; }
            }
        }

        if added >= cfg.max_docs { break; }
    }

    // Attach combined messages as an `AiContext` component on the requester entity
    use crate::rag::AiContext;
    if !per_entity.is_empty() {
        let mut context = AiContext::new();
        let mut observations = Vec::new();
        for (_entity, vec) in per_entity.iter() {
            // Join extractor outputs into natural language descriptions
            let joined = vec.join(". ");
            observations.push(joined);
        }
        // Format as observations about the scene, not about the AI itself
        let scene_description = format!("{}", observations.join("; "));
        context.add_context(scene_description);
        // Safe to insert component even if present; replace existing context
        world.entity_mut(ent).insert(context);
    }
}
