//! Derive macros for bevy_real_ai typed parsing.
//!
//! Provides `#[derive(AiAction)]` which generates JSON parsing capabilities
//! for structs, allowing AI responses to be automatically converted into typed data
//! and action payloads.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Convert a CamelCase or PascalCase string to snake_case.
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push(ch);
        }
    }
    result
}

/// Derive macro for AI response parsing and action payload conversion.
///
/// Generates implementations of:
/// - `AiParsable` - for parsing JSON from AI responses
/// - `IntoActionPayload` - for converting parsed structs into action payloads
/// - A static `register` method for registering handlers with `AiActionRegistry`
///
/// The struct must also derive `serde::Deserialize` and `serde::Serialize`.
///
/// The action name is derived from the struct name in snake_case.
/// For example, `SpawnEntityAction` becomes `"spawn_entity_action"`.
///
/// # Example
/// ```ignore
/// use bevy_real_ai_derive::AiAction;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Clone, Debug, Serialize, Deserialize, AiAction)]
/// struct SpawnAction {
///     pub name: String,
///     pub x: f32,
///     pub y: f32,
/// }
///
/// // Register a handler using the generated method (action name is automatic):
/// SpawnAction::register(&mut registry, |In(action): In<SpawnAction>, mut commands: Commands| { /* handle */ });
///
/// // One way is to use with prompt_typed_action:
/// prompt_typed_action::<SpawnAction>(&backend, "spawn a player at 0,0", entity, &mut pending)?;
/// ```
#[proc_macro_derive(AiAction)]
pub fn derive_ai_action(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Extract field information for schema generation and action payload
    let (fields_schema, field_params) = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let field_schemas: Vec<_> = fields
                    .named
                    .iter()
                    .map(|f| {
                        let field_name = f.ident.as_ref().expect("Named field must have ident");
                        let field_name_str = field_name.to_string();
                        let field_type = &f.ty;
                        quote! {
                            (#field_name_str, <#field_type as bevy_real_ai::parse::AiSchemaType>::type_name())
                        }
                    })
                    .collect();

                // Generate the with_param calls for each field
                let field_param_calls: Vec<_> = fields
                    .named
                    .iter()
                    .map(|f| {
                        let field_name = f.ident.as_ref().expect("Named field must have ident");
                        let field_name_str = field_name.to_string();
                        quote! {
                            .with_param(#field_name_str, serde_json::json!(self.#field_name))
                        }
                    })
                    .collect();

                (
                    quote! { vec![#(#field_schemas),*] },
                    quote! { #(#field_param_calls)* },
                )
            }
            _ => (quote! { vec![] }, quote! {}),
        },
        _ => (quote! { vec![] }, quote! {}),
    };

    let struct_name_str = name.to_string();
    let action_name_str = to_snake_case(&struct_name_str);

    // For named-field structs, prepare default initializers and type bounds for Default impl
    let (default_inits, default_type_bounds) = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let inits: Vec<_> = fields
                    .named
                    .iter()
                    .map(|f| {
                        let field_name = f.ident.as_ref().expect("Named field must have ident");
                        quote! { #field_name: std::default::Default::default() }
                    })
                    .collect();
                let types: Vec<_> = fields.named.iter().map(|f| &f.ty).collect();
                (
                    quote! { #(#inits),* },
                    quote! { #(#types: std::default::Default),* },
                )
            }
            _ => (quote! {}, quote! {}),
        },
        _ => (quote! {}, quote! {}),
    };

    // Build the merged where clause for Default impl (include existing where predicates if present)
    let default_where_clause = if default_type_bounds.to_string().trim().is_empty() {
        quote! {}
    } else if let Some(wc) = &input.generics.where_clause {
        let preds = &wc.predicates;
        quote! { where #preds, #default_type_bounds }
    } else {
        quote! { where #default_type_bounds }
    };

    let expanded = quote! {
        impl #impl_generics bevy_real_ai::parse::AiParsable for #name #ty_generics #where_clause {
            fn schema_description() -> String {
                let fields: Vec<(&str, &str)> = #fields_schema;
                let field_descs: Vec<String> = fields
                    .iter()
                    .map(|(name, ty)| format!("  \"{}\": <{}>", name, ty))
                    .collect();
                format!(
                    "JSON object with fields:\n{{\n{}\n}}",
                    field_descs.join(",\n")
                )
            }

            fn type_name() -> &'static str {
                #struct_name_str
            }

            fn parse_from_ai_response(response: &str) -> Result<Self, String>
            where
                Self: Sized + serde::de::DeserializeOwned,
            {
                bevy_real_ai::parse::extract_and_parse_json(response)
            }
        }

        impl #impl_generics bevy_real_ai::actions::IntoActionPayload for #name #ty_generics #where_clause {
            fn action_name() -> &'static str {
                #action_name_str
            }

            fn into_action_payload(self) -> bevy_real_ai::actions::ActionPayload {
                bevy_real_ai::actions::ActionPayload::new(#action_name_str)
                    #field_params
            }
        }

        impl #impl_generics #name #ty_generics #where_clause {
            /// Register a handler for this action type with the given registry.
            /// The handler receives the parsed struct as `In<Self>` and can use any Bevy system params.
            ///
            /// # Example
            /// ```ignore
            /// SpawnAction::register(&mut registry, |In(action): In<SpawnAction>, mut cmds: Commands| {
            ///     cmds.spawn(...);
            /// });
            /// ```
            pub fn register<S, M>(registry: &mut bevy_real_ai::actions::AiActionRegistry, system: S)
            where
                S: bevy::ecs::system::IntoSystem<bevy::ecs::system::In<Self>, (), M> + 'static,
                Self: Sized + 'static + Send + Sync,
            {
                registry.register_typed::<Self, S, M>(
                    <Self as bevy_real_ai::actions::IntoActionPayload>::action_name(),
                    system,
                );
            }
        }

        // Implement Default using field defaults when applicable
        impl #impl_generics std::default::Default for #name #ty_generics #default_where_clause {
            fn default() -> Self {
                Self { #default_inits }
            }
        }
    };

    TokenStream::from(expanded)
}
