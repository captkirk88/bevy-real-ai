//! Traits and utilities for parsing AI responses into typed structs.
//!
//! This module provides the `AiParsable` trait and helper functions for extracting
//! structured data from AI model responses.

use serde::de::DeserializeOwned;

use crate::actions::IntoActionPayload;

/// Trait for types that can be parsed from AI responses.
///
/// Implement this trait (typically via `#[derive(AiAction)]`) to enable automatic
/// parsing of AI responses into your typed structs.
pub trait AiParsable: IntoActionPayload + Clone + Send + Sync + 'static {
    /// Returns a human-readable description of the expected JSON schema.
    /// This is included in prompts to guide the AI's output format.
    fn schema_description() -> String;

    /// Returns the type name for schema descriptions.
    fn type_name() -> &'static str;

    /// Parse an AI response string into this type.
    /// The response may contain JSON embedded in text; this method extracts and parses it.
    fn parse_from_ai_response(response: &str) -> Result<Self, String>
    where
        Self: Sized + DeserializeOwned;
}

/// Helper trait for generating type descriptions in schemas.
/// Implemented for common types to provide human-readable type names.
pub trait AiSchemaType {
    fn type_name() -> &'static str;
}

// Implement AiSchemaType for common types
impl AiSchemaType for String {
    fn type_name() -> &'static str {
        "string"
    }
}

impl AiSchemaType for i8 {
    fn type_name() -> &'static str {
        "integer"
    }
}

impl AiSchemaType for i16 {
    fn type_name() -> &'static str {
        "integer"
    }
}

impl AiSchemaType for i32 {
    fn type_name() -> &'static str {
        "integer"
    }
}

impl AiSchemaType for i64 {
    fn type_name() -> &'static str {
        "integer"
    }
}

impl AiSchemaType for u8 {
    fn type_name() -> &'static str {
        "integer"
    }
}

impl AiSchemaType for u16 {
    fn type_name() -> &'static str {
        "integer"
    }
}

impl AiSchemaType for u32 {
    fn type_name() -> &'static str {
        "integer"
    }
}

impl AiSchemaType for u64 {
    fn type_name() -> &'static str {
        "integer"
    }
}

impl AiSchemaType for f32 {
    fn type_name() -> &'static str {
        "number"
    }
}

impl AiSchemaType for f64 {
    fn type_name() -> &'static str {
        "number"
    }
}

impl AiSchemaType for bool {
    fn type_name() -> &'static str {
        "boolean"
    }
}

impl<T: AiSchemaType> AiSchemaType for Vec<T> {
    fn type_name() -> &'static str {
        "array"
    }
}

impl<T: AiSchemaType> AiSchemaType for Option<T> {
    fn type_name() -> &'static str {
        // For optional fields, we indicate the inner type
        T::type_name()
    }
}

/// Extract JSON from an AI response and parse it into the target type.
///
/// This function handles various common AI response formats:
/// - Pure JSON
/// - JSON wrapped in markdown code blocks (```json ... ```)
/// - JSON embedded in explanatory text
pub fn extract_and_parse_json<T: DeserializeOwned>(response: &str) -> Result<T, String> {
    // First, try to parse the entire response as JSON
    if let Ok(parsed) = serde_json::from_str::<T>(response.trim()) {
        return Ok(parsed);
    }

    // Try to find JSON in a code block
    if let Some(json_str) = extract_json_from_code_block(response) {
        if let Ok(parsed) = serde_json::from_str::<T>(&json_str) {
            return Ok(parsed);
        }
    }

    // Try to find a JSON object anywhere in the response
    if let Some(json_str) = extract_json_object(response) {
        if let Ok(parsed) = serde_json::from_str::<T>(&json_str) {
            return Ok(parsed);
        }

        // If parsing failed, attempt a best-effort repair for common issues (missing array brackets etc.)
        let repaired = try_repair_json(&json_str);
        if repaired != json_str {
            if let Ok(parsed) = serde_json::from_str::<T>(&repaired) {
                return Ok(parsed);
            }
        }
    }

    Err(format!(
        "Failed to parse JSON from AI response. Response was: {}",
        if response.len() > 200 {
            format!("{}...", &response[..200])
        } else {
            response.to_string()
        }
    ))
}

/// Extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
fn extract_json_from_code_block(text: &str) -> Option<String> {
    // Try ```json first
    if let Some(start) = text.find("```json") {
        let content_start = start + 7;
        if let Some(end) = text[content_start..].find("```") {
            return Some(text[content_start..content_start + end].trim().to_string());
        }
    }

    // Try plain ``` block
    if let Some(start) = text.find("```") {
        let content_start = start + 3;
        // Skip to end of line in case there's a language specifier
        let newline_pos = text[content_start..]
            .find('\n')
            .map(|p| content_start + p + 1)
            .unwrap_or(content_start);
        if let Some(end) = text[newline_pos..].find("```") {
            return Some(text[newline_pos..newline_pos + end].trim().to_string());
        }
    }

    None
}

/// Extract a JSON object from text by finding matching braces
fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in text[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }

    None
}

/// Try to repair common JSON mistakes produced by LLMs such as missing closing
/// array brackets. This is a best-effort heuristic — it attempts to balance
/// '[' / ']' by inserting missing ']' after the last object in the array.
fn try_repair_json(input: &str) -> String {
    // Quick check: if parsing works, return original
    if serde_json::from_str::<serde_json::Value>(input).is_ok() {
        return input.to_string();
    }

    let mut in_string = false;
    let mut escape = false;
    let mut last_open_array_pos: Option<usize> = None;

    for (i, ch) in input.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == '[' {
            last_open_array_pos = Some(i);
        }
        if ch == ']' {
            // matched an array; clear last_open_array_pos
            last_open_array_pos = None;
        }
    }

    // If there was an unmatched '[', insert a ']' after the last '}' following it.
    if let Some(open_pos) = last_open_array_pos {
        // Find the position of the first '}' after the open_pos
        let mut in_string = false;
        let mut escape = false;
        for (i, ch) in input.char_indices().skip(open_pos) {
            if escape {
                escape = false;
                continue;
            }
            if ch == '\\' && in_string {
                escape = true;
                continue;
            }
            if ch == '"' {
                in_string = !in_string;
                continue;
            }
            if in_string { continue; }
            if ch == '}' {
                // Insert a closing ']' right after this brace
                let mut repaired = String::with_capacity(input.len() + 2);
                repaired.push_str(&input[..i + 1]);
                repaired.push(']');
                repaired.push_str(&input[i + 1..]);
                // Also remove any ',]' sequences that can be invalid (trailing commas)
                repaired = repaired.replace(",]", "]");
                // If this repairs the JSON, return it; otherwise, fall through to appending at end
                if serde_json::from_str::<serde_json::Value>(&repaired).is_ok() {
                    return repaired;
                } else {
                    // keep repaired attempt but continue to final append option
                    return repaired;
                }
            }
        }

        // If we didn't find a '}', append a ']' just before the final '}' if present
        if let Some(last_brace_pos) = input.rfind('}') {
            let mut repaired = String::with_capacity(input.len() + 2);
            repaired.push_str(&input[..last_brace_pos]);
            repaired.push(']');
            repaired.push_str(&input[last_brace_pos..]);
            repaired = repaired.replace(",]", "]");
            return repaired;
        }

        // As a last resort, append ']' at the end
        let mut repaired = input.to_string();
        repaired.push(']');
        return repaired;
    }

    // No repair heuristic applied — return original
    input.to_string()
}

/// Build a system prompt that instructs the AI to respond with the expected JSON format.
pub fn build_typed_prompt<T: AiParsable>(user_message: &str) -> String {
    format!(
        "You must respond with ONLY a valid JSON object matching this schema:\n{}\n\nUser request: {}\n\nRespond with only the JSON object, no explanation.",
        T::schema_description(),
        user_message
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, Deserialize, PartialEq)]
    struct TestStruct {
        name: String,
        value: i32,
    }

    #[test]
    fn test_parse_pure_json() {
        let json = r#"{"name": "test", "value": 42}"#;
        let result: TestStruct = extract_and_parse_json(json).expect("should parse");
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_parse_json_in_code_block() {
        let response = r#"Here is the result:
```json
{"name": "test", "value": 42}
```
"#;
        let result: TestStruct = extract_and_parse_json(response).expect("should parse");
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_parse_json_embedded_in_text() {
        let response = r#"Sure, here's what you asked for: {"name": "test", "value": 42} Hope that helps!"#;
        let result: TestStruct = extract_and_parse_json(response).expect("should parse");
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_extract_json_object_handles_nested() {
        let text = r#"Result: {"outer": {"inner": 1}, "value": 2}"#;
        let json = extract_json_object(text).expect("should find json");
        assert_eq!(json, r#"{"outer": {"inner": 1}, "value": 2}"#);
    }

    #[test]
    fn test_repair_missing_array_bracket() {
        // The incoming AI response is missing the closing ']' for the actions array.
        let broken = r#"{"name":"spawn","type":"action","args":{"target":"player","actions":[{"actor":"npc_goblins.girl", "x": 7 , "y" :9} },"id":"goblin_spawn"}"#;
        // First ensure naive extraction gives us the broken string
        let extracted = extract_json_object(broken).expect("should extract");
        assert!(serde_json::from_str::<serde_json::Value>(&extracted).is_err());

        // Try repair
        let repaired = try_repair_json(&extracted);
        // Now repaired should parse
        let v: serde_json::Value = serde_json::from_str(&repaired).expect("repaired json should parse");
        assert_eq!(v["name"], "spawn");
        assert_eq!(v["id"], "goblin_spawn");
        // Ensure actions is an array
        assert!(v["args"]["actions"].is_array());
    }
}

pub(crate) mod json_parser {
    use super::*;
    use kalosm::language::{Parser, CreateParserState, ParseStatus, ParserError};
    use std::borrow::Cow;

    /// A very small `Parser` implementation that extracts JSON from the accumulated
    /// input and attempts to parse it with `serde_json`. This is used when callers
    /// want to constrain model output to JSON without building a complex parser.
    pub struct JsonParser;

    #[derive(Clone, Debug)]
    pub struct JsonParserState {
        buffer: Vec<u8>,
    }

    impl CreateParserState for JsonParser {
        fn create_parser_state(&self) -> JsonParserState {
            JsonParserState { buffer: Vec::new() }
        }
    }

    impl Parser for JsonParser {
        type Output = serde_json::Value;
        type PartialState = JsonParserState;

        fn parse<'a>(&self, state: &Self::PartialState, input: &'a [u8]) -> Result<ParseStatus<'a, Self::PartialState, Self::Output>, ParserError> {
            // Combine previous buffer and new input to search for JSON
            let mut combined = state.buffer.clone();
            combined.extend_from_slice(input);
            let text = String::from_utf8_lossy(&combined).to_string();

            // First try code block JSON
            if let Some(json_str) = extract_json_from_code_block(&text) {
                match serde_json::from_str::<serde_json::Value>(&json_str) {
                    Ok(v) => {
                        // find position to compute remaining bytes in `input`
                        if let Some(pos) = text.find(&json_str) {
                            let end = pos + json_str.len();
                            let buffer_len = state.buffer.len();
                            let remaining = if end <= buffer_len { &input[0..0] } else { &input[end - buffer_len..] };
                            return Ok(ParseStatus::Finished { result: v, remaining });
                        }
                        return Ok(ParseStatus::Finished { result: v, remaining: &input[0..0] });
                    }
                    Err(e) => return Err(ParserError::msg(format!("invalid json: {}", e))),
                }
            }

            // Try to find a JSON object anywhere in the text
            if let Some(obj) = extract_json_object(&text) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&obj) {
                    if let Some(pos) = text.find(&obj) {
                        let end = pos + obj.len();
                        let buffer_len = state.buffer.len();
                        let remaining = if end <= buffer_len { &input[0..0] } else { &input[end - buffer_len..] };
                        return Ok(ParseStatus::Finished { result: v, remaining });
                    }
                    return Ok(ParseStatus::Finished { result: v, remaining: &input[0..0] });
                } else {
                    return Err(ParserError::msg("invalid json"));
                }
            }

            // No JSON found yet: request more input
            Ok(ParseStatus::Incomplete {
                new_state: JsonParserState { buffer: combined },
                required_next: Cow::Borrowed("}"),
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde::Deserialize;

        #[derive(Debug, Clone, Deserialize, PartialEq)]
        struct TestStruct {
            name: String,
            value: i32,
        }

        #[test]
        fn json_parser_parses_embedded() {
            let parser = JsonParser;
            let state = parser.create_parser_state();
            let input = b"Some text before {\"name\": \"x\", \"value\": 3} trailing";
            match parser.parse(&state, input).expect("parse") { 
                ParseStatus::Finished { result, .. } => {
                    let s: TestStruct = serde_json::from_value(result).expect("deserialize");
                    assert_eq!(s.name, "x");
                    assert_eq!(s.value, 3);
                }
                other => panic!("unexpected parse status: {:?}", other),
            }
        }
    }
}
