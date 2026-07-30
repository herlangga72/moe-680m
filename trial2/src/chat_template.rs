use crate::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

/// A single message in the conversation.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

/// A message's content can be plain text or a list of content blocks.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block within a message (text, tool_use, tool_result).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

// ---------------------------------------------------------------------------
// ChatTemplate
// ---------------------------------------------------------------------------

/// Qwen `<|im_start|>role\n...<|im_end|>\n` chat format.
///
/// ponytail: Qwen format is simple enough to hardcode — doesn't change
/// between model revisions.
pub struct ChatTemplate;

impl ChatTemplate {
    /// Apply the Qwen chat template to a list of messages.
    ///
    /// Produces a token sequence ending with `<|im_start|>assistant\n`, ready
    /// for generation.
    pub fn apply(
        tokenizer: &Tokenizer,
        messages: &[Message],
        system: Option<&str>,
    ) -> Vec<u32> {
        let mut tokens = Vec::new();

        // Optional system prompt
        if let Some(sys) = system {
            if !sys.is_empty() {
                tokens.push(tokenizer.im_start());
                tokens.extend(tokenizer.encode("system\n"));
                tokens.extend(tokenizer.encode(sys));
                tokens.push(tokenizer.im_end());
                tokens.push(b'\n' as u32);
            }
        }

        // Conversation messages
        for msg in messages {
            tokens.push(tokenizer.im_start());
            tokens.extend(tokenizer.encode(&msg.role));
            tokens.push(b'\n' as u32);

            match &msg.content {
                MessageContent::Text(text) => {
                    tokens.extend(tokenizer.encode(text));
                }
                MessageContent::Blocks(blocks) => {
                    for block in blocks {
                        match block {
                            ContentBlock::Text { text } => {
                                tokens.extend(tokenizer.encode(text));
                            }
                            ContentBlock::ToolUse { name, input, .. } => {
                                let tool_json = format!(
                                    r#"{{"name":"{}","arguments":{}}}"#,
                                    name,
                                    serde_json::to_string(input).unwrap_or_default(),
                                );
                                tokens.extend(tokenizer.encode(&tool_json));
                            }
                            ContentBlock::ToolResult { content, .. } => {
                                tokens.extend(tokenizer.encode(content));
                            }
                        }
                    }
                }
            }

            tokens.push(tokenizer.im_end());
            tokens.push(b'\n' as u32);
        }

        // Assistant header — generation starts after this
        tokens.push(tokenizer.im_start());
        tokens.extend(tokenizer.encode("assistant\n"));

        tokens
    }
}
