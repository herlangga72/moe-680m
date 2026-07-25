// Job 13: Anthropic-compatible HTTP API server
// Single-user. POST /v1/messages — Anthropic Messages API format.

use crate::inference::{InferenceEngine, InferenceState};
use crate::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;

#[derive(Deserialize)]
struct MessagesRequest {
    model: Option<String>,
    messages: Vec<Message>,
    system: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    stream: Option<bool>,
}

#[derive(Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct MessagesResponse {
    id: String,
    #[serde(rename = "type")]
    resp_type: String,
    role: String,
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: Usage,
}

#[derive(Serialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
}

#[derive(Serialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

pub struct Server {
    pub host: String,
    pub port: u16,
}

impl Server {
    pub fn new(host: &str, port: u16) -> Self {
        Self { host: host.to_string(), port }
    }

    pub fn run(&self, _cfg: crate::gguf::ModelConfig, tokenizer: Tokenizer,
               engine: Mutex<InferenceEngine>) {
        let addr = format!("{}:{}", self.host, self.port);
        eprintln!("Server listening on http://{}", addr);

        let listener = TcpListener::bind(&addr).expect("Failed to bind");
        let mut session = ConversationSession::new();

        for stream in listener.incoming() {
            match stream {
                Ok(mut s) => {
                    if let Err(e) = handle_connection(&mut s, &tokenizer, &engine, &mut session) {
                        eprintln!("Connection error: {}", e);
                    }
                }
                Err(e) => eprintln!("Accept error: {}", e),
            }
        }
    }
}

fn handle_connection(stream: &mut TcpStream, tokenizer: &Tokenizer,
                     engine: &Mutex<InferenceEngine>,
                     session: &mut ConversationSession) -> Result<(), String> {
    let mut buf = [0u8; 65536];
    let n = stream.read(&mut buf).map_err(|e| format!("Read: {}", e))?;
    if n == 0 { return Ok(()); }

    let request = std::str::from_utf8(&buf[..n]).map_err(|_| "Invalid UTF-8".to_string())?;

    let body_start = request.find("\r\n\r\n").ok_or("No headers")? + 4;
    let content_len = request.lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1)?.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let body_end = body_start + content_len.min(buf.len() - body_start);
    let body = &buf[body_start..body_end];

    if !request.starts_with("POST /v1/messages") {
        write_json(stream, 404, r#"{"error":"not found"}"#)?;
        return Ok(());
    }

    let req: MessagesRequest = serde_json::from_slice(body)
        .map_err(|e| format!("JSON parse: {}", e))?;

    let prompt = apply_chat_template(&req.messages, req.system.as_deref());
    let input_ids = tokenizer.encode(&prompt);
    let max_tokens = req.max_tokens.unwrap_or(4096);

    if req.stream.unwrap_or(false) {
        write_streaming(stream, tokenizer, engine, &input_ids, max_tokens, session)
    } else {
        write_json_response(stream, tokenizer, engine, &input_ids, max_tokens, session)
    }
}

fn apply_chat_template(messages: &[Message], system: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(sys) = system {
        out.push_str(&format!("<|im_start|>system\n{}\n<|im_end|>\n", sys));
    }
    for msg in messages {
        let role = match msg.role.as_str() {
            "assistant" => "assistant",
            "user" => "user",
            "system" => "system",
            _ => "user",
        };
        out.push_str(&format!("<|im_start|>{}\n{}\n<|im_end|>\n", role, msg.content));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn write_json_response(stream: &mut TcpStream, tokenizer: &Tokenizer,
                       engine: &Mutex<InferenceEngine>, input_ids: &[u32],
                       max_tokens: u32, _session: &mut ConversationSession) -> Result<(), String> {
    let mut engine = engine.lock().unwrap();
    let mut state = InferenceState::new();
    let mut output_ids = Vec::new();

    for _ in 0..max_tokens {
        match engine.generate(input_ids, &mut state) {
            Ok(token) => {
                output_ids.push(token);
                if token == tokenizer.eos_id { break; }
            }
            Err(e) => { eprintln!("Inference error: {}", e); break; }
        }
    }

    let text = tokenizer.decode(&output_ids);

    let resp = MessagesResponse {
        id: format!("msg_{:016x}", 42u64),
        resp_type: "message".into(),
        role: "assistant".into(),
        content: vec![ContentBlock { block_type: "text".into(), text }],
        model: "qwen3.6-35b-a3b".into(),
        stop_reason: Some("end_turn".into()),
        stop_sequence: None,
        usage: Usage { input_tokens: input_ids.len() as u32, output_tokens: output_ids.len() as u32 },
    };

    let body = serde_json::to_string(&resp).map_err(|e| format!("Serialize: {}", e))?;
    write_json(stream, 200, &body)
}

fn write_streaming(stream: &mut TcpStream, tokenizer: &Tokenizer,
                   engine: &Mutex<InferenceEngine>, input_ids: &[u32],
                   max_tokens: u32, _session: &mut ConversationSession) -> Result<(), String> {
    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n")
        .map_err(|e| format!("Write: {}", e))?;
    stream.flush().ok();

    let msg_id = format!("msg_{:016x}", 42u64);

    sse_write(stream, "message_start",
        &format!(r#"{{"type":"message_start","message":{{"id":"{}","type":"message","role":"assistant","content":[],"model":"qwen3.6-35b-a3b","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{},"output_tokens":0}}}}}}"#, msg_id, input_ids.len()))?;

    sse_write(stream, "content_block_start",
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#)?;

    let mut engine = engine.lock().unwrap();
    let mut state = InferenceState::new();
    let mut output_ids = Vec::new();
    let mut token_count = 0u32;

    for _ in 0..max_tokens {
        match engine.generate(input_ids, &mut state) {
            Ok(token) => {
                output_ids.push(token);
                token_count += 1;
                let text = tokenizer.decode(&[token]);

                sse_write(stream, "content_block_delta",
                    &format!(r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":{}"#,
                        serde_json::to_string(&text).unwrap_or_default()))?;

                if token == tokenizer.eos_id { break; }
            }
            Err(e) => { eprintln!("Inference error: {}", e); break; }
        }
    }

    sse_write(stream, "content_block_stop", r#"{"type":"content_block_stop","index":0}"#)?;
    sse_write(stream, "message_delta",
        &format!(r#"{{"type":"message_delta","delta":{{"stop_reason":"end_turn","stop_sequence":null}},"usage":{{"output_tokens":{}}}}}"#, token_count))?;
    sse_write(stream, "message_stop", r#"{"type":"message_stop"}"#)?;

    Ok(())
}

fn sse_write(stream: &mut TcpStream, event: &str, data: &str) -> Result<(), String> {
    write!(stream, "event: {}\ndata: {}\n\n", event, data)
        .map_err(|e| format!("SSE write: {}", e))?;
    stream.flush().ok();
    Ok(())
}

fn write_json(stream: &mut TcpStream, status: u16, body: &str) -> Result<(), String> {
    write!(stream, "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        status, if status == 200 { "OK" } else { "Error" },
        body.len(), body)
        .map_err(|e| format!("Write: {}", e))?;
    stream.flush().ok();
    Ok(())
}

// Session struct for multi-turn (placeholder)
struct ConversationSession {
    // In production: store KV cache state, conversation history
}

impl ConversationSession {
    fn new() -> Self { Self {} }
}
