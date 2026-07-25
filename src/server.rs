use crate::inference::{InferenceEngine, InferenceState};
use crate::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Deserialize)]
struct MsgReq {
    messages: Vec<Msg>,
    system: Option<String>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct Msg {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct MsgResp {
    id: String,
    #[serde(rename = "type")]
    resp_type: String,
    role: String,
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: Option<String>,
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

pub fn serve(host: &str, port: u16, tokenizer: Tokenizer, engine: Mutex<InferenceEngine>) {
    let addr = format!("{}:{}", host, port);
    eprintln!("Server on http://{}", addr);
    let listener = TcpListener::bind(&addr).expect("bind");
    for stream in listener.incoming() {
        match stream {
            Ok(mut s) => handle_conn(&mut s, &tokenizer, &engine),
            Err(e) => eprintln!("accept: {}", e),
        }
    }
}

fn handle_conn(stream: &mut TcpStream, tokenizer: &Tokenizer, engine: &Mutex<InferenceEngine>) {
    let mut buf = [0u8; 65536];
    let n = match stream.read(&mut buf) {
        Ok(0) => return,
        Ok(n) => n,
        Err(e) => { eprintln!("read: {}", e); return; }
    };
    let raw = match std::str::from_utf8(&buf[..n]) {
        Ok(s) => s,
        Err(_) => return,
    };
    let body_start = match raw.find("\r\n\r\n") {
        Some(i) => i + 4,
        None => return,
    };
    let body = &buf[body_start..n];
    if !raw.starts_with("POST /v1/messages") {
        let _ = write!(stream, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        return;
    }
    let req: MsgReq = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => { eprintln!("json: {}", e); return; }
    };
    let prompt = build_prompt(&req.messages, req.system.as_deref());
    let input_ids = tokenizer.encode(&prompt);
    let max_tokens = req.max_tokens.unwrap_or(4096);
    match req.stream.unwrap_or(false) {
        true => stream_response(stream, tokenizer, engine, &input_ids, max_tokens),
        false => json_response(stream, tokenizer, engine, &input_ids, max_tokens),
    }
}

fn json_response(stream: &mut TcpStream, tokenizer: &Tokenizer,
                 engine: &Mutex<InferenceEngine>, input_ids: &[u32], max_tokens: u32) {
    let mut engine = engine.lock().unwrap();
    let mut state = InferenceState::new();
    let mut output = Vec::new();
    for _ in 0..max_tokens {
        match engine.generate(input_ids, &mut state) {
            Ok(t) => { output.push(t); if t == tokenizer.eos_id { break; } }
            Err(e) => { eprintln!("infer: {}", e); break; }
        }
    }
    let text = tokenizer.decode(&output);
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let resp = MsgResp {
        id: format!("msg_{:x}", ts),
        resp_type: "message".into(),
        role: "assistant".into(),
        content: vec![ContentBlock { block_type: "text".into(), text }],
        model: env!("CARGO_PKG_NAME").into(),
        stop_reason: Some("end_turn".into()),
        usage: Usage { input_tokens: input_ids.len() as u32, output_tokens: output.len() as u32 },
    };
    let body = serde_json::to_string(&resp).unwrap_or_default();
    let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", body.len(), body);
}

fn stream_response(stream: &mut TcpStream, tokenizer: &Tokenizer,
                   engine: &Mutex<InferenceEngine>, input_ids: &[u32], max_tokens: u32) {
    let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n");
    let _ = stream.flush();

    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let msg_id = format!("msg_{:x}", ts);
    let model = env!("CARGO_PKG_NAME");
    sse(stream, "message_start", &format!(
        r#"{{"type":"message_start","message":{{"id":"{}","type":"message","role":"assistant","content":[],"model":"{}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{},"output_tokens":0}}}}}}"#,
        msg_id, model, input_ids.len()));
    sse(stream, "content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#);

    let mut engine = engine.lock().unwrap();
    let mut state = InferenceState::new();
    let mut n = 0u32;
    for _ in 0..max_tokens {
        match engine.generate(input_ids, &mut state) {
            Ok(t) => {
                n += 1;
                let text = tokenizer.decode(&[t]);
                sse(stream, "content_block_delta",
                    &format!(r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":{}}}}}"#,
                        serde_json::to_string(&text).unwrap_or_default()));
                if t == tokenizer.eos_id { break; }
            }
            Err(e) => { eprintln!("infer: {}", e); break; }
        }
    }
    sse(stream, "content_block_stop", r#"{"type":"content_block_stop","index":0}"#);
    sse(stream, "message_delta", &format!(
        r#"{{"type":"message_delta","delta":{{"stop_reason":"end_turn","stop_sequence":null}},"usage":{{"output_tokens":{}}}}}"#, n));
    sse(stream, "message_stop", r#"{"type":"message_stop"}"#);
}

fn build_prompt(messages: &[Msg], system: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(sys) = system {
        out.push_str(&format!("<|im_start|>system\n{}\n<|im_end|>\n", sys));
    }
    for m in messages {
        let role = match m.role.as_str() {
            "assistant" => "assistant",
            _ => "user",
        };
        out.push_str(&format!("<|im_start|>{}\n{}\n<|im_end|>\n", role, m.content));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn sse(stream: &mut TcpStream, event: &str, data: &str) {
    let _ = write!(stream, "event: {}\ndata: {}\n\n", event, data);
    let _ = stream.flush();
}
