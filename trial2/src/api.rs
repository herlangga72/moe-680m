use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::{Method, StatusCode, Request, Response};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use crate::chat_template::{ChatTemplate, Message};
use crate::engine::Engine;
use crate::error::Result;
use crate::tokenizer::Tokenizer;

// ===========================================================================
// Helpers
// ===========================================================================

/// Wrap a string into a boxed HTTP body (never fails).
fn box_body(data: String) -> BoxBody<Bytes, Infallible> {
    Full::new(Bytes::from(data)).boxed()
}

/// Build a JSON error response matching the Anthropic error format:
/// `{"type":"error","error":{"type":"...","message":"..."}}`
fn error_response(
    status: u16,
    err_type: &str,
    message: &str,
) -> Response<BoxBody<Bytes, Infallible>> {
    let body = serde_json::json!({
        "type": "error",
        "error": { "type": err_type, "message": message }
    });
    Response::builder()
        .status(
            StatusCode::from_u16(status)
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        )
        .body(box_body(body.to_string()))
        .unwrap()
}

/// Format a single SSE event frame: `event: {event}\ndata: {data}\n\n`
fn sse_event(event: &str, data: &str) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

// ===========================================================================
// Anthropic Messages API request type
// ===========================================================================

#[derive(serde::Deserialize)]
struct AnthropicRequest {
    #[allow(dead_code)]
    model: String,
    messages: Vec<Message>,
    system: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    tools: Vec<serde_json::Value>,
    max_tokens: u32,
    #[serde(default)]
    stream: Option<bool>,
    #[allow(dead_code)]
    #[serde(default)]
    temperature: Option<f32>,
    #[allow(dead_code)]
    #[serde(default)]
    top_p: Option<f32>,
    #[allow(dead_code)]
    #[serde(default)]
    top_k: Option<u32>,
    #[allow(dead_code)]
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
}

// ===========================================================================
// Server entry point
// ===========================================================================

/// Run the Anthropic Messages API server.
///
/// Spawns one task per connection on the current tokio runtime.  The caller
/// should use a `current_thread` runtime and call this with `block_on`.
pub async fn serve(
    addr: &str,
    engine: Arc<Mutex<Engine>>,
    tokenizer: Arc<Tokenizer>,
) -> Result<()> {
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|e| crate::error::Error::Api(format!("bind {}: {}", addr, e)))?;
    println!("Server listening on http://{}", addr);

    loop {
        let (stream, _) = listener
            .accept()
            .await
            .map_err(|e| crate::error::Error::Api(format!("accept: {}", e)))?;
        let eng = engine.clone();
        let tok = tokenizer.clone();

        tokio::spawn(async move {
            let service = service_fn(move |req| {
                let eng = eng.clone();
                let tok = tok.clone();
                async move { handle_request(req, eng, tok).await }
            });
            if let Err(e) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .await
            {
                eprintln!("Connection error: {}", e);
            }
        });
    }
}

// ===========================================================================
// Request router
// ===========================================================================

async fn handle_request(
    req: Request<Incoming>,
    engine: Arc<Mutex<Engine>>,
    tokenizer: Arc<Tokenizer>,
) -> std::result::Result<Response<BoxBody<Bytes, Infallible>>, Infallible> {
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/v1/messages") => {
            Ok(handle_messages(req, engine, tokenizer).await)
        }
        (&Method::POST, "/v1/messages/count_tokens") => {
            Ok(handle_count_tokens(req, tokenizer).await)
        }
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(box_body("Not Found".into()))
            .unwrap()),
    }
}

// ===========================================================================
// POST /v1/messages — SSE streaming or non-streaming
// ===========================================================================

async fn handle_messages(
    req: Request<Incoming>,
    engine: Arc<Mutex<Engine>>,
    tokenizer: Arc<Tokenizer>,
) -> Response<BoxBody<Bytes, Infallible>> {
    // ---- Parse request body via http_body_util::BodyExt::collect ----
    let collected = match req.into_body().collect().await {
        Ok(c) => c,
        Err(e) => {
            return error_response(400, "invalid_request_error", &e.to_string());
        }
    };
    let body_bytes = collected.to_bytes();
    let req: AnthropicRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(400, "invalid_request_error", &e.to_string());
        }
    };

    // Apply chat template to produce input token sequence
    let input_tokens =
        ChatTemplate::apply(&tokenizer, &req.messages, req.system.as_deref());
    let input_count = input_tokens.len() as u32;

    // =======================================================================
    // Streaming response (Server-Sent Events)
    // =======================================================================
    if req.stream.unwrap_or(false) {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);
        let eng = engine.clone();
        let tok = tokenizer.clone();
        let max_tokens = req.max_tokens;
        let eos_id = tokenizer.eos_token();

        tokio::spawn(async move {
            // --- Phase 1: Inference (synchronous, holds Mutex lock) ---
            // All error sends use try_send to avoid holding the non-Send
            // MutexGuard across an .await point.
            let generated = {
                let mut eng_guard = match eng.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        let _ = tx.try_send(sse_event(
                            "error",
                            r#"{"type":"error","message":"engine lock poisoned"}"#,
                        ));
                        return;
                    }
                };

                let first = match eng_guard.prefill(&input_tokens) {
                    Ok(t) => t,
                    Err(e) => {
                        let err = serde_json::json!({
                            "type": "error",
                            "message": e.to_string(),
                        });
                        let _ = tx.try_send(sse_event("error", &err.to_string()));
                        return;
                    }
                };

                let mut tokens = vec![first];
                let mut current = first;
                for _ in 0..max_tokens {
                    if current == eos_id {
                        break;
                    }
                    match eng_guard.decode(current) {
                        Ok((next, _)) => {
                            tokens.push(next);
                            current = next;
                        }
                        Err(_) => break,
                    }
                }
                tokens
            }; // MutexGuard<Engine> dropped here

            // --- Phase 2: SSE event emission (async, no lock) ---

            // message_start
            let _ = tx
                .send(sse_event(
                    "message_start",
                    &serde_json::json!({
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": "qwen-3.6-35b-a3b-mtp",
                            "stop_reason": null,
                            "stop_sequence": null,
                            "usage": {
                                "input_tokens": input_count,
                                "output_tokens": 0
                            }
                        }
                    })
                    .to_string(),
                ))
                .await;

            // content_block_start
            let _ = tx
                .send(sse_event(
                    "content_block_start",
                    &serde_json::json!({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": { "type": "text" }
                    })
                    .to_string(),
                ))
                .await;

            // Token deltas
            for &token in &generated {
                let text = match tok.decode(&[token]) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": text }
                });
                let _ = tx
                    .send(sse_event("content_block_delta", &delta.to_string()))
                    .await;
                if token == eos_id {
                    break;
                }
            }

            // content_block_stop
            let _ = tx
                .send(sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ))
                .await;

            // message_delta
            let _ = tx
                .send(sse_event(
                    "message_delta",
                    &serde_json::json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": "end_turn",
                            "stop_sequence": null
                        },
                        "usage": {
                            "input_tokens": input_count,
                            "output_tokens": generated.len()
                        }
                    })
                    .to_string(),
                ))
                .await;

            // message_stop
            let _ = tx
                .send(sse_event("message_stop", r#"{"type":"message_stop"}"#))
                .await;
        });

        // Collect all SSE events from the spawned task (buffered SSE).
        // A true streaming implementation would use StreamBody with a
        // streaming body type; this simpler approach sends the complete
        // event stream as one response body.
        let mut all_events = Vec::new();
        while let Some(event) = rx.recv().await {
            all_events.push(event);
        }
        let body = all_events.join("");

        return Response::builder()
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .body(box_body(body))
            .unwrap();
    }

    // =======================================================================
    // Non-streaming response (not yet implemented)
    // =======================================================================
    error_response(501, "not_implemented", "only streaming is supported")
}

// ===========================================================================
// POST /v1/messages/count_tokens
// ===========================================================================

async fn handle_count_tokens(
    req: Request<Incoming>,
    tokenizer: Arc<Tokenizer>,
) -> Response<BoxBody<Bytes, Infallible>> {
    let collected = match req.into_body().collect().await {
        Ok(c) => c,
        Err(e) => {
            return error_response(400, "invalid_request_error", &e.to_string());
        }
    };
    let body_bytes = collected.to_bytes();
    let req: AnthropicRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(400, "invalid_request_error", &e.to_string());
        }
    };
    let tokens = ChatTemplate::apply(&tokenizer, &req.messages, req.system.as_deref());
    let body = serde_json::json!({ "input_tokens": tokens.len() });
    Response::new(box_body(body.to_string()))
}
