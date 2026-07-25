// Simple TUI for moe-680m parameter setup.
// No deps — stdin/stdout only.

use std::io::{self, Write, BufRead};
use std::path::Path;

pub struct Config {
    pub model_path: Option<String>,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_k: u32,
    pub server_port: Option<u16>,
    pub debug: bool,
    pub vk_validation: bool,
    pub smoke: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: None,
            prompt: String::new(),
            max_tokens: 100,
            temperature: 1.0,
            top_k: 0,
            server_port: None,
            debug: false,
            vk_validation: false,
            smoke: false,
        }
    }
}

fn read_line() -> String {
    let mut s = String::new();
    io::stdin().lock().read_line(&mut s).ok();
    s.trim().to_string()
}

fn prompt(label: &str, default: &str) -> String {
    print!("  {} [{}]: ", label, default);
    io::stdout().flush().ok();
    let s = read_line();
    if s.is_empty() { default.to_string() } else { s }
}

fn prompt_bool(label: &str, current: bool) -> bool {
    let tag = if current { "ON" } else { "off" };
    let s = prompt(&format!("{} ({})", label, tag), if current { "y" } else { "n" });
    s.starts_with('y') || s.starts_with('Y')
}

fn prompt_u32(label: &str, default: u32) -> u32 {
    let s = prompt(label, &default.to_string());
    s.parse().unwrap_or(default)
}

fn prompt_f32(label: &str, default: f32) -> f32 {
    let s = prompt(label, &default.to_string());
    s.parse().unwrap_or(default)
}

fn prompt_path(label: &str, current: &Option<String>) -> Option<String> {
    let def = current.as_deref().unwrap_or("");
    let s = prompt(label, def);
    if s.is_empty() { current.clone() } else if Path::new(&s).exists() { Some(s) }
    else { eprintln!("  File not found: {}", s); current.clone() }
}

fn clear_screen() {
    print!("\x1B[2J\x1B[H");
    io::stdout().flush().ok();
}

pub fn run() -> Option<Config> {
    let mut cfg = Config::default();

    clear_screen();
    loop {
        let prompt_preview = if cfg.prompt.len() > 48 {
            format!("{}..", &cfg.prompt[..48])
        } else if cfg.prompt.is_empty() {
            "(empty)".to_string()
        } else { cfg.prompt.clone() };

        println!(
"┌─────────────────────────────────────────────┐
│           moe-680m Configuration            │
├─────────────────────────────────────────────┤
│ Model                                       │
│  1. Model path:  {}
│  2. Prompt:      {}
│ Sampling                                    │
│  3. Temperature: {:.1}
│  4. Top-K:       {}
│  5. Max tokens:  {}
│ Server                                      │
│  6. Server mode: {}
│  7. Server port: {}
│ Debug                                       │
│  8. Debug:       {}
│  9. Vulkan validation: {}
├─────────────────────────────────────────────┤
│  r. Run inference                           │
│  s. Smoke test (Vulkan)                     │
│  q. Quit                                    │
└─────────────────────────────────────────────┘",
            cfg.model_path.as_deref().unwrap_or("(not set)"),
            prompt_preview,
            cfg.temperature, cfg.top_k, cfg.max_tokens,
            if cfg.server_port.is_some() { "ON" } else { "off" },
            cfg.server_port.unwrap_or(8080),
            if cfg.debug { "ON" } else { "off" },
            if cfg.vk_validation { "ON" } else { "off" });

        print!("  Choice: ");
        io::stdout().flush().ok();

        match read_line().to_lowercase().as_str() {
            "1" => cfg.model_path = prompt_path("Model path", &cfg.model_path),
            "2" => cfg.prompt = prompt("Prompt text", &cfg.prompt),
            "3" => cfg.temperature = prompt_f32("Temperature", cfg.temperature),
            "4" => cfg.top_k = prompt_u32("Top-K", cfg.top_k),
            "5" => cfg.max_tokens = prompt_u32("Max tokens", cfg.max_tokens),
            "6" => {
                let on = prompt_bool("Server mode", cfg.server_port.is_some());
                cfg.server_port = if on { Some(cfg.server_port.unwrap_or(8080)) } else { None };
            }
            "7" => cfg.server_port = Some(prompt_u32("Server port", cfg.server_port.unwrap_or(8080) as u32) as u16),
            "8" => cfg.debug = prompt_bool("Debug", cfg.debug),
            "9" => cfg.vk_validation = prompt_bool("Vulkan validation", cfg.vk_validation),
            "r" => {
                if cfg.model_path.is_none() {
                    eprintln!("  Set model path first (option 1)");
                    continue;
                }
                if cfg.prompt.is_empty() {
                    cfg.prompt = prompt("Prompt text", "");
                    if cfg.prompt.is_empty() {
                        eprintln!("  Prompt cannot be empty");
                        continue;
                    }
                }
                return Some(cfg);
            }
            "s" => {
                cfg.smoke = true;
                return Some(cfg);
            }
            "q" => return None,
            "" => continue,
            _ => eprintln!("  Unknown"),
        }
    }
}
