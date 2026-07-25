// Interactive TUI for moe-680m parameter setup.
// Arrow key navigation, field selection, inline editing.
// Uses libc for raw terminal mode (already a dep).

use std::io::{self, Write, Read};
use std::path::Path;
use libc::{tcgetattr, tcsetattr, TCSANOW, termios, ECHO, ICANON, VMIN, VTIME, STDIN_FILENO};

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
    pub chat: bool,
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
            chat: false,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy)]
enum Key {
    Up, Down, Enter, Esc, Char(u8), Ctrl(char),
}

// ── Raw terminal mode via libc ──

fn set_raw(orig: &mut termios) {
    unsafe {
        tcgetattr(STDIN_FILENO, orig);
        let mut raw = *orig;
        raw.c_lflag &= !(ICANON | ECHO);
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    }
}

fn restore(orig: &termios) {
    unsafe { tcsetattr(STDIN_FILENO, TCSANOW, orig); }
}

// ── Key reader with arrow key support ──

fn read_key() -> Key {
    let mut buf = [0u8; 1];
    if io::stdin().read(&mut buf).ok() != Some(1) { return Key::Esc; }
    match buf[0] {
        0x1B => {
            // Check for escape sequence with 50ms timeout
            let mut fds = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
            let r = unsafe { libc::poll(&mut fds, 1, 50) };
            if r > 0 {
                let mut seq = [0u8; 2];
                if io::stdin().read(&mut seq[..1]).ok() == Some(1) && seq[0] == b'[' {
                    if io::stdin().read(&mut seq[1..2]).ok() == Some(1) {
                        return match seq[1] { b'A' => Key::Up, b'B' => Key::Down, _ => Key::Esc };
                    }
                }
            }
            Key::Esc
        }
        0x0A | 0x0D => Key::Enter,
        3 => Key::Ctrl('c'),
        c => Key::Char(c),
    }
}

// ── Field types ──

#[derive(Clone, Copy, PartialEq)]
enum Field { Model, Prompt, Temperature, TopK, MaxTokens, Server, Port, Debug, VkValidation, Chat }

const FIELDS: &[Field] = &[
    Field::Model, Field::Prompt, Field::Temperature, Field::TopK,
    Field::MaxTokens, Field::Server, Field::Port, Field::Debug, Field::VkValidation, Field::Chat,
];

// ── Edit a field value interactively ──

fn edit_field(label: &str, current: &str) -> String {
    // restore cooked mode to read a line
    println!("\x1B[K"); // clear status line
    print!("  {}: ", label);
    io::stdout().flush().ok();
    let mut s = String::new();
    io::stdin().read_line(&mut s).ok();
    let trimmed = s.trim().to_string();
    if trimmed.is_empty() { current.to_string() } else { trimmed }
}

// ── Form rendering ──

fn render(cfg: &Config, sel: usize) {
    let mv = |row: u16| print!("\x1B[{}H", row);
    let hi = |s: &str, on: bool| if on { print!("\x1B[7m{}\x1B[0m", s) } else { print!("{}", s) };

    // box top
    mv(1); print!("┌────────────────────────────────────────────────┐");
    mv(2); print!("│          moe-680m Configuration               │");
    mv(3); print!("├────────────────────────────────────────────────┤");

    // Model
    mv(4); print!("│ Model                                          │");
    mv(5); println!("\x1B[K"); mv(5);
    let model = cfg.model_path.as_deref().unwrap_or("(not set)");
    print!("  │ "); hi("▶", sel == 0); print!(" Model path:  "); hi(model, sel == 0);
    println!();

    mv(6); println!("\x1B[K"); mv(6);
    let preview = if cfg.prompt.len() > 47 { format!("{}..", &cfg.prompt[..47]) }
        else if cfg.prompt.is_empty() { "(empty)".into() } else { cfg.prompt.clone() };
    print!("  │ "); hi("▶", sel == 1); print!(" Prompt:      "); hi(&preview, sel == 1);
    println!();

    // Sampling
    mv(7); print!("│ Sampling                                       │");
    mv(8); println!("\x1B[K"); mv(8);
    print!("  │ "); hi("▶", sel == 2); print!(" Temperature: {:.1}", cfg.temperature);
    // clear to eol
    mv(8); print!("  │ "); hi("▶", sel == 2); print!(" Temperature: ");
    if sel == 2 { print!("\x1B[7m{:.1}\x1B[0m", cfg.temperature); } else { print!("{:.1}", cfg.temperature); }
    println!();

    mv(9); println!("\x1B[K"); mv(9);
    print!("  │ "); hi("▶", sel == 3); print!(" Top-K:       ");
    if sel == 3 { print!("\x1B[7m{}\x1B[0m", cfg.top_k); } else { print!("{}", cfg.top_k); }
    println!();

    mv(10); println!("\x1B[K"); mv(10);
    print!("  │ "); hi("▶", sel == 4); print!(" Max tokens:  ");
    if sel == 4 { print!("\x1B[7m{}\x1B[0m", cfg.max_tokens); } else { print!("{}", cfg.max_tokens); }
    println!();

    // Server
    mv(11); print!("│ Server                                         │");
    mv(12); println!("\x1B[K"); mv(12);
    let smode = if cfg.server_port.is_some() { "ON " } else { "OFF" };
    print!("  │ "); hi("▶", sel == 5); print!(" Server mode: ");
    if sel == 5 { print!("\x1B[7m{}\x1B[0m", smode); } else { print!("{}", smode); }
    println!();

    mv(13); println!("\x1B[K"); mv(13);
    print!("  │ "); hi("▶", sel == 6); print!(" Server port: ");
    if sel == 6 { print!("\x1B[7m{}\x1B[0m", cfg.server_port.unwrap_or(8080)); }
    else { print!("{}", cfg.server_port.unwrap_or(8080)); }
    println!();

    // Debug
    mv(14); print!("│ Debug                                          │");
    mv(15); println!("\x1B[K"); mv(15);
    print!("  │ "); hi("▶", sel == 7); print!(" Debug:        ");
    if sel == 7 { print!("\x1B[7m{}\x1B[0m", if cfg.debug { "ON" } else { "OFF" }); }
    else { print!("{}", if cfg.debug { "ON" } else { "OFF" }); }
    println!();

    mv(16); println!("\x1B[K"); mv(16);
    print!("  │ "); hi("▶", sel == 8); print!(" Vulkan valid: ");
    if sel == 8 { print!("\x1B[7m{}\x1B[0m", if cfg.vk_validation { "ON" } else { "OFF" }); }
    else { print!("{}", if cfg.vk_validation { "ON" } else { "OFF" }); }
    println!();

    // footer
    mv(17); println!("\x1B[K"); mv(17);
    print!("  │ "); hi("▶", sel == 9); print!(" Interactive chat: ");
    if sel == 9 { print!("\x1B[7m{}\x1B[0m", if cfg.chat { "ON " } else { "OFF" }); }
    else { print!("{}", if cfg.chat { "ON " } else { "OFF" }); }
    println!();

    mv(18); print!("├────────────────────────────────────────────────┤");
    mv(19); print!("│  ↑↓ navigate  ↵ edit  r run  s smoke  q quit  │");
    mv(20); print!("└────────────────────────────────────────────────┘");
    mv(21); print!("\x1B[K");
    io::stdout().flush().ok();
}

pub fn run() -> Option<Config> {
    let mut cfg = Config::default();
    let mut sel = 0usize;
    let mut orig = unsafe { std::mem::zeroed::<termios>() };

    set_raw(&mut orig);
    print!("\x1B[?25l"); // hide cursor
    print!("\x1B[2J\x1B[H");
    io::stdout().flush().ok();

    loop {
        render(&cfg, sel);
        match read_key() {
            Key::Up if sel > 0 => sel -= 1,
            Key::Down if sel < FIELDS.len() - 1 => sel += 1,
            Key::Enter => {
                match FIELDS[sel] {
                    Field::Model => {
                        let s = edit_field("Model path", cfg.model_path.as_deref().unwrap_or(""));
                        if !s.is_empty() && Path::new(&s).exists() { cfg.model_path = Some(s); }
                        else if !s.is_empty() { eprintln!("\n  Not found: {}", s); }
                    }
                    Field::Prompt => {
                        let s = edit_field("Prompt text", &cfg.prompt);
                        cfg.prompt = s;
                    }
                    Field::Temperature => {
                        let s = edit_field("Temperature", &format!("{:.1}", cfg.temperature));
                        cfg.temperature = s.parse().unwrap_or(cfg.temperature);
                    }
                    Field::TopK => {
                        let s = edit_field("Top-K", &cfg.top_k.to_string());
                        cfg.top_k = s.parse().unwrap_or(cfg.top_k);
                    }
                    Field::MaxTokens => {
                        let s = edit_field("Max tokens", &cfg.max_tokens.to_string());
                        cfg.max_tokens = s.parse().unwrap_or(cfg.max_tokens);
                    }
                    Field::Server => {
                        let on = cfg.server_port.is_some();
                        cfg.server_port = if !on { Some(cfg.server_port.unwrap_or(8080)) } else { None };
                    }
                    Field::Port => {
                        let s = edit_field("Server port", &cfg.server_port.unwrap_or(8080).to_string());
                        cfg.server_port = Some(s.parse().unwrap_or(cfg.server_port.unwrap_or(8080)) as u16);
                    }
                    Field::Debug => { cfg.debug = !cfg.debug; }
                    Field::VkValidation => { cfg.vk_validation = !cfg.vk_validation; }
                    Field::Chat => { cfg.chat = !cfg.chat; }
                }
                print!("\x1B[2J\x1B[H");
                io::stdout().flush().ok();
            }
            Key::Char(b'r') => {
                if cfg.model_path.is_none() {
                    // show error inline at bottom
                    mv_status(20, &format!("Set model path first"));
                    continue;
                }
                if cfg.prompt.is_empty() {
                    let s = edit_field("Prompt text", "");
                    print!("\x1B[2J\x1B[H");
                    cfg.prompt = s;
                    if cfg.prompt.is_empty() { continue; }
                }
                break;
            }
            Key::Char(b's') => { cfg.smoke = true; break; }
            Key::Char(b'q') | Key::Esc | Key::Ctrl('c') => { restore(&mut orig); return None; }
            _ => {}
        }
    }

    print!("\x1B[?25h"); // show cursor
    restore(&mut orig);
    print!("\x1B[2J\x1B[H");
    io::stdout().flush().ok();
    Some(cfg)
}

fn mv_status(row: u16, msg: &str) {
    print!("\x1B[{}H\x1B[K  {}", row, msg);
    io::stdout().flush().ok();
}
