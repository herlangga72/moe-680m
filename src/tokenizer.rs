// ── Tokenizer ──
// With `--features gigatoken`: uses Gigatoken (exact tiktoken BPE).
// Without: uses minimal BPE fallback (works, no deps).

pub struct TokenizerData {
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub merges: Vec<String>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub model_type: String,
}

impl TokenizerData {
    pub fn from_gguf_meta(
        get_str: &dyn Fn(&str) -> Option<String>,
        get_arr: &dyn Fn(&str) -> Option<Vec<String>>,
        get_float_arr: &dyn Fn(&str) -> Option<Vec<f32>>,
        get_int: &dyn Fn(&str) -> Option<u32>,
    ) -> Result<Self, String> {
        Ok(TokenizerData {
            model_type: get_str("tokenizer.ggml.model").unwrap_or_else(|| "gpt2".into()),
            tokens: get_arr("tokenizer.ggml.tokens").ok_or("Missing tokenizer.ggml.tokens")?,
            scores: get_float_arr("tokenizer.ggml.scores").unwrap_or_default(),
            merges: get_arr("tokenizer.ggml.merges").unwrap_or_default(),
            bos_id: get_int("tokenizer.ggml.bos_token_id").unwrap_or(0),
            eos_id: get_int("tokenizer.ggml.eos_token_id").unwrap_or(0),
        })
    }
}

// ── Gigatoken backend (exact tiktoken BPE) ──
#[cfg(feature = "gigatoken")]
pub struct Tokenizer {
    inner: gigatoken::Tokenizer,
    pub bos_id: u32,
    pub eos_id: u32,
}

#[cfg(feature = "gigatoken")]
impl Tokenizer {
    pub fn from_data(data: &TokenizerData) -> Result<Self, String> {
        use gigatoken::Tokenizer as Gt;

        // Build from GGUF data
        let mut builder = Gt::builder("gpt2".into())
            .map_err(|e| format!("Gigatoken builder: {}", e))?;

        // Add tokens (byte-level BPE expects byte_fallback)
        for (i, token) in data.tokens.iter().enumerate() {
            builder = builder
                .add_token(token, i as u32, data.scores.get(i).copied().unwrap_or(0.0))
                .map_err(|e| format!("Add token {}: {}", i, e))?;
        }

        // Add merges (format: "A B")
        for merge in &data.merges {
            builder = builder
                .add_merge(merge)
                .map_err(|e| format!("Add merge: {}", e))?;
        }

        let inner = builder
            .build()
            .map_err(|e| format!("Gigatoken build: {}", e))?;

        Ok(Tokenizer { inner, bos_id: data.bos_id, eos_id: data.eos_id })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text, false).unwrap_or_default()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, false).unwrap_or_default()
    }

    pub fn vocab_size(&self) -> usize { self.inner.vocab_size() }
}

// ── Minimal BPE fallback (no deps) ──
#[cfg(not(feature = "gigatoken"))]
pub struct Tokenizer {
    id_to_token: Vec<String>,
    token_to_id: std::collections::HashMap<String, u32>,
    merges: std::collections::HashMap<(u32, u32), u32>,
    pub bos_id: u32,
    pub eos_id: u32,
}

#[cfg(not(feature = "gigatoken"))]
impl Tokenizer {
    pub fn from_data(data: &TokenizerData) -> Result<Self, String> {
        let mut id_to_token = Vec::with_capacity(data.tokens.len());
        let mut token_to_id = std::collections::HashMap::with_capacity(data.tokens.len());
        for (i, t) in data.tokens.iter().enumerate() {
            id_to_token.push(t.clone());
            token_to_id.insert(t.clone(), i as u32);
        }

        let mut merges = std::collections::HashMap::new();
        for (i, m) in data.merges.iter().enumerate() {
            let p: Vec<&str> = m.splitn(2, ' ').collect();
            if p.len() == 2 {
                if let (Some(&l), Some(&r)) = (token_to_id.get(p[0]), token_to_id.get(p[1])) {
                    let merged = format!("{}{}", p[0], p[1]);
                    let mid = token_to_id.entry(merged).or_insert(data.tokens.len() as u32 + i as u32);
                    merges.insert((l, r), *mid);
                }
            }
        }

        Ok(Tokenizer { id_to_token, token_to_id, merges, bos_id: data.bos_id, eos_id: data.eos_id })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            if !ids.is_empty() {
                if let Some(&s) = self.token_to_id.get(" ") { ids.push(s); }
            }
            if let Some(&id) = self.token_to_id.get(word) { ids.push(id); continue; }
            let mut bpe: Vec<u32> = Vec::new();
            for ch in word.chars() {
                if let Some(&id) = self.token_to_id.get(&ch.to_string()) { bpe.push(id); }
            }
            loop {
                let mut done = true;
                for i in 0..bpe.len().saturating_sub(1) {
                    if let Some(&mid) = self.merges.get(&(bpe[i], bpe[i+1])) {
                        bpe[i] = mid; bpe.remove(i+1); done = false; break;
                    }
                }
                if done { break; }
            }
            ids.extend(bpe);
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(s) = self.id_to_token.get(id as usize) {
                if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
                    if let Ok(b) = u8::from_str_radix(&s[3..5], 16) { out.push(b as char); continue; }
                }
                out.push_str(s);
            }
        }
        out
    }

    pub fn vocab_size(&self) -> usize { self.id_to_token.len() }
}
