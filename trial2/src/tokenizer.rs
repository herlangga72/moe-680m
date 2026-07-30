use crate::error::{Error, Result};

/// Qwen tokenizer backed by the `tokenizers` crate (HuggingFace).
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    eos_token: u32,
    im_start: u32,
    im_end: u32,
}

impl Tokenizer {
    pub fn new(model_dir: &std::path::Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        if !path.exists() {
            return Err(Error::Tokenizer("tokenizer.json not found".into()));
        }
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| Error::Tokenizer(format!("{}", e)))?;

        let eos = inner
            .token_to_id("<|im_end|>")
            .or_else(|| inner.token_to_id("<|endoftext|>"))
            .unwrap_or(0);
        let im_start = inner.token_to_id("<|im_start|>").unwrap_or(0);
        let im_end = inner.token_to_id("<|im_end|>").unwrap_or(eos);

        Ok(Self { inner, eos_token: eos, im_start, im_end })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| Error::Tokenizer(format!("{}", e)))
    }

    pub fn eos_token(&self) -> u32 { self.eos_token }
    pub fn im_start(&self) -> u32  { self.im_start }
    pub fn im_end(&self) -> u32    { self.im_end }
}
