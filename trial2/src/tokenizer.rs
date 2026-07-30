use crate::error::{Error, Result};

/// Qwen-style tokenizer backed by gigatoken.
///
/// Stores special-token IDs for the `<|im_start|>` / `<|im_end|>` chat format
/// used by Qwen models, as well as the end-of-sequence token.
pub struct Tokenizer {
    inner: gigatoken::Tokenizer,
    eos_token: u32,
    im_start: u32,
    im_end: u32,
}

impl Tokenizer {
    /// Load a `tokenizer.json` from the model directory.
    ///
    /// The model directory is expected to contain a HuggingFace‑compatible
    /// `tokenizer.json` file alongside the GGUF weights.
    pub fn new(model_dir: &std::path::Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::Tokenizer(
                "tokenizer.json not found in model directory".into(),
            ));
        }
        let inner = gigatoken::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("gigatoken: {}", e)))?;

        let eos_token = inner
            .token_to_id("<|im_end|>")
            .unwrap_or(inner.eos_token_id());
        let im_start = inner.token_to_id("<|im_start|>").unwrap_or(0);
        let im_end = inner.token_to_id("<|im_end|>").unwrap_or(eos_token);

        Ok(Self {
            inner,
            eos_token,
            im_start,
            im_end,
        })
    }

    /// Encode a text string into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text).ids()
    }

    /// Decode token IDs back into a string.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens)
            .map_err(|e| Error::Tokenizer(format!("decode: {}", e)))
    }

    /// The end‑of‑sequence token ID (`<|im_end|>`).
    pub fn eos_token(&self) -> u32 {
        self.eos_token
    }

    /// The `<|im_start|>` special token ID.
    pub fn im_start(&self) -> u32 {
        self.im_start
    }

    /// The `<|im_end|>` special token ID.
    pub fn im_end(&self) -> u32 {
        self.im_end
    }
}
