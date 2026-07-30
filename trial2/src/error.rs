use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Vulkan: {0}")]
    Vulkan(#[from] ash::vk::Result),
    #[error("GGUF parse: {0}")]
    Gguf(String),
    #[error("API: {0}")]
    Api(String),
    #[error("Tokenizer: {0}")]
    Tokenizer(String),
    #[error("Model architecture mismatch: {0}")]
    Architecture(String),
    #[error("Out of memory: needed {needed}, available {available}")]
    OutOfMemory { needed: u64, available: u64 },
    #[error("KV cache full: {0} tokens, max {1}")]
    KvCacheFull(usize, usize),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
