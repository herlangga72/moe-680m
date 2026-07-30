use ash::{Device, vk};
use crate::gguf::ModelConfig;
use crate::memory::{Arena, Buffer};

pub struct KvCache {
    pub k_q4: Buffer,      // Q4_0: 18 bytes per 32 elements
    pub v_i8: Buffer,      // Int8: 1 byte per element
    pub seq_len: u32,
    pub max_seq: u32,
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
}

impl KvCache {
    pub fn new(
        config: &ModelConfig,
        max_seq: u32,
        arena: &mut Arena,
        device: &Device,
    ) -> crate::error::Result<Self> {
        let n_layers = config.n_layers;
        let n_kv_heads = config.n_heads_kv;
        let head_dim = config.head_dim;

        // K (Q4_0): n_layers × n_kv_heads × max_seq × head_dim elements
        // Q4_0 packs 32 elements into 18 bytes → (head_dim/32) * 18 bytes per (head, pos)
        let k_blocks_per_head = head_dim / 32;
        let k_bytes_per_head_pos = k_blocks_per_head * 18;
        let k_size = n_layers as u64 * n_kv_heads as u64 * max_seq as u64 * k_bytes_per_head_pos as u64;

        // V (Int8): n_layers × n_kv_heads × max_seq × head_dim bytes
        let v_size = n_layers as u64 * n_kv_heads as u64 * max_seq as u64 * head_dim as u64;

        let kv_usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let k_q4 = Buffer::new(device, k_size, kv_usage)?;
        let v_i8 = Buffer::new(device, v_size, kv_usage)?;

        arena.allocate("kv_cache_k", k_size)?;
        arena.bind_buffer("kv_cache_k", &k_q4)?;
        arena.allocate("kv_cache_v", v_size)?;
        arena.bind_buffer("kv_cache_v", &v_i8)?;

        Ok(Self {
            k_q4,
            v_i8,
            seq_len: 0,
            max_seq,
            n_layers,
            n_kv_heads,
            head_dim,
        })
    }

    /// Linear offset for K cache: [layer][head][pos][block]
    pub fn k_offset(&self, layer: u32, head: u32, pos: u32) -> u64 {
        let block_bytes = (self.head_dim / 32) as u64 * 18;
        ((layer * self.n_kv_heads + head) as u64 * self.max_seq as u64 + pos as u64)
            * block_bytes
    }

    /// Linear offset for V cache: [layer][head][pos]
    pub fn v_offset(&self, layer: u32, head: u32, pos: u32) -> u64 {
        ((layer * self.n_kv_heads + head) as u64 * self.max_seq as u64 + pos as u64)
            * self.head_dim as u64
    }

    pub fn advance(&mut self) -> crate::error::Result<u32> {
        if self.seq_len >= self.max_seq {
            return Err(crate::error::Error::KvCacheFull(
                self.seq_len as usize,
                self.max_seq as usize,
            ));
        }
        let pos = self.seq_len;
        self.seq_len += 1;
        Ok(pos)
    }

    pub fn position(&self) -> u32 {
        self.seq_len
    }
}
