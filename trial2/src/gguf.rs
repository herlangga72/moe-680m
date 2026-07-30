use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use memmap2::Mmap;
use crate::error::{Error, Result};

// GGUF value types (subset we need)
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    ArrayU32(Vec<u32>),
    ArrayF32(Vec<f32>),
    ArrayI64(Vec<i64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgufDtype {
    F32,    // 0
    F16,    // 1
    Q4_0,   // 2
    Q8_0,   // 8
    IQ4_XS, // custom
    Unknown(u32),
}

impl GgufDtype {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            _ => Self::Unknown(raw),
        }
    }

    pub fn type_size(&self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,  // 32 elements in 18 bytes
            Self::Q8_0 => 34,  // 32 elements in 34 bytes
            Self::IQ4_XS => 17, // 32 4-bit values + scale
            Self::Unknown(_) => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufDtype,
    pub offset: u64,
    pub n_elements: u64,
    pub size_bytes: u64,
}

pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    pub data: Mmap,
}

pub struct ModelConfig {
    pub n_layers: u32,
    pub hidden_dim: u32,
    pub n_heads_q: u32,
    pub n_heads_kv: u32,
    pub head_dim: u32,
    pub ffn_intermediate: u32,
    pub n_experts: u32,
    pub n_active_experts: u32,
    pub n_shared_experts: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub rope_theta: f32,
    pub rope_type: String,
    pub n_mtp_modules: u32,
    pub mtp_depth: u32,
    pub eps: f32,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let data = unsafe { Mmap::map(&file)? };

        if data.len() < 24 {
            return Err(Error::Gguf("file too small for GGUF header".into()));
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != 0x46475547 { // "GGUF"
            return Err(Error::Gguf("bad magic — not a GGUF file".into()));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != 3 {
            return Err(Error::Gguf(format!("unsupported GGUF version {}", version)));
        }

        let n_tensors = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let n_kv = u64::from_le_bytes(data[16..24].try_into().unwrap());
        let mut pos: usize = 24;

        // Parse key-value metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = Self::read_string(&data, &mut pos)?;
            let val_type = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let val = Self::read_value(&data, &mut pos, val_type)?;
            metadata.insert(key, val);
        }

        // Parse tensor infos
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = Self::read_string(&data, &mut pos)?;
            let n_dims = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let mut shape = Vec::with_capacity(n_dims as usize);
            let mut n_elements: u64 = 1;
            for _ in 0..n_dims {
                let d = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
                pos += 8;
                shape.push(d);
                n_elements = n_elements.saturating_mul(d);
            }
            let dtype_raw = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let dtype = GgufDtype::from_raw(dtype_raw);
            let offset = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
            pos += 8;
            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                offset,
                n_elements,
                size_bytes: 0, // computed after parse
            });
        }

        // Compute sizes (contiguous storage)
        for i in 0..tensors.len() {
            let this_offset = tensors[i].offset;
            let next_offset = if i + 1 < tensors.len() {
                tensors[i + 1].offset
            } else {
                data.len() as u64
            };
            tensors[i].size_bytes = next_offset - this_offset;
        }

        Ok(Self { metadata, tensors, data })
    }

    pub fn model_config(&self) -> Result<ModelConfig> {
        let get_u32 = |key: &str| -> Result<u32> {
            match self.metadata.get(key) {
                Some(GgufValue::U32(v)) => Ok(*v),
                _ => Err(Error::Architecture(format!("missing or wrong type: {}", key))),
            }
        };
        let get_f32 = |key: &str| -> Result<f32> {
            match self.metadata.get(key) {
                Some(GgufValue::F32(v)) => Ok(*v),
                _ => Err(Error::Architecture(format!("missing or wrong type: {}", key))),
            }
        };
        let get_str = |key: &str| -> Result<String> {
            match self.metadata.get(key) {
                Some(GgufValue::String(v)) => Ok(v.clone()),
                _ => Err(Error::Architecture(format!("missing or wrong type: {}", key))),
            }
        };

        let n_heads_q = get_u32("qwen3.attention.head_count")?;
        let n_heads_kv = get_u32("qwen3.attention.head_count_kv")?;
        let hidden_dim = get_u32("qwen3.embedding_length")?;
        let head_dim = hidden_dim / n_heads_q;

        Ok(ModelConfig {
            n_layers: get_u32("qwen3.block_count")?,
            hidden_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            ffn_intermediate: get_u32("qwen3.feed_forward_length")?,
            n_experts: get_u32("qwen3.expert_count").unwrap_or(1),
            n_active_experts: get_u32("qwen3.expert_used_count").unwrap_or(1),
            n_shared_experts: get_u32("qwen3.expert_shared_count").unwrap_or(0),
            vocab_size: get_u32("qwen3.vocab_size")?,
            max_seq_len: get_u32("qwen3.context_length").unwrap_or(32768),
            rope_theta: get_f32("qwen3.rope.freq_base").unwrap_or(1_000_000.0),
            rope_type: get_str("qwen3.rope.type").unwrap_or_else(|_| "default".into()),
            n_mtp_modules: get_u32("qwen3.mtp.module_count").unwrap_or(0),
            mtp_depth: get_u32("qwen3.mtp.depth").unwrap_or(0),
            eps: get_f32("qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
        })
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_data(&self, tensor: &TensorInfo) -> &[u8] {
        let start = tensor.offset as usize;
        &self.data[start..start + tensor.size_bytes as usize]
    }

    fn read_string(data: &[u8], pos: &mut usize) -> Result<String> {
        let end = pos.checked_add(8).ok_or_else(|| {
            Error::Gguf("string length prefix position overflow".into())
        })?;
        if end > data.len() {
            return Err(Error::Gguf("truncated GGUF: string length prefix out of bounds".into()));
        }
        let len = u64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap()) as usize;
        *pos += 8;
        let str_end = pos.checked_add(len).ok_or_else(|| {
            Error::Gguf("string data length overflow".into())
        })?;
        if str_end > data.len() {
            return Err(Error::Gguf("truncated GGUF: string data out of bounds".into()));
        }
        let s = String::from_utf8_lossy(&data[*pos..*pos+len]).to_string();
        *pos += len;
        Ok(s)
    }

    fn read_value(data: &[u8], pos: &mut usize, val_type: u32) -> Result<GgufValue> {
        match val_type {
            1 => {
                if *pos >= data.len() {
                    return Err(Error::Gguf("truncated GGUF: U8 value out of bounds".into()));
                }
                let v = GgufValue::U8(data[*pos]);
                *pos += 1;
                Ok(v)
            }
            4 => {
                if *pos + 4 > data.len() {
                    return Err(Error::Gguf("truncated GGUF: U32 value out of bounds".into()));
                }
                let v = GgufValue::U32(u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap()));
                *pos += 4;
                Ok(v)
            }
            5 => {
                if *pos + 8 > data.len() {
                    return Err(Error::Gguf("truncated GGUF: U64 value out of bounds".into()));
                }
                let v = GgufValue::U64(u64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap()));
                *pos += 8;
                Ok(v)
            }
            7 => {
                if *pos + 4 > data.len() {
                    return Err(Error::Gguf("truncated GGUF: I32 value out of bounds".into()));
                }
                let v = GgufValue::I32(i32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap()));
                *pos += 4;
                Ok(v)
            }
            8 => {
                if *pos + 8 > data.len() {
                    return Err(Error::Gguf("truncated GGUF: I64 value out of bounds".into()));
                }
                let v = GgufValue::I64(i64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap()));
                *pos += 8;
                Ok(v)
            }
            22 => {
                if *pos + 4 > data.len() {
                    return Err(Error::Gguf("truncated GGUF: F32 value out of bounds".into()));
                }
                let v = GgufValue::F32(f32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap()));
                *pos += 4;
                Ok(v)
            }
            23 => {
                if *pos + 8 > data.len() {
                    return Err(Error::Gguf("truncated GGUF: F64 value out of bounds".into()));
                }
                let v = GgufValue::F64(f64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap()));
                *pos += 8;
                Ok(v)
            }
            26 => {
                if *pos >= data.len() {
                    return Err(Error::Gguf("truncated GGUF: Bool value out of bounds".into()));
                }
                let v = GgufValue::Bool(data[*pos] != 0);
                *pos += 1;
                Ok(v)
            }
            13 => { // array of u32
                if *pos + 4 > data.len() {
                    return Err(Error::Gguf("truncated GGUF: array length out of bounds".into()));
                }
                let len = u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap()) as usize;
                *pos += 4;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    if *pos + 4 > data.len() {
                        return Err(Error::Gguf("truncated GGUF: array element out of bounds".into()));
                    }
                    arr.push(u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap()));
                    *pos += 4;
                }
                Ok(GgufValue::ArrayU32(arr))
            }
            _ => Err(Error::Gguf(format!("unknown value type {}", val_type))),
        }
    }
}
