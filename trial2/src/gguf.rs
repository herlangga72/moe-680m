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
        if magic != 0x46554747 { // "GGUF"
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
        for i in 0..n_kv {
            let key = Self::read_string(&data, &mut pos)
                .map_err(|e| Error::Gguf(format!("kv[{}] key read: {} at pos {}", i, e, pos)))?;
            if pos + 4 > data.len() {
                return Err(Error::Gguf(format!("kv[{}] val_type at pos {} OOB", i, pos)));
            }
            let val_type = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let val = Self::read_value(&data, &mut pos, val_type)
                .map_err(|e| Error::Gguf(format!("kv[{}] key='{}' type={}: {}", i, key, val_type, e)))?;
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

        let arch = get_str("general.architecture").unwrap_or_default();
        let prefix = if arch.contains("qwen") { &arch } else { "qwen3" };

        let n_heads_q = get_u32(&format!("{}.attention.head_count", prefix))?;
        let n_heads_kv = get_u32(&format!("{}.attention.head_count_kv", prefix))?;
        let hidden_dim = get_u32(&format!("{}.embedding_length", prefix))?;
        let head_dim = get_u32(&format!("{}.attention.key_length", prefix))
            .unwrap_or(hidden_dim / n_heads_q);

        Ok(ModelConfig {
            n_layers: get_u32(&format!("{}.block_count", prefix))?,
            hidden_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            ffn_intermediate: get_u32(&format!("{}.expert_feed_forward_length", prefix))
                .or_else(|_| get_u32(&format!("{}.feed_forward_length", prefix)))
                .unwrap_or(hidden_dim * 4),
            n_experts: get_u32(&format!("{}.expert_count", prefix)).unwrap_or(1),
            n_active_experts: get_u32(&format!("{}.expert_used_count", prefix)).unwrap_or(1),
            n_shared_experts: get_u32(&format!("{}.expert_shared_count", prefix)).unwrap_or(0),
            vocab_size: {
                let from_meta = get_u32(&format!("{}.vocab_size", prefix));
                from_meta.unwrap_or_else(|_| {
                    // token_embd.weight shape is [dim, vocab] or [vocab, dim]
                    self.find_tensor("token_embd.weight")
                        .and_then(|t| {
                            let s0 = t.shape.first().copied().unwrap_or(0);
                            let s1 = t.shape.get(1).copied().unwrap_or(0);
                            if s0 as u32 == hidden_dim { Some(s1 as u32) }
                            else if s1 as u32 == hidden_dim { Some(s0 as u32) }
                            else { Some(s0.max(s1) as u32) }
                        })
                        .unwrap_or(0)
                })
            },
            max_seq_len: get_u32(&format!("{}.context_length", prefix)).unwrap_or(32768),
            rope_theta: get_f32(&format!("{}.rope.freq_base", prefix)).unwrap_or(1_000_000.0),
            rope_type: get_str(&format!("{}.rope.type", prefix)).unwrap_or_else(|_| "default".into()),
            n_mtp_modules: get_u32(&format!("{}.nextn_predict_layers", prefix)).unwrap_or(0),
            mtp_depth: get_u32(&format!("{}.nextn_predict_layers", prefix)).unwrap_or(0),
            eps: get_f32(&format!("{}.attention.layer_norm_rms_epsilon", prefix)).unwrap_or(1e-6),
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
        // GGUF v3 type codes:
        // 0=u8, 1=i8, 2=u16, 3=i16, 4=u32, 5=i32, 6=f32, 7=bool,
        // 8=string, 9=array, 10=u64, 11=i64, 12=f64
        let check = |pos: &usize, n: usize| -> Result<()> {
            if pos.checked_add(n).map_or(true, |e| e > data.len()) {
                Err(Error::Gguf(format!("truncated at pos {} + {}", pos, n)))
            } else { Ok(()) }
        };
        Ok(match val_type {
            0 => { check(pos, 1)?; let v = GgufValue::U8(data[*pos]); *pos += 1; v }
            1 => { check(pos, 1)?; let v = GgufValue::I32(data[*pos] as i8 as i32); *pos += 1; v }
            2 => { check(pos, 2)?; let v = GgufValue::U32(u16::from_le_bytes(data[*pos..*pos+2].try_into().unwrap()) as u32); *pos += 2; v }
            3 => { check(pos, 2)?; let v = GgufValue::I32(i16::from_le_bytes(data[*pos..*pos+2].try_into().unwrap()) as i32); *pos += 2; v }
            4 => { check(pos, 4)?; let v = GgufValue::U32(u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap())); *pos += 4; v }
            5 => { check(pos, 4)?; let v = GgufValue::I32(i32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap())); *pos += 4; v }
            6 => { check(pos, 4)?; let v = GgufValue::F32(f32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap())); *pos += 4; v }
            7 => { check(pos, 1)?; let v = GgufValue::Bool(data[*pos] != 0); *pos += 1; v }
            8 => { GgufValue::String(Self::read_string(data, pos)?) }
            9 => {
                check(pos, 8)?;
                let elem_type = u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap());
                *pos += 4;
                let len = u64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap()) as usize;
                *pos += 8;
                // ponytail: array count is u64 in GGUF v3, not u32
                Self::read_array(data, pos, elem_type, len)?
            }
            10 => { check(pos, 8)?; let v = GgufValue::U64(u64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap())); *pos += 8; v }
            11 => { check(pos, 8)?; let v = GgufValue::I64(i64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap())); *pos += 8; v }
            12 => { check(pos, 8)?; let v = GgufValue::F64(f64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap())); *pos += 8; v }
            _ => return Err(Error::Gguf(format!("unknown value type {} at pos {}", val_type, pos))),
        })
    }

    fn read_array(data: &[u8], pos: &mut usize, elem_type: u32, len: usize) -> Result<GgufValue> {
        match elem_type {
            5 => { // i32 array
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(f32::from_bits(u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap())));
                    *pos += 4;
                }
                Ok(GgufValue::ArrayF32(arr))
            }
            11 => { // i64 array
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(i64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap()));
                    *pos += 8;
                }
                Ok(GgufValue::ArrayI64(arr))
            }
            8 => { // string array — skip, not used by our config
                for _ in 0..len { Self::read_string(data, pos)?; }
                Ok(GgufValue::ArrayU32(vec![]))
            }
            _ => {
                // Skip unknown array element types
                let elem_size = match elem_type {
                    0|1|7 => 1, 2|3 => 2, 4|5|6 => 4, 10|11|12 => 8,
                    _ => return Err(Error::Gguf(format!("unknown array elem type {}", elem_type))),
                };
                *pos += len * elem_size;
                Ok(GgufValue::ArrayU32(vec![]))
            }
        }
    }
}
