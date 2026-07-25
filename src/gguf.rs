// Job 2: GGUF Parser
// Parses GGUF file header, metadata, tensor index.
// See SPEC.md §7 for format reference.

use std::collections::HashMap;

// ── GGUF Types ──

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ4_XS = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            30 => Some(Self::IQ4_XS),
            _ => None,
        }
    }

    /// Block size for this quantization type (number of weights per block).
    pub fn block_size(&self) -> u32 {
        match self {
            Self::IQ4_XS => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::F32 | Self::F16 => 1,
            Self::Q2_K | Self::Q3_K => 256,
        }
    }

    /// Size in bytes of one block (for calculating tensor data size).
    pub fn block_bytes(&self) -> u32 {
        match self {
            Self::IQ4_XS => 36, // approximate; verify against ggml
            Self::Q4_K => 144,  // 256 * 4.5 bpw ≈ 144 bytes
            Self::Q5_K => 160,
            Self::Q6_K => 192,
            Self::Q8_K => 256,
            Self::Q4_0 => 20,
            Self::Q4_1 => 24,
            Self::Q5_0 => 24,
            Self::Q5_1 => 28,
            Self::Q8_0 => 40,
            Self::Q8_1 => 48,
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q2_K => 80,
            Self::Q3_K => 104,
        }
    }

    /// Bytes per weight.
    pub fn bpw(&self) -> f32 {
        self.block_bytes() as f32 / self.block_size() as f32
    }
}

// ── Tensors ──

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: GgmlType,
    pub offset: u64,
    pub size: u64,
}

// ── Model Configuration ──

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub architecture: String,
    pub block_count: u32,
    pub context_length: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub quant_type: GgmlType,
}

impl ModelConfig {
    fn from_metadata(meta: &HashMap<String, MetadataValue>) -> Result<Self, String> {
        let arch = meta_get_str(meta, "general.architecture")?;
        let quant_type_val = meta_get_int(meta, "general.file_type").unwrap_or(30) as u32;
        let quant_type = GgmlType::from_u32(quant_type_val).unwrap_or(GgmlType::IQ4_XS);

        Ok(ModelConfig {
            architecture: arch,
            block_count: meta_get_int(meta, "llama.block_count")? as u32,
            context_length: meta_get_int(meta, "llama.context_length")? as u32,
            embedding_length: meta_get_int(meta, "llama.embedding_length")? as u32,
            feed_forward_length: meta_get_int(meta, "llama.feed_forward_length")? as u32,
            expert_count: meta_get_int(meta, "llama.expert_count")? as u32,
            expert_used_count: meta_get_int(meta, "llama.expert_used_count")? as u32,
            quant_type,
        })
    }
}

// ── GGUF Reader ──

pub struct GgufReader<'a> {
    pub data: &'a [u8],
    pub tensor_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: u64,  // byte offset where tensor data starts
    pub config: ModelConfig,
}

impl<'a> GgufReader<'a> {
    /// Parse a GGUF file from its memory-mapped bytes.
    pub fn parse(data: &'a [u8]) -> Result<Self, String> {
        let mut off = 0u64;

        // Magic
        if data.len() < 16 {
            return Err("File too small for GGUF header".into());
        }
        let magic = &data[0..4];
        if magic != b"GGUF" {
            return Err(format!("Bad magic: {:02x?}, expected GGUF", magic));
        }
        off += 4;

        // Version
        let version = read_u32(data, &mut off);
        if version != 3 && version != 2 {
            return Err(format!("Unsupported GGUF version: {}", version));
        }

        // Tensor count
        let tensor_count = read_u64(data, &mut off);

        // Metadata KV count
        let metadata_kv_count = read_u64(data, &mut off);

        // Metadata KVs
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let (key, value) = parse_metadata_kv(data, &mut off)?;
            metadata.insert(key, value);
        }

        // Tensor infos
        let alignment = if version == 2 { 32u64 } else { 32u64 };
        // Tensor data starts at the next aligned offset after last tensor info
        let _tensor_infos_start = off;

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut max_end_offset = 0u64;
        for _ in 0..tensor_count {
            let ti = parse_tensor_info(data, &mut off)?;
            let end = ti.offset + ti.size;
            if end > max_end_offset {
                max_end_offset = end;
            }
            tensors.push(ti);
        }

        // The tensor data section begins at some aligned offset.
        // In GGUF v3, the data starts at the first offset ≥ tensor_infos_end
        // that is aligned to alignment (typically 32 bytes).
        let tensor_infos_end = off;
        let tensor_data_offset = align_up(tensor_infos_end, alignment);

        // Parse config from metadata
        let config = ModelConfig::from_metadata(&metadata)?;

        Ok(GgufReader {
            data,
            tensor_count,
            metadata,
            tensors,
            tensor_data_offset,
            config,
        })
    }

}

// ── Metadata Values ──

#[derive(Clone, Debug)]
pub enum MetadataValue {
    Float32(f32),
    Uint32(u32),
    Int32(i32),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
}

fn parse_metadata_kv(data: &[u8], off: &mut u64) -> Result<(String, MetadataValue), String> {
    let key = read_string(data, off)?;
    let value_type = read_u32(data, off);
    let value = read_metadata_value(data, off, value_type)?;
    Ok((key, value))
}

fn read_metadata_value(data: &[u8], off: &mut u64, ty: u32) -> Result<MetadataValue, String> {
    // Types this parser doesn't use: advance offset by size, return dummy
    let mut skip = |n: u64| { *off += n; Ok(MetadataValue::Uint32(0)) };
    match ty {
        0 | 1 => skip(1),
        2 | 3 => skip(2),
        7 => skip(4),   // GGUF_BOOL
        12 => skip(8),  // GGUF_FLOAT64
        6 => Ok(MetadataValue::Float32(f32::from_bits(read_u32(data, off)))),
        4 => Ok(MetadataValue::Uint32(read_u32(data, off))),
        5 => Ok(MetadataValue::Int32(read_i32(data, off))),
        8 => { let s = read_string(data, off)?; Ok(MetadataValue::String(s)) }
        9 => {
            let arr_type = read_u32(data, off);
            let arr_len = read_u64(data, off);
            let mut vals = Vec::with_capacity(arr_len as usize);
            for _ in 0..arr_len {
                vals.push(read_metadata_value(data, off, arr_type)?);
            }
            Ok(MetadataValue::Array(vals))
        }
        10 => Ok(MetadataValue::Uint64(read_u64(data, off))),
        11 => Ok(MetadataValue::Int64(read_i64(data, off))),
        _ => Err(format!("Unknown metadata value type: {}", ty)),
    }
}

// ── Tensor Info Parsing ──

fn parse_tensor_info(data: &[u8], off: &mut u64) -> Result<TensorInfo, String> {
    let name = read_string(data, off)?;
    let n_dims = read_u32(data, off);
    let mut dims = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        dims.push(read_u64(data, off));
    }
    let ggml_type_val = read_u32(data, off);
    let ggml_type = GgmlType::from_u32(ggml_type_val)
        .ok_or_else(|| format!("Unknown GGML type: {} for tensor {}", ggml_type_val, name))?;
    let offset = read_u64(data, off);

    // Compute size from dims + type
    let num_weights: u64 = dims.iter().product();
    let size = if ggml_type == GgmlType::F32 || ggml_type == GgmlType::F16 {
        num_weights * ggml_type.block_bytes() as u64
    } else {
        let blocks = (num_weights + ggml_type.block_size() as u64 - 1) / ggml_type.block_size() as u64;
        blocks * ggml_type.block_bytes() as u64
    };

    Ok(TensorInfo {
        name,
        ggml_type,
        offset,
        size,
    })
}

// ── Metadata Helper Accessors ──

fn meta_get_str(meta: &HashMap<String, MetadataValue>, key: &str) -> Result<String, String> {
    meta.get(key)
        .and_then(|v| {
            if let MetadataValue::String(s) = v {
                Some(s.clone())
            } else {
                None
            }
        })
        .ok_or_else(|| format!("Missing metadata key: {}", key))
}

fn meta_get_int(meta: &HashMap<String, MetadataValue>, key: &str) -> Result<u64, String> {
    meta.get(key)
        .map(|v| match v {
            MetadataValue::Uint32(x) => *x as u64,
            MetadataValue::Int32(x) => *x as u64,
            MetadataValue::Uint64(x) => *x,
            MetadataValue::Int64(x) => *x as u64,
            _ => 0,
        })
        .ok_or_else(|| format!("Missing metadata key: {}", key))
}

// ── Binary Readers ──

fn read_u32(data: &[u8], off: &mut u64) -> u32 {
    let v = u32::from_le_bytes([
        data[*off as usize],
        data[*off as usize + 1],
        data[*off as usize + 2],
        data[*off as usize + 3],
    ]);
    *off += 4;
    v
}

fn read_i32(data: &[u8], off: &mut u64) -> i32 {
    let v = i32::from_le_bytes([
        data[*off as usize],
        data[*off as usize + 1],
        data[*off as usize + 2],
        data[*off as usize + 3],
    ]);
    *off += 4;
    v
}

fn read_u64(data: &[u8], off: &mut u64) -> u64 {
    let v = u64::from_le_bytes([
        data[*off as usize],
        data[*off as usize + 1],
        data[*off as usize + 2],
        data[*off as usize + 3],
        data[*off as usize + 4],
        data[*off as usize + 5],
        data[*off as usize + 6],
        data[*off as usize + 7],
    ]);
    *off += 8;
    v
}

fn read_i64(data: &[u8], off: &mut u64) -> i64 {
    let bytes: [u8; 8] = [
        data[*off as usize],
        data[*off as usize + 1],
        data[*off as usize + 2],
        data[*off as usize + 3],
        data[*off as usize + 4],
        data[*off as usize + 5],
        data[*off as usize + 6],
        data[*off as usize + 7],
    ];
    *off += 8;
    i64::from_le_bytes(bytes)
}

fn read_string(data: &[u8], off: &mut u64) -> Result<String, String> {
    let len = read_u64(data, off);
    let start = *off as usize;
    let end = start + len as usize;
    if end > data.len() {
        return Err("String exceeds file bounds".into());
    }
    let s = std::str::from_utf8(&data[start..end])
        .map_err(|e| format!("Invalid UTF-8 in GGUF string: {}", e))?;
    *off += len;
    Ok(s.to_string())
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

// ── Print / Debug ──

impl std::fmt::Display for ModelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} | {} blocks, {} ctx, {} hidden, {} inter, {} experts ({} active), {:.2} bpw",
            self.architecture,
            self.block_count,
            self.context_length,
            self.embedding_length,
            self.feed_forward_length,
            self.expert_count,
            self.expert_used_count,
            self.quant_type.bpw(),
        )
    }
}
