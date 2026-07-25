# GLSL → SPIR-V compilation for MoE-680M
# Needs glslc from Vulkan SDK on PATH

SPV_DIR = src/shaders
COMP_DIR = shaders

# Auto-detect existing .comp files (skip unimplemented stubs)
COMP_SRCS = $(wildcard $(COMP_DIR)/*.comp)
COMP_SRCS := $(filter-out $(COMP_DIR)/common.glsl, $(COMP_SRCS))
SPV_FILES = $(patsubst $(COMP_DIR)/%.comp, $(SPV_DIR)/%.spv, $(COMP_SRCS))

.PHONY: all clean gemm

all: $(SPV_FILES)

$(SPV_DIR)/%.spv: $(COMP_DIR)/%.comp $(COMP_DIR)/common.glsl
	@mkdir -p $(SPV_DIR)
	glslc -fshader-stage=compute -I$(COMP_DIR) $< -o $@

clean:
	rm -f $(SPV_FILES)

# Fast iteration: compile only the GEMM kernels
gemm: $(SPV_DIR)/w1_w3_fused.spv $(SPV_DIR)/w2.spv $(SPV_DIR)/w2_scatter.spv $(SPV_DIR)/kv_write.spv $(SPV_DIR)/residual_add.spv
