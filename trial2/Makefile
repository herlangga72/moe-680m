SPV_DIR  := src/shaders
COMP_DIR := shaders
MODEL    ?= model/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf

COMP_SRCS := $(filter-out $(COMP_DIR)/common.glsl, $(wildcard $(COMP_DIR)/*.comp))
SPV_FILES := $(patsubst $(COMP_DIR)/%.comp, $(SPV_DIR)/%.spv, $(COMP_SRCS))
CARGO := CARGO_TARGET_DIR=./target cargo

.PHONY: all shaders build smoke clean help

all: shaders build

shaders: $(SPV_FILES)

$(SPV_DIR)/%.spv: $(COMP_DIR)/%.comp $(COMP_DIR)/common.glsl
	@mkdir -p $(SPV_DIR)
	glslc --target-env=vulkan1.3 -fshader-stage=compute -I$(COMP_DIR) $< -o $@

build:
	$(CARGO) build --release

smoke: build
	./target/release/moe-680m --smoke

# ponytail: full test targets added after engine modules exist
clean:
	cargo clean
	rm -f $(SPV_DIR)/*.spv

help:
	@echo "all | shaders | build | smoke | clean"
