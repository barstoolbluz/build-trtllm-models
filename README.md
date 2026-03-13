# TRT-LLM Model Packages for Triton Inference Server

Pre-built TensorRT-LLM engine packages for use with the [triton-runtime](../../triton-runtime/) environment. Each package contains a TRT-LLM engine, tokenizer, and config template that `triton-setup-models` assembles into a ready-to-serve model at activation time.

## Model package contract

Every model package installs to `$out/share/models/<name>/` with this layout:

```
$out/share/models/<name>/
  config.pbtxt.template    # Triton config with token placeholders
  engine/                  # TRT-LLM engine files (rank0.engine, config.json)
  tokenizer/               # HuggingFace tokenizer files
  1/                       # Empty version directory (Triton convention)
```

## Token placeholders

`config.pbtxt.template` files contain tokens that `triton-setup-models` expands at `flox activate` time:

| Token | Expanded to |
|-------|-------------|
| `@EXECUTOR_WORKER_PATH@` | `$TRITON_BACKEND_DIR/tensorrtllm/trtllmExecutorWorker` |
| `@GPT_MODEL_PATH@` | `$FLOX_ENV_CACHE/models/<name>/engine` |
| `@TOKENIZER_DIR@` | `$FLOX_ENV_CACHE/models/<name>/tokenizer` |

The expanded result is written as `config.pbtxt` in the assembled model directory. The original template is not modified.

## Current models

| Model | Package | Engine | Precision | Size |
|-------|---------|--------|-----------|------|
| Qwen2.5-0.5B | `triton-model-qwen2-5-05b-trtllm` | TRT-LLM r26.02 | float16 | 1.2 GB |

## Quick start

```bash
# Build
flox build triton-model-qwen2-5-05b-trtllm

# Inspect output
ls result-triton-model-qwen2-5-05b-trtllm/share/models/qwen2_5_05b_trtllm/

# Publish to Flox catalog
flox publish
```

## Build versioning

Same pattern as [build-triton-server](../build-triton-server/): each package writes a version marker at `$out/share/<pname>/flox-build-version-<N>`. Version metadata lives in `build-meta/<package>.json` and is read by the Nix expression at eval time.

```bash
# Check build version
cat result-triton-model-qwen2-5-05b-trtllm/share/triton-model-qwen2-5-05b-trtllm/flox-build-version-*
```

## Adding new models

1. **Build the TRT-LLM engine** using [trtllm-tools](../../triton-trtllm-tools/):
   - Convert the HuggingFace model to a TRT-LLM checkpoint
   - Run `trtllm-build` to produce the engine

2. **Prepare the tokenizer**: copy tokenizer files from the HuggingFace model

3. **Create the tarball**:
   ```bash
   tar -czf <model>-r26.02.tar.gz engine/ tokenizer/
   ```

4. **Upload to GitHub Releases** under the appropriate tag (e.g., `v26.02`)

5. **Create the config template** at `models/<name>/config.pbtxt.template` using the `@EXECUTOR_WORKER_PATH@`, `@GPT_MODEL_PATH@`, and `@TOKENIZER_DIR@` tokens. Use the Qwen2.5-0.5B template as a reference.

6. **Write the Nix expression** at `.flox/pkgs/<pname>.nix` following the pattern in `triton-model-qwen2-5-05b-trtllm.nix`:
   - `fetchurl` the tarball from GitHub Releases
   - Copy engine, tokenizer, and config template into `$out/share/models/<name>/`
   - Create an empty `1/` version directory
   - Write a version marker

7. **Create build metadata** at `build-meta/<pname>.json` with the standard schema

8. **Build and publish**:
   ```bash
   git add .flox/pkgs/<pname>.nix build-meta/<pname>.json models/<name>/
   flox build <pname>
   flox publish
   ```

## Consuming in triton-runtime

Add the model package to the runtime manifest:

```toml
# triton-runtime/.flox/env/manifest.toml
[install]
triton-model-qwen2-5-05b-trtllm.pkg-path = "flox/triton-model-qwen2-5-05b-trtllm"
triton-model-qwen2-5-05b-trtllm.pkg-group = "trtllm-models"
```

At `flox activate`, `triton-setup-models` discovers the model in `$FLOX_ENV/share/models/`, copies it to `$FLOX_ENV_CACHE/models/`, expands the config template, and exports `TRITON_MODEL_REPOSITORY`. No further configuration is needed.

## License

Model weights are subject to their respective upstream licenses (e.g., Apache 2.0 for Qwen2.5).
