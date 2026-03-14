# Qwen2.5-0.5B TRT-LLM model for NVIDIA Triton Inference Server
#
# Pre-built TensorRT-LLM engine (float16, single GPU, r26.02).
# Bundle contains:
#   engine/    - rank0.engine + config.json (TRT-LLM engine)
#   tokenizer/ - HuggingFace tokenizer files (Qwen/Qwen2.5-0.5B)
#
# Output layout:
#   $out/share/models/qwen2_5_05b_trtllm/
#     config.pbtxt.template  - @EXECUTOR_WORKER_PATH@, @GPT_MODEL_PATH@, @TOKENIZER_DIR@
#     engine/                - TRT-LLM engine files
#     tokenizer/             - tokenizer files
#     1/                     - empty version directory (Triton convention)
#   $out/share/models/qwen2_5_05b_trtllm_preprocessing/
#     config.pbtxt.template  - @TOKENIZER_DIR@
#     tokenizer -> ../qwen2_5_05b_trtllm/tokenizer
#     1/model.py             - tokenization (Python backend)
#   $out/share/models/qwen2_5_05b_trtllm_postprocessing/
#     config.pbtxt.template  - @TOKENIZER_DIR@
#     tokenizer -> ../qwen2_5_05b_trtllm/tokenizer
#     1/model.py             - detokenization (Python backend)
#   $out/share/models/qwen2_5_05b_trtllm_ensemble/
#     config.pbtxt           - static ensemble DAG config
{ pkgs ? import <nixpkgs> {} }:

let
  pname = "triton-model-qwen2-5-05b-trtllm";
  tag = "r26.02";

  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/triton-model-qwen2-5-05b-trtllm.json);
  buildVersion = buildMeta.build_version;
  version = "0.1.0+${buildMeta.git_rev_short}";

  modelName = "qwen2_5_05b_trtllm";

  bundle = pkgs.fetchurl {
    url = "https://github.com/barstoolbluz/build-trtllm-models/releases/download/v26.02/qwen2_5_05b_trtllm-r26.02.tar.gz";
    hash = "sha256-mX9G4OaQlLVMhD352uuZBj6+oSmh/A37pvIn/LN42Fc=";
  };

  configTemplate = ../../models/${modelName}/config.pbtxt.template;

in pkgs.stdenv.mkDerivation {
  inherit pname version;

  src = bundle;

  sourceRoot = ".";
  unpackPhase = ''
    mkdir -p source
    tar -xzf ${bundle} -C source
    cd source
  '';

  dontBuild = true;
  dontConfigure = true;

  installPhase = ''
    runHook preInstall

    modelDir="$out/share/models/${modelName}"
    mkdir -p "$modelDir"

    # Engine files
    cp -r engine "$modelDir/"

    # Tokenizer files
    cp -r tokenizer "$modelDir/"

    # Config template (tokens expanded at activation by triton-setup-models)
    cp ${configTemplate} "$modelDir/config.pbtxt.template"

    # Empty version directory (Triton convention)
    mkdir -p "$modelDir/1"

    # Preprocessing model
    preDir="$out/share/models/${modelName}_preprocessing"
    mkdir -p "$preDir/1"
    cp ${../../models/${modelName}_preprocessing/config.pbtxt.template} "$preDir/config.pbtxt.template"
    cp ${../../models/${modelName}_preprocessing/1/model.py} "$preDir/1/model.py"
    ln -s ../qwen2_5_05b_trtllm/tokenizer "$preDir/tokenizer"

    # Postprocessing model
    postDir="$out/share/models/${modelName}_postprocessing"
    mkdir -p "$postDir/1"
    cp ${../../models/${modelName}_postprocessing/config.pbtxt.template} "$postDir/config.pbtxt.template"
    cp ${../../models/${modelName}_postprocessing/1/model.py} "$postDir/1/model.py"
    ln -s ../qwen2_5_05b_trtllm/tokenizer "$postDir/tokenizer"

    # Ensemble model
    ensDir="$out/share/models/${modelName}_ensemble"
    mkdir -p "$ensDir/1"
    cp ${../../models/${modelName}_ensemble/config.pbtxt} "$ensDir/config.pbtxt"

    # Version marker
    mkdir -p "$out/share/${pname}"
    cat > "$out/share/${pname}/flox-build-version-${toString buildVersion}" <<'MARKER'
    build-version: ${toString buildVersion}
    upstream-version: ${version}
    upstream-tag: ${tag}
    git-rev: ${buildMeta.git_rev}
    git-rev-short: ${buildMeta.git_rev_short}
    force-increment: ${toString buildMeta.force_increment}
    changelog: ${buildMeta.changelog}
    MARKER

    runHook postInstall
  '';

  dontStrip = true;
  dontFixup = true;

  meta = with pkgs.lib; {
    description = "Qwen2.5-0.5B TRT-LLM model for NVIDIA Triton Inference Server";
    homepage = "https://huggingface.co/Qwen/Qwen2.5-0.5B";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
