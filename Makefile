SHELL := /bin/bash


# create all venvs (executed from NGC-24.11)
venvs: neucodec cosyvoice2 xcodec2 wavtokenizer glm4voice


# Shared vllm-omni serving venv — for the Stage-3 batched VoxCPM2 path.
# No official aarch64 wheels exist for vllm or vllm-omni, so we build both
# from source against NGC's aarch64 torch (inherited via --system-site-packages).
#
# Expected first-build cost: ~30-60 min of CUDA compile on GH200 (sm_90).
# After that, .build_complete short-circuits the rebuild.
#
# Known risks (iterate when these fail):
#   - Transitive dep list below is a best-guess from vllm 0.20.0's pyproject;
#     expect to add packages as imports surface in the server log.
#   - vllm-omni PyPI wheel may be x86_64-only — the recipe falls through to
#     a from-source vllm-omni build on failure.
#   - TORCH_CUDA_ARCH_LIST="9.0" is set for GH200; change for other GPUs.
#   - MAX_JOBS=4 keeps the compile from OOMing on small nodes; tune up if
#     you have headroom and want a faster first build.
vllm-omni:
	mv .venv-vllm-omni .venv-vllm-omni-old || true
	rm -rf .venv-vllm-omni-old &

	uv venv .venv-vllm-omni --system-site-packages

	source .venv-vllm-omni/bin/activate && \
	export TORCH_CUDA_ARCH_LIST="9.0" && \
	export MAX_JOBS=4 && \
	export VLLM_TARGET_DEVICE=cuda && \
	uv pip install --no-deps ninja cmake packaging wheel 'setuptools-scm<9' pybind11 cython && \
	uv pip install --no-deps \
	    transformers tokenizers huggingface-hub sentencepiece protobuf safetensors \
	    aiohttp fastapi 'uvicorn[standard]' httpx pydantic pyzmq msgspec \
	    cloudpickle gguf typing-extensions tqdm prometheus-client \
	    importlib-metadata partial-json-parser regex requests \
	    pillow psutil six && \
	if [ ! -d vllm-src ]; then \
	    git clone --branch v0.20.0 --depth 1 https://github.com/vllm-project/vllm.git vllm-src; \
	fi && \
	cd vllm-src && uv pip install --no-deps --no-build-isolation -e . && cd .. && \
	( uv pip install --no-deps vllm-omni || ( \
	    echo "PyPI vllm-omni failed (likely no aarch64 wheel) — building from source"; \
	    if [ ! -d vllm-omni-src ]; then \
	        git clone --depth 1 https://github.com/vllm-project/vllm-omni.git vllm-omni-src; \
	    fi && \
	    cd vllm-omni-src && uv pip install --no-deps --no-build-isolation -e . && cd .. \
	) ) && \
	python -c "import vllm, vllm_omni; print('vllm + vllm-omni import OK')" && \
	touch .venv-vllm-omni/.build_complete


# Shared scorer venv — Whisper WER (and future audio metrics) used by
# `benchmark_tts.py score` / `aggregate`. Split out so we don't reinstall
# Whisper's deps inside every backend's venv, and so backend dep-pins
# (numpy, transformers, torch) can't fight Whisper's.
# Inherits NGC's aarch64 torch / torchaudio via --system-site-packages.
scorer:
	mv .venv-scorer .venv-scorer-old || true
	rm -rf .venv-scorer-old &

	uv venv .venv-scorer --system-site-packages

	source .venv-scorer/bin/activate && \
	uv pip install --no-deps --no-build-isolation \
	    git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps openai-whisper tiktoken more-itertools \
	    numba llvmlite && \
	python -c "import torchaudio, whisper; print('Scorer import OK')" && \
	touch .venv-scorer/.build_complete


# XTTS v2 — REFERENCE-ONLY (non-commercial license, Coqui Public Model License).
# Not part of the active TTS pipeline. Build only if running a reference-only
# comparison via `benchmark_tts.py generate --backend xtts --allow-reference`.
# Uses the Idiap-maintained `coqui-tts` fork (Coqui's `TTS==0.22.0` is pinned
# to Python <3.12 and won't build on NGC 24.11). Same import path.
# Inherits NGC's aarch64 torch; torchaudio installed from source.
xtts-reference-only:
	mv .venv-xtts .venv-xtts-old || true
	rm -rf .venv-xtts-old &

	uv venv .venv-xtts --system-site-packages

	source .venv-xtts/bin/activate && \
	uv pip install --no-deps --no-build-isolation \
	    git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps --no-build-isolation coqui-tts && \
	uv pip install --no-deps coqpit-config coqui-tts-trainer encodec \
	    gruut[de,es,fr] anyascii inflect pysbd num2words bangla \
	    bnnumerizer bnunicodenormalizer nltk pypinyin jieba spacy \
	    g2pkk hangul_romanize jamo \
	    'transformers==4.50.0' 'tokenizers>=0.21,<0.22' 'huggingface_hub<1.0' \
	    regex tqdm safetensors packaging filelock pyyaml && \
	python -c "from TTS.api import TTS; print('XTTS import OK')" && \
	touch .venv-xtts/.build_complete


# OmniVoice (k2-fsa) — Apache 2.0, 600+ langs, voice cloning, requires ref_text.
# Inherits NGC's aarch64 torch (Clariden has no x86_64 torch 2.8 wheels);
# torchaudio installed from source since NGC 24.11 doesn't ship it.
omnivoice:
	mv .venv-omnivoice .venv-omnivoice-old || true
	rm -rf .venv-omnivoice-old &

	uv venv .venv-omnivoice --system-site-packages

	source .venv-omnivoice/bin/activate && \
	uv pip install --no-deps --no-build-isolation \
	    git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps omnivoice && \
	uv pip install --no-deps soundfile safetensors accelerate \
	    'transformers>=4.55,<4.56' 'tokenizers>=0.21,<0.22' 'huggingface_hub<1.0' \
	    einops regex tqdm packaging filelock pyyaml && \
	python -c "from omnivoice import OmniVoice; print('OmniVoice import OK')" && \
	touch .venv-omnivoice/.build_complete


# VoxCPM2 (OpenBMB) — Apache 2.0, 30 languages incl. Polish, voice cloning.
# Supervisor's recommended TTS. Direct PyTorch path; vLLM-omni serving wired
# separately later. Inherits NGC's aarch64 torch.
voxcpm2:
	mv .venv-voxcpm2 .venv-voxcpm2-old || true
	rm -rf .venv-voxcpm2-old &

	uv venv .venv-voxcpm2 --system-site-packages

	source .venv-voxcpm2/bin/activate && \
	uv pip install --no-deps --no-build-isolation \
	    git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps voxcpm && \
	uv pip install --no-deps soundfile safetensors accelerate \
	    'transformers>=4.55,<4.56' 'tokenizers>=0.21,<0.22' 'huggingface_hub<1.0' \
	    einops regex tqdm packaging filelock pyyaml && \
	uv pip install --no-deps lhotse cytoolz click intervaltree audioread tabulate && \
	python -c "from voxcpm import VoxCPM; import lhotse; print('VoxCPM2 + lhotse import OK')" && \
	touch .venv-voxcpm2/.build_complete


# Piper TTS (rhasspy) — MIT, Polish-native, ONNXRuntime-based, lightweight.
# Requires the pl_PL-gosia-medium voice files (.onnx + .onnx.json) under
# voices/. Voice files are tiny (~25 MB) and downloaded once.
piper:
	mv .venv-piper .venv-piper-old || true
	rm -rf .venv-piper-old &

	uv venv .venv-piper --system-site-packages

	source .venv-piper/bin/activate && \
	uv pip install --no-deps --no-build-isolation \
	    git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps piper-tts onnxruntime pathvalidate \
	    flatbuffers protobuf packaging && \
	mkdir -p voices && \
	if [ ! -f voices/pl_PL-gosia-medium.onnx ]; then \
	    cd voices && \
	    curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx && \
	    curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json && \
	    cd ..; \
	fi && \
	python -c "from piper import PiperVoice; print('Piper import OK')" && \
	touch .venv-piper/.build_complete


# Parler-TTS Mini Multilingual — Apache 2.0, prompt-controlled, multilingual.
# Inherits NGC's aarch64 torch; pulls parler-tts package + transformers.
parler:
	mv .venv-parler .venv-parler-old || true
	rm -rf .venv-parler-old &

	uv venv .venv-parler --system-site-packages

	source .venv-parler/bin/activate && \
	uv pip install --no-deps --no-build-isolation \
	    git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps git+https://github.com/huggingface/parler-tts.git && \
	uv pip install --no-deps soundfile safetensors accelerate \
	    'transformers==4.50.0' 'tokenizers>=0.21,<0.22' 'huggingface_hub<1.0' \
	    sentencepiece protobuf descript-audio-codec encodec einops regex tqdm \
	    packaging filelock pyyaml && \
	uv pip install --no-deps descript-audiotools argbind librosa flatten-dict \
	    ffmpy gitpython numba llvmlite pyloudnorm randomname rich pystoi \
	    julius markdown2 scipy && \
	python -c "from parler_tts import ParlerTTSForConditionalGeneration; print('Parler-TTS import OK')" && \
	touch .venv-parler/.build_complete


# F5-TTS — MIT code, diffusion, voice cloning.
# Base checkpoint is English/Chinese; Polish either via cross-lingual prompt
# or via a community Polish fine-tune via `--checkpoint`.
f5:
	mv .venv-f5 .venv-f5-old || true
	rm -rf .venv-f5-old &

	uv venv .venv-f5 --system-site-packages

	source .venv-f5/bin/activate && \
	uv pip install --no-deps --no-build-isolation \
	    git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps f5-tts && \
	uv pip install --no-deps soundfile safetensors accelerate \
	    'transformers==4.50.0' 'tokenizers>=0.21,<0.22' 'huggingface_hub<1.0' \
	    librosa vocos x-transformers einops einx jieba pypinyin \
	    cached_path datasets bitsandbytes scipy hydra-core omegaconf \
	    boto3 \
	    google-api-core google-cloud-storage google-cloud-core google-auth \
	    google-resumable-media google-crc32c \
	    googleapis-common-protos proto-plus cachetools pyasn1 pyasn1-modules rsa \
	    requests \
	    regex tqdm packaging filelock pyyaml && \
	python -c "from f5_tts.api import F5TTS; print('F5-TTS import OK')" && \
	touch .venv-f5/.build_complete



# xcodec2 with cuda
xcodec2:
	mv .venv-xcodec2 .venv-xcodec2-old || true
	rm -rf .venv-xcodec2-old &

	uv venv .venv-xcodec2 --system-site-packages

	uv pip compile requirements-xcodec2-topdeps.txt -o requirements-xcodec2-subdeps.txt
	sed -i '/^torch==/d' requirements-xcodec2-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-xcodec2-subdeps.txt
	
	source .venv-xcodec2/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/ao.git@v0.6.1 && \
	uv pip install --no-deps -r requirements-xcodec2-subdeps.txt && \
	uv pip install --no-deps xcodec2==0.1.5 && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# neucodec with cuda
neucodec:
	mv .venv-neucodec .venv-neucodec-old || true
	rm -rf .venv-neucodec-old &

	uv venv .venv-neucodec --system-site-packages

	uv pip compile requirements-neucodec-topdeps.txt -o requirements-neucodec-subdeps.txt
	sed -i '/^torch==/d' requirements-neucodec-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-neucodec-subdeps.txt
	
	source .venv-neucodec/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/ao.git@v0.6.1 && \
	uv pip install --no-deps -r requirements-neucodec-subdeps.txt && \
	uv pip install --no-deps neucodec==0.0.4 && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# glm4voice
glm4voice:
	mv .venv-glm4voice .venv-glm4voice-old || true
	rm -rf .venv-glm4voice-old &

	uv venv .venv-glm4voice --system-site-packages

	uv pip compile requirements-glm4voice-topdeps.txt -o requirements-glm4voice-subdeps.txt
	sed -i '/^torch==/d' requirements-glm4voice-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-glm4voice-subdeps.txt
	
	source .venv-glm4voice/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps -r requirements-glm4voice-subdeps.txt && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# wavtokenizer
wavtokenizer:
	mv .venv-wavtokenizer .venv-wavtokenizer-old || true
	rm -rf .venv-wavtokenizer-old &

	uv venv .venv-wavtokenizer --system-site-packages

	uv pip compile requirements-wavtokenizer-topdeps.txt -o requirements-wavtokenizer-subdeps.txt
	sed -i '/^torch==/d' requirements-wavtokenizer-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-wavtokenizer-subdeps.txt

	source .venv-wavtokenizer/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps -r requirements-wavtokenizer-subdeps.txt && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# cosyvoice2
cosyvoice2:
	mv .venv-cosyvoice2 .venv-cosyvoice2-old || true
	rm -rf .venv-cosyvoice2-old &

	uv venv .venv-cosyvoice2 --system-site-packages

	uv pip compile requirements-cosyvoice2-topdeps.txt -o requirements-cosyvoice2-subdeps.txt
	sed -i '/^torch==/d' requirements-cosyvoice2-subdeps.txt

	source .venv-cosyvoice2/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps -r requirements-cosyvoice2-subdeps.txt && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# # xcodec2 CPU-only torch
# xcodec2:
# 	mv .venv-xcodec2 .venv-xcodec2-old || true
# 	rm -rf .venv-xcodec2-old &

# 	uv venv .venv-xcodec2
	
# 	source .venv-xcodec2/bin/activate && \
# 	uv pip install --no-deps xcodec2==0.1.5 && \
# 	uv pip install -r requirements-xcodec2-freeze.txt && \
# 	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# #############################################################################


# # xcodec2 with CUDA intent (not working because torchtune leads to AttributeError: module 'torch' has no attribute 'int1')
# xcodec2:
# 	mv .venv-xcodec2 .venv-xcodec2-old || true
# 	rm -rf .venv-xcodec2-old &

# 	uv venv .venv-xcodec2 --system-site-packages

# 	uv pip compile requirements-xcodec2-topdeps.txt -o requirements-xcodec2-subdeps.txt
# 	sed -i '/^torch==/d' requirements-xcodec2-subdeps.txt
	
# 	source .venv-xcodec2/bin/activate && \
# 	uv pip install --no-deps xcodec2==0.1.5 && \
# 	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
# 	uv pip install --no-deps -r requirements-xcodec2-subdeps.txt && \
# 	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


