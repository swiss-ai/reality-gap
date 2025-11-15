SHELL := /bin/bash

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


# xcodec2 CPU-only torch
xcodec2:
	mv .venv-xcodec2 .venv-xcodec2-old || true
	rm -rf .venv-xcodec2-old &

	uv venv .venv-xcodec2 --system-site-packages
	
	source .venv-xcodec2/bin/activate && \
	uv pip install --no-deps xcodec2==0.1.5 && \
	uv pip install -r requirements-xcodec2-freeze.txt && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"

