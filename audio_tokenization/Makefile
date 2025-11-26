SHELL := /bin/bash


venv:
	mv .venv .venv-old || true
	rm -rf .venv-old &

	uv venv .venv --system-site-packages

	uv pip compile requirements.txt -o requirements-subdeps.txt > /dev/null 2>&1
	
	sed -i '/^torch==/d' requirements-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-subdeps.txt

	source .venv/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/ao.git@v0.6.1 && \
	uv pip install --no-deps -r requirements-subdeps.txt && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"
