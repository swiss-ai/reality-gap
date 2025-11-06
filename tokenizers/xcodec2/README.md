# Setup of xcodec2 CPU-only Virtual Environment using uv
Make sure you are in the desired directory to install the virtual environment.
Quickly (re)create a fresh virtual environment with xcodec2 installed, e.g. by running:

```bash
deactivate
mv .venv-xcodec2 .venv-xcodec2-old 2>/dev/null
rm -rf /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2-old &

uv venv .venv-xcodec2
source .venv-xcodec2/bin/activate
```

Then, install the required packages. Most importantly we have to install xcodec2 without dependencies to avoid conflicts:

```bash
uv pip install --no-deps xcodec2==0.1.5
uv pip install -r tokenizers/xcodec2/requirements.txt
```