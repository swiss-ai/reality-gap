# Setup ngc-24.11
```toml
# ~/.edf/ngc-24.11.toml
image = "nvcr.io/nvidia/pytorch:24.11-py3"

mounts = [
    "/capstor",
    "/iopsstor",
    "/users/${USER}/benchmark-audio-tokenizer"
] 

workdir = "${HOME}/benchmark-audio-tokenizer"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true" 
com.hooks.aws_ofi_nccl.variant = "cuda12"

[env]
NCCL_DEBUG = "INFO" 
CUDA_CACHE_DISABLE = "1" 
TORCH_NCCL_ASYNC_ERROR_HANDLING = "1" 
MPICH_GPU_SUPPORT_ENABLED = "0"
```


```bash
srun --time=08:00:00 \
  --environment=ngc-24.11 \
  --account=infra01 \
  --partition=normal \
  --container-mounts="$HOME/vscode-cli-$(arch):/code" \
  --pty /code/code tunnel --accept-server-license-terms \
  --name="$CLUSTER_NAME-tunnel"
```


# xcodec2==0.1.5 mit uv, cpuonly ATTEMPT
```bash
deactivate
mv .venv-xcodec2 .venv-xcodec2-old 2>/dev/null
rm -rf /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2-old &

uv venv .venv-xcodec2
source .venv-xcodec2/bin/activate

uv pip install --no-deps xcodec2==0.1.5
uv pip install torch torchaudio torchao torchtune vector-quantize-pytorch
uv pip install numpy>=2.0.2 einops==0.8.0 transformers>=4.45.2 vector-quantize-pytorch==1.17.8 transformers==4.45.2
uv pip install soundfile
```

```bash
(.venv-xcodec2) lmantel@nid007415:~/benchmark-audio-tokenizer/examples/xcodec2$ uv pip list
Using Python 3.12.3 environment at: /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2
Package                 Version
----------------------- -----------
aiohappyeyeballs        2.6.1
aiohttp                 3.13.2
aiosignal               1.4.0
antlr4-python3-runtime  4.9.3
anyio                   4.11.0
attrs                   25.4.0
blobfile                3.1.0
certifi                 2025.10.5
cffi                    2.0.0
charset-normalizer      3.4.4
click                   8.3.0
datasets                4.4.1
dill                    0.4.0
einops                  0.8.0
einx                    0.3.0
filelock                3.20.0
frozendict              2.4.6
frozenlist              1.8.0
fsspec                  2025.10.0
h11                     0.16.0
hf-xet                  1.2.0
httpcore                1.0.9
httpx                   0.28.1
huggingface-hub         0.36.0
idna                    3.11
jinja2                  3.1.6
kagglehub               0.3.13
lxml                    6.0.2
markupsafe              3.0.3
mpmath                  1.3.0
multidict               6.7.0
multiprocess            0.70.18
networkx                3.5
numpy                   2.3.4
omegaconf               2.3.0
packaging               25.0
pandas                  2.3.3
pillow                  12.0.0
propcache               0.4.1
psutil                  7.1.3
pyarrow                 22.0.0
pycparser               2.23
pycryptodomex           3.23.0
python-dateutil         2.9.0.post0
pytz                    2025.2
pyyaml                  6.0.3
regex                   2025.11.3
requests                2.32.5
safetensors             0.6.2
sentencepiece           0.2.1
setuptools              80.9.0
shellingham             1.5.4
six                     1.17.0
sniffio                 1.3.1
soundfile               0.13.1
sympy                   1.14.0
tiktoken                0.12.0
tokenizers              0.20.3
torch                   2.9.0
torchao                 0.14.1
torchaudio              2.9.0
torchdata               0.11.0
torchtune               0.6.1
tqdm                    4.67.1
transformers            4.45.2
typer-slim              0.20.0
typing-extensions       4.15.0
tzdata                  2025.2
urllib3                 2.5.0
vector-quantize-pytorch 1.17.8
xcodec2                 0.1.5
xxhash                  3.6.0
yarl                    1.22.0
```

# Was eben recht gut geklappt hat: xcodec2==0.1.4 mit uv, cpuonly, torch 2.9.0
```bash
deactivate
mv /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2 /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2-old 2>/dev/null
rm -rf /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2-old &
cd /users/lmantel/benchmark-audio-tokenizer
uv venv .venv-xcodec2
source .venv-xcodec2/bin/activate
```

```bash
uv pip install torch torchvision torchaudio xcodec2 soundfile
```

```bash
(.venv-xcodec2) lmantel@nid007415:~/benchmark-audio-tokenizer$ uv pip list
Using Python 3.12.3 environment at: .venv-xcodec2
Package                  Version
------------------------ -----------
accelerate               1.11.0
aiohappyeyeballs         2.6.1
aiohttp                  3.13.2
aiosignal                1.4.0
annotated-types          0.7.0
antlr4-python3-runtime   4.9.3
anyio                    4.11.0
async-timeout            5.0.1
attrs                    25.4.0
audioread                3.1.0
auraloss                 0.4.0
blobfile                 3.1.0
certifi                  2025.10.5
cffi                     2.0.0
charset-normalizer       3.4.4
click                    8.3.0
contourpy                1.3.3
cycler                   0.12.1
datasets                 4.4.1
decorator                5.2.1
deepspeed                0.18.2
dill                     0.4.0
docker-pycreds           0.4.0
einops                   0.8.1
einx                     0.3.0
exceptiongroup           1.3.0
filelock                 3.20.0
fire                     0.7.1
fonttools                4.60.1
frozendict               2.4.6
frozenlist               1.8.0
fsspec                   2025.10.0
gin-config               0.5.0
gitdb                    4.0.12
gitpython                3.1.45
h11                      0.16.0
hf-transfer              0.1.9
hf-xet                   1.2.0
hjson                    3.1.0
httpcore                 1.0.9
httpx                    0.28.1
huggingface-hub          0.36.0
hydra-core               1.3.2
idna                     3.11
importlib-resources      6.5.2
iniconfig                2.3.0
jinja2                   3.1.6
joblib                   1.5.2
kagglehub                0.3.13
kiwisolver               1.4.9
lazy-loader              0.4
librosa                  0.11.0
lightning-utilities      0.15.2
llvmlite                 0.45.1
lxml                     6.0.2
markupsafe               3.0.3
matplotlib               3.10.7
meson                    1.9.1
mpmath                   1.3.0
msgpack                  1.1.2
multidict                6.7.0
multiprocess             0.70.18
networkx                 3.5
ninja                    1.13.0
numba                    0.62.1
numpy                    2.3.4
nvidia-cublas-cu12       12.9.1.4
nvidia-cuda-cupti-cu12   12.9.79
nvidia-cuda-nvrtc-cu12   12.9.86
nvidia-cuda-runtime-cu12 12.9.79
nvidia-cudnn-cu12        9.15.0.57
nvidia-cufft-cu12        11.4.1.4
nvidia-curand-cu12       10.3.10.19
nvidia-cusolver-cu12     11.7.5.82
nvidia-cusparse-cu12     12.5.10.65
nvidia-ml-py             13.580.82
nvidia-nccl-cu12         2.28.7
nvidia-nvjitlink-cu12    12.9.86
nvidia-nvtx-cu12         12.9.79
omegaconf                2.3.0
packaging                25.0
pandas                   2.3.3
pesq                     0.0.4
pillow                   12.0.0
platformdirs             4.5.0
pluggy                   1.6.0
pooch                    1.8.2
propcache                0.4.1
protobuf                 6.33.0
psutil                   7.1.3
py-cpuinfo               9.0.0
pyarrow                  22.0.0
pycparser                2.23
pycryptodomex            3.23.0
pydantic                 2.12.4
pydantic-core            2.41.5
pydub                    0.25.1
pygments                 2.19.2
pyparsing                3.2.5
pystoi                   0.4.1
pytest                   8.4.2
python-dateutil          2.9.0.post0
pytorch-lightning        2.5.6
pytz                     2025.2
pyyaml                   6.0.3
regex                    2025.11.3
requests                 2.32.5
rotary-embedding-torch   0.8.9
safetensors              0.6.2
scikit-learn             1.7.2
scipy                    1.16.3
sentencepiece            0.2.1
sentry-sdk               2.43.0
setproctitle             1.3.7
setuptools               80.9.0
six                      1.17.0
smmap                    5.0.2
sniffio                  1.3.1
soundfile                0.13.1
soxr                     1.0.0
sympy                    1.14.0
termcolor                3.2.0
threadpoolctl            3.6.0
tiktoken                 0.12.0
tokenizers               0.22.1
tomli                    2.3.0
torch                    2.9.0
torchao                  0.14.1
torchaudio               2.9.0
torchdata                0.11.0
torchmetrics             1.8.2
torchtune                0.6.1
torchvision              0.24.0
tqdm                     4.67.1
transformers             4.57.1
triton                   3.5.0
typing-extensions        4.15.0
typing-inspection        0.4.2
tzdata                   2025.2
urllib3                  2.5.0
vector-quantize-pytorch  1.17.8
wandb                    0.22.3
xcodec2                  0.1.4
xxhash                   3.6.0
yarl                     1.22.0
zipp                     3.23.0
```

# Paused Attempts
## xcodec2==0.1.5 mit uv ATTEMPT
```bash
deactivate
mv /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2 /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2-old 2>/dev/null
rm -rf /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2-old &
cd /users/lmantel/benchmark-audio-tokenizer
uv venv .venv-xcodec2
source .venv-xcodec2/bin/activate
```

```bash
uv pip install --no-deps xcodec2==0.1.5
uv pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"
```

```bash
(.venv-xcodec2) lmantel@nid007415:~/benchmark-audio-tokenizer$ uv pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
Using Python 3.12.3 environment at: .venv-xcodec2
Resolved 11 packages in 1.46s
Prepared 3 packages in 17.83s
Installed 11 packages in 40.64s
 + filelock==3.19.1
 + fsspec==2025.9.0
 + jinja2==3.1.6
 + markupsafe==2.1.5
 + mpmath==1.3.0
 + networkx==3.5
 + setuptools==70.2.0
 + sympy==1.13.1
 + torch==2.6.0+cu126
 + torchaudio==2.6.0
 + typing-extensions==4.15.0
[1]+  Done                    rm -rf /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2-old
(.venv-xcodec2) lmantel@nid007415:~/benchmark-audio-tokenizer$ python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/__init__.py", line 2108, in <module>
    from torch import _VF as _VF, functional as functional  # usort: skip
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/functional.py", line 7, in <module>
    import torch.nn.functional as F
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/nn/__init__.py", line 8, in <module>
    from torch.nn.modules import *  # usort: skip # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/nn/modules/__init__.py", line 1, in <module>
    from .module import Module  # usort: skip
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/nn/modules/module.py", line 29, in <module>
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/utils/__init__.py", line 8, in <module>
    from torch.utils import (
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/utils/data/__init__.py", line 1, in <module>
    from torch.utils.data.dataloader import (
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 20, in <module>
    import torch.distributed as dist
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/distributed/__init__.py", line 122, in <module>
    from .device_mesh import DeviceMesh, init_device_mesh
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/distributed/device_mesh.py", line 40, in <module>
    from torch.distributed.distributed_c10d import (
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 234, in <module>
    class Backend(str):
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 285, in Backend
    XCCL: ProcessGroup.BackendType.XCCL,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: type object 'torch._C._distributed_c10d.BackendType' has no attribute 'XCCL'. Did you mean: 'NCCL'?
```


## xcodec2==0.1.5 mit venv ATTEMPT
```bash
deactivate
mv /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2 /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2-old 2>/dev/null
rm -rf /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2-old &
cd /users/lmantel/benchmark-audio-tokenizer
python3 -m venv .venv-xcodec2 --system-site-packages
source .venv-xcodec2/bin/activate
```

```bash
pip install --no-deps xcodec2==0.1.5
pip install transformers>=4.45.2
```

```bash
Successfully installed torchaudio-2.6.0
(.venv-xcodec2) lmantel@nid007415:~/benchmark-audio-tokenizer$ python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"
PyTorch Version: 2.6.0a0+df5bbc09d1.nv24.11
CUDA Available: True
CUDA Version: 12.6
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torchaudio/__init__.py", line 2, in <module>
    from . import _extension  # noqa  # usort: skip
    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torchaudio/_extension/__init__.py", line 38, in <module>
    _load_lib("libtorchaudio")
  File "/users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torchaudio/_extension/utils.py", line 60, in _load_lib
    torch.ops.load_library(path)
  File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1354, in load_library
    ctypes.CDLL(path)
  File "/usr/lib/python3.12/ctypes/__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: /users/lmantel/benchmark-audio-tokenizer/.venv-xcodec2/lib/python3.12/site-packages/torchaudio/lib/libtorchaudio.so: undefined symbol: _ZN2at4_ops9fft_irfft4callERKNS_6TensorESt8optionalIN3c106SymIntEElS5_ISt17basic_string_viewIcSt11char_traitsIcEEE
```

Das Problem ist dass der container kein `torchaudio` installiert hat, und die version 2.6.0a0+df5bbc09d1.nv24.11 von pytorch nicht mit der version 2.6.0 von torchaudio kompatibel ist.


