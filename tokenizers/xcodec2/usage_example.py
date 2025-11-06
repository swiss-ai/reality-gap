import torch
import soundfile as sf
from pathlib import Path
 
from xcodec2.modeling_xcodec2 import XCodec2Model
 
model_path = "HKUSTAudio/xcodec2"  

model = XCodec2Model.from_pretrained(model_path)
model.eval()#.cuda()


ROOT_DIR = Path(__file__).parent.parent.parent
test_path = ROOT_DIR / "data" / "raw" / "audio.wav"
wav, sr = sf.read(test_path)



wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)

# Convert multi-channel waveform to mono and ensure shape is (1, T)
if wav_tensor.dim() == 3:
    # assumed shape (1, T, C) -> average over channel dim to get (1, T)
    wav_tensor = wav_tensor.mean(dim=-1)

with torch.no_grad():
   # Only 16khz speech
   # Only supports single input. For batch inference, please refer to the link below.
    vq_code = model.encode_code(input_waveform=wav_tensor)
    print("Code:", vq_code )  

    recon_wav = model.decode_code(vq_code)#.cpu()       # Shape: (1, 1, T')

 
sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), sr)
print("Done! Check reconstructed.wav")
