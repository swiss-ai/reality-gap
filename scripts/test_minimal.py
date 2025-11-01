import os
from datasets import load_dataset

# Set token directly
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Download 3 samples
print("Downloading...")
stream = load_dataset("disco-eth/EuroSpeech", "france", split="train", streaming=True)

for i, sample in enumerate(stream):
    if i >= 3:
        break
    print(f"Sample {i+1}: Got it")

print("SUCCESS")
