import torch
import hydra

from models.modules.base import load_inner_model_state_dict
from models.modules.diffusion_llm import DiffusionLLMLightningModule
from models.tokenizers.char import CharLevelTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current device is {device}")

# Load network
with hydra.initialize(version_base=None, config_path="config"):
    cfg = hydra.compose(config_name="dllm")
cfg.learning.batch_size = 128
weights_path = "weights/dllm.ckpt"
model: DiffusionLLMLightningModule = hydra.utils.instantiate(cfg.module)
model = load_inner_model_state_dict(model, weights_path).to(device).eval()  # type: ignore

tokenizer = CharLevelTokenizer()

TEXT = "What is the meaning of life?"
tokenized_text = tokenizer.encode(TEXT)
tokenized_text = torch.tensor(tokenized_text).unsqueeze(0)

with torch.no_grad():
    generated_tokens = model.generate(
        init_tokens=tokenized_text, seq_len=1024, vocab_size=66
    )
generated_text = tokenizer.decode(generated_tokens[0, ...].cpu().numpy())  # type: ignore

print(generated_text)
