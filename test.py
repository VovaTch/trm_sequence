import torch
import hydra
import sys
import math
import shutil

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


def rows_used(text, width):
    # how many terminal rows this block will occupy (approx; good enough for ASCII)
    total = 0
    for line in text.splitlines() or [""]:
        total += max(1, math.ceil(max(1, len(line.expandtabs())) / max(1, width)))
    return total


prev_rows = 0
with torch.no_grad():
    for generated_tokens in model.stream(
        init_tokens=tokenized_text, seq_len=1024, vocab_size=66, temperature=0.7
    ):
        generated_text = tokenizer.decode(generated_tokens.squeeze().cpu().numpy())  # type: ignore

        # Move to the beginning of the previous block
        if prev_rows:
            # \x1b[{n}F = move cursor up n lines to column 1 (beginning-of-line)
            sys.stdout.write(f"\x1b[{prev_rows}F")

        # Clear from cursor to end of screen, then print the new block
        sys.stdout.write("\x1b[J")
        sys.stdout.write(
            generated_text + ("\n" if not generated_text.endswith("\n") else "")
        )
        sys.stdout.flush()

        width = shutil.get_terminal_size().columns or 80
        prev_rows = rows_used(generated_text, width)

print()  # leave the cursor on a clean line at the end
