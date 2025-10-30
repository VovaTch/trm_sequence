import torch
import hydra
import sys
import math
import shutil

from models.modules.base import load_inner_model_state_dict
from models.modules.trm_diffusion import LanguageTRMModule
from models.tokenizers.char import CharLevelTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current device is {device}")

# Load network
with hydra.initialize(version_base=None, config_path="config"):
    cfg = hydra.compose(config_name="trm_dllm")
cfg.learning.batch_size = 128
weights_path = "weights/trm_dllm.ckpt"
model: LanguageTRMModule = hydra.utils.instantiate(cfg.module)
model = load_inner_model_state_dict(model, weights_path).to(device).eval()  # type: ignore

tokenizer = CharLevelTokenizer()

# TEXT = "What is the meaning of life?"
TEXT = "The meaning of life is"
tokenized_text = tokenizer.encode(TEXT)
tokenized_text = torch.tensor(tokenized_text).unsqueeze(0)


def rows_used(text: str, width: int) -> int:
    total = 0
    for line in text.splitlines() or [""]:
        total += max(1, math.ceil(max(1, len(line.expandtabs())) / max(1, width)))
    return total


prev_rows = 0
with torch.no_grad():
    for generated_tokens in model.stream(
        init_tokens=tokenized_text,
        seq_len=1024,
        vocab_size=65,
        temperature=0.1,
        init_step=0,
    ):
        generated_text = tokenizer.decode(generated_tokens.squeeze().cpu().numpy())  # type: ignore

        if prev_rows:
            sys.stdout.write(f"\x1b[{prev_rows}F")

        sys.stdout.write("\x1b[J")
        sys.stdout.write(
            generated_text + ("\n" if not generated_text.endswith("\n") else "")
        )
        sys.stdout.flush()

        width = shutil.get_terminal_size().columns or 80
        prev_rows = rows_used(generated_text, width)

print()  # leave the cursor on a clean line at the end
