"""
Script to visualize TRM latents and outputs during generation as a matplotlib video.
"""

import argparse
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from models.modules.base import load_inner_model_state_dict
from models.modules.trm_ar import ARLanguageTRMModule

OmegaConf.register_new_resolver("sum", lambda *args: sum(args))


def load_module(
    config_path: str, checkpoint_path: str | None = None
) -> ARLanguageTRMModule:
    """Load the TRM module from config and optionally a checkpoint."""
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg: DictConfig = hydra.compose(config_name=config_path)

    module: ARLanguageTRMModule = hydra.utils.instantiate(
        cfg.module, _convert_="partial"
    )

    if checkpoint_path is not None:
        module = load_inner_model_state_dict(module, checkpoint_path)

    module.eval()
    return module


def create_latent_video(
    latents: list[list[torch.Tensor]],
    outputs: list[list[torch.Tensor]],
    tokens: torch.Tensor,
    tokenizer,
    output_path: str,
    fps: int = 10,
    batch_idx: int = 0,
) -> None:
    """
    Create a matplotlib animation video from latents and outputs.

    Args:
        latents: Nested list of latent tensors [token_step][recursion_step]
        outputs: Nested list of output tensors [token_step][recursion_step]
        tokens: Generated token tensor
        tokenizer: Tokenizer for decoding tokens
        output_path: Path to save the video
        fps: Frames per second
        batch_idx: Which batch element to visualize
    """
    all_frames = []
    token_texts = []

    for token_idx, (token_latents, token_outputs) in enumerate(zip(latents, outputs)):
        current_tokens = tokens[batch_idx, : token_idx + 1].tolist()
        current_text = (
            tokenizer.decode(current_tokens) if tokenizer else str(current_tokens)
        )

        for step_idx, (latent, output) in enumerate(zip(token_latents, token_outputs)):
            latent_np = latent[batch_idx].cpu().numpy()
            output_np = output[batch_idx].cpu().numpy()
            all_frames.append(
                {
                    "latent": latent_np,
                    "output": output_np,
                    "token_idx": token_idx,
                    "step_idx": step_idx,
                    "text": current_text,
                }
            )

    if not all_frames:
        print("No frames to visualize!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TRM Generation Visualization", fontsize=14)

    def update(frame_idx: int):
        frame = all_frames[frame_idx]
        latent = frame["latent"]
        output = frame["output"]

        for ax in axes.flat:
            ax.clear()

        ax_latent_heat = axes[0, 0]
        im1 = ax_latent_heat.imshow(
            latent.T, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        ax_latent_heat.set_title(
            f"Latent Heatmap (Token {frame['token_idx']}, Step {frame['step_idx']})"
        )
        ax_latent_heat.set_xlabel("Sequence Position")
        ax_latent_heat.set_ylabel("Hidden Dimension")

        ax_output_heat = axes[0, 1]
        im2 = ax_output_heat.imshow(
            output.T, aspect="auto", cmap="plasma", interpolation="nearest"
        )
        ax_output_heat.set_title(
            f"Output Heatmap (Token {frame['token_idx']}, Step {frame['step_idx']})"
        )
        ax_output_heat.set_xlabel("Sequence Position")
        ax_output_heat.set_ylabel("Hidden Dimension")

        ax_latent_norm = axes[1, 0]
        latent_norms = np.linalg.norm(latent, axis=1)
        ax_latent_norm.bar(range(len(latent_norms)), latent_norms, color="steelblue")
        ax_latent_norm.set_title("Latent L2 Norm per Position")
        ax_latent_norm.set_xlabel("Sequence Position")
        ax_latent_norm.set_ylabel("L2 Norm")

        ax_text = axes[1, 1]
        ax_text.axis("off")
        wrapped_text = "\n".join(
            [frame["text"][i : i + 60] for i in range(0, len(frame["text"]), 60)]
        )
        ax_text.text(
            0.5,
            0.5,
            wrapped_text,
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax_text.set_title("Generated Text So Far")

        fig.tight_layout()
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=len(all_frames), interval=1000 // fps, blit=False
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = animation.FFMpegWriter(fps=fps, metadata={"title": "TRM Generation"})
    anim.save(str(output_path), writer=writer)
    print(f"Video saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TRM generation latents and outputs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="trm_ar",
        help="Config name (without .yaml extension)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/trm_ar.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The meaning of life is ",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generation_video.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for the video"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    args = parser.parse_args()

    print(f"Loading module from config: {args.config}")
    module = load_module(args.config, args.checkpoint)
    module = module.to(args.device)

    tokenizer = module._tokenizer
    if tokenizer is None:
        raise ValueError("Module does not have a tokenizer configured")

    print(f"Tokenizing prompt: {args.prompt}")
    input_tokens = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([input_tokens]).to(args.device)

    print(f"Generating sequence (max length: {args.max_length})...")
    result = module.verbose_generate(
        input_tensor,
        max_seq_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_text = tokenizer.decode(result["tokens"][0].tolist())
    print(f"Generated text: {generated_text}")

    print(f"Creating visualization video...")
    create_latent_video(
        latents=result["all_latents"],
        outputs=result["all_outputs"],
        tokens=result["tokens"],
        tokenizer=tokenizer,
        output_path=args.output,
        fps=args.fps,
    )

    print("Done!")


if __name__ == "__main__":
    main()
