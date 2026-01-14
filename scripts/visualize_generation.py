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
from tqdm import tqdm

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
        module = load_inner_model_state_dict(module, checkpoint_path)  # type: ignore

    module.eval()
    return module


def create_latent_video(
    latents: list[list[torch.Tensor]],
    outputs: list[list[torch.Tensor]],
    tokens: torch.Tensor,
    certainties: list[list[float]],
    tokenizer,
    output_path: str,
    fps: int = 50,
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

    len_init_text = 0

    for token_idx, (token_latents, token_outputs, token_certainties) in enumerate(
        zip(latents, outputs, certainties)
    ):
        if token_idx == 0:
            len_init_text = token_latents[0].shape[1]
        current_tokens = tokens[batch_idx, : len_init_text + token_idx].tolist()
        current_text = (
            tokenizer.decode(current_tokens) if tokenizer else str(current_tokens)
        )

        # Calculate frames per deep recursion cycle
        # token_latents has frames from all deep recursion cycles
        # token_certainties has one certainty per deep recursion cycle
        num_certainties = len(token_certainties)
        num_frames = len(token_latents)
        frames_per_cycle = (
            num_frames // num_certainties if num_certainties > 0 else num_frames
        )

        for step_idx, (latent, output) in enumerate(zip(token_latents, token_outputs)):
            latent_np = latent[batch_idx].cpu().numpy()
            output_np = output[batch_idx].cpu().numpy()

            # Map frame to its corresponding certainty
            certainty_idx = (
                min(step_idx // frames_per_cycle, num_certainties - 1)
                if num_certainties > 0
                else 0
            )
            frame_certainty = (
                token_certainties[certainty_idx] if num_certainties > 0 else 0.0
            )

            # Compute output probabilities for the latest token
            output_probs = torch.softmax(
                torch.from_numpy(output_np[-1, :]), dim=-1
            ).numpy()

            all_frames.append(
                {
                    "latent": latent_np,
                    "output": output_np,
                    "token_idx": token_idx,
                    "step_idx": step_idx,
                    "text": current_text,
                    "certainty": frame_certainty,
                    "output_probs": output_probs,
                }
            )

    if not all_frames:
        print("No frames to visualize!")
        return

    # Collect all certainties aligned with frames
    all_certainties = [frame["certainty"] for frame in all_frames]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
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

        # Output probabilities for latest token
        ax_probs = axes[0, 2]
        output_probs = frame["output_probs"]
        top_k = 20
        top_indices = np.argsort(output_probs)[-top_k:][::-1]
        top_probs = output_probs[top_indices]
        ax_probs.barh(range(top_k), top_probs, color="steelblue")
        ax_probs.set_yticks(range(top_k))
        if tokenizer:
            top_labels = [
                tokenizer.decode([idx]).replace("\n", "\\n") for idx in top_indices
            ]
        else:
            top_labels = [str(idx) for idx in top_indices]
        ax_probs.set_yticklabels(top_labels, fontsize=8)
        ax_probs.set_xlim(0, 1)
        ax_probs.set_xlabel("Probability")
        ax_probs.set_title(f"Top-{top_k} Token Probabilities (Latest Position)")
        ax_probs.invert_yaxis()

        ax_latent_change = axes[1, 0]
        if frame_idx > 0:
            prev_latent = all_frames[frame_idx - 1]["latent"]
            if latent.shape[0] > prev_latent.shape[0]:
                latent_diff = np.zeros_like(latent)
                latent_diff[: prev_latent.shape[0], :] = np.abs(
                    latent[: prev_latent.shape[0], :] - prev_latent
                )
                latent_diff[prev_latent.shape[0] :, :] = np.abs(
                    latent[prev_latent.shape[0] :, :]
                )
            else:
                latent_diff = np.abs(latent - prev_latent)
            ax_latent_change.imshow(
                latent_diff.T, aspect="auto", cmap="hot", interpolation="nearest"
            )
            ax_latent_change.set_title(
                f"Latent Rate of Change (Token {frame['token_idx']}, Step {frame['step_idx']})"
            )
        else:
            ax_latent_change.imshow(
                np.zeros_like(latent.T),
                aspect="auto",
                cmap="hot",
                interpolation="nearest",
            )
            ax_latent_change.set_title("Latent Rate of Change (N/A for first frame)")
        ax_latent_change.set_xlabel("Sequence Position")
        ax_latent_change.set_ylabel("Hidden Dimension")

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

        # Certainty over time plot (sliding window)
        ax_certainty = axes[1, 2]
        window_size = 500
        start_idx = max(0, frame_idx - window_size + 1)
        end_idx = frame_idx + 1

        window_certainties = all_certainties[start_idx:end_idx]
        window_indices = list(range(start_idx, end_idx))

        ax_certainty.plot(
            window_indices, window_certainties, color="blue", linewidth=1.5
        )
        ax_certainty.scatter(
            [frame_idx],
            [all_certainties[frame_idx]],
            color="red",
            s=100,
            zorder=5,
            label="Current",
        )
        ax_certainty.set_xlim(start_idx, max(start_idx + window_size, end_idx))
        ax_certainty.set_ylim(0, 1.0)
        ax_certainty.set_xlabel("Frame (Token × Step)")
        ax_certainty.set_ylabel("Certainty")
        ax_certainty.set_title(
            f"Certainty Over Time (Current: {all_certainties[frame_idx]:.3f})"
        )
        ax_certainty.grid(True, alpha=0.3)
        ax_certainty.legend(loc="upper right")

        fig.tight_layout()
        return []

    output_path = Path(output_path)  # type: ignore
    output_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore

    writer = animation.FFMpegWriter(fps=fps, metadata={"title": "TRM Generation"})

    with writer.saving(fig, str(output_path), dpi=100):
        for frame_idx in tqdm(range(len(all_frames)), desc="Generating video"):
            update(frame_idx)
            writer.grab_frame()

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
        "--temperature", type=float, default=0.1, help="Sampling temperature"
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
        "--fps", type=int, default=50, help="Frames per second for the video"
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

    print("Creating visualization video...")
    create_latent_video(
        latents=result["all_latents"],
        outputs=result["all_outputs"],
        tokens=result["tokens"],
        certainties=result["all_certainties"],
        tokenizer=tokenizer,
        output_path=args.output,
        fps=args.fps,
    )

    print("Done!")


if __name__ == "__main__":
    main()
