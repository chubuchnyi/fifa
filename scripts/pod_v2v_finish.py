#!/usr/bin/env python3
"""Structure-locked v2v finishing spike (research priority 1, 2026-07-03).

Restyles rendered deliverable frames into photoreal night-broadcast look while
keeping OUR geometry as the structure lock (R-6: the model re-skins appearance,
it does not invent motion):

  render PNGs -> depth maps (Depth-Anything-V2 on the CG frames — geometry is
  ours, so estimated depth of a clean render is reliable) -> Wan2.1-VACE
  (control video = depth, reference image = a source-clip frame for the night
  look, text prompt for the broadcast style) -> mp4.

Runs on the pod inside the genfinish venv (see scripts/pod_v2v.sh).
"""

import argparse
import glob
import os
import sys

# Colour wording is load-bearing (measured 2026-07-04, four times): whatever surface colour
# the prompt does NOT state, Wan repaints from its prior, overriding a clip-matched control —
# and whatever colour it DOES state, Wan amplifies past the words. "deep green pitch" turned
# H-82 grass into H-120 emerald; an uncoloured "packed crowd" turned the H-69 warm crowd into
# H-200 cold gray; "warm yellow and amber shirts" turned a clip-exact S-.49 stand into S-.94
# glowing panels, while "muted dark amber ... half in shadow" landed S .74; a bare "yellow
# jerseys / cyan jerseys" turned kit-zoned bodies into shirtless-skin torsos, while spelling
# out the full kit (shirts+shorts+socks per team) restored every shirt (batch #2 A/B). State
# the measured look of EVERY large surface AT its measured intensity, push failure colours
# into the negative.
DEFAULT_PROMPT = (
    "Professional television broadcast of a floodlit night football match. "
    "Dark stadium bowl at night, bright white floodlights, muted yellow-green "
    "night grass with faint mowing stripes, one team in yellow short-sleeved "
    "shirts with bright white shorts and deep red socks, the other team in an all "
    "sky-blue kit with sky-blue shirts, sky-blue shorts and sky-blue socks, "
    "dark-skinned players, distant dark stands densely packed with "
    "thousands of tiny individual fans in muted dark amber and brown clothing, "
    "crowd dimly lit and half in shadow behind advertising boards, long-lens "
    "broadcast camera, photorealistic, sharp, high detail."
)
# "red trousers / shirtless" = batch #2 tail-1 failure modes (socks smeared into trousers,
# torsos painted as bare skin). No "yellow …" terms here — negatives bleed compositionally
# and would dim the yellow shirts.
DEFAULT_NEGATIVE = (
    "cartoon, anime, illustration, CGI render, video game, daylight, blue sky, "
    "vivid emerald grass, oversaturated colors, "
    "red trousers, red leggings, shirtless players, bare chest, "
    "blurry, low quality, distorted bodies, extra limbs, merged players, "
    "text, watermark, static image"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", required=True,
                   help="directory of rendered frame_*.png for ONE camera")
    p.add_argument("--ref-image", required=True,
                   help="source-clip frame (night-look reference)")
    p.add_argument("--out", required=True, help="output mp4 path")
    p.add_argument("--model", default="Wan-AI/Wan2.1-VACE-1.3B-diffusers")
    p.add_argument("--depth-model",
                   default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--control", choices=["depth", "gray", "rgb"],
                   default="depth",
                   help="what the structure-lock control video is built from")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--num-frames", type=int, default=57,
                   help="must be 4k+1 for Wan")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--flow-shift", type=float, default=3.0,
                   help="UniPC flow shift: 3.0 for 480p, 5.0 for 720p+")
    p.add_argument("--conditioning-scale", type=float, default=1.0,
                   help="VACE control-stream strength (diffusers >=0.34)")
    p.add_argument("--fps", type=float, default=25.0,
                   help="output fps (matches the deliverable, not Wan's 16)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative", default=DEFAULT_NEGATIVE)
    p.add_argument("--no-ref", action="store_true",
                   help="skip the reference image (prompt-only restyle)")
    p.add_argument("--save-control", default="",
                   help="optional dir to dump the control frames for eyeballing")
    return p.parse_args(argv)


def load_render_frames(frames_dir, num_frames, size):
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if len(paths) < num_frames:
        sys.exit(f"need {num_frames} frames, found {len(paths)} in {frames_dir}")
    frames = [Image.open(p).convert("RGB").resize(size, Image.LANCZOS)
              for p in paths[:num_frames]]
    print(f"== loaded {len(frames)} render frames from {frames_dir}", flush=True)
    return frames


def build_control(frames, mode, depth_model):
    if mode == "rgb":
        return frames
    if mode == "gray":
        return [f.convert("L").convert("RGB") for f in frames]
    import torch
    from transformers import pipeline as hf_pipeline
    device = 0 if torch.cuda.is_available() else -1
    depth = hf_pipeline("depth-estimation", model=depth_model, device=device)
    control = []
    for i, f in enumerate(frames):
        d = depth(f)["depth"].convert("RGB").resize(f.size)
        control.append(d)
        if i % 10 == 0:
            print(f"== depth {i + 1}/{len(frames)}", flush=True)
    del depth
    if device == 0:
        torch.cuda.empty_cache()
    print("== control video ready (depth)", flush=True)
    return control


def main(argv=None):
    args = parse_args(argv)
    if (args.num_frames - 1) % 4 != 0:
        sys.exit(f"--num-frames must be 4k+1, got {args.num_frames}")

    import torch
    from PIL import Image

    size = (args.width, args.height)
    frames = load_render_frames(args.frames_dir, args.num_frames, size)
    control = build_control(frames, args.control, args.depth_model)
    if args.save_control:
        os.makedirs(args.save_control, exist_ok=True)
        for i, c in enumerate(control):
            c.save(os.path.join(args.save_control, f"control_{i:04d}.png"))
        print(f"== control frames dumped to {args.save_control}", flush=True)

    ref_images = None
    if not args.no_ref:
        ref = Image.open(args.ref_image).convert("RGB").resize(size, Image.LANCZOS)
        ref_images = [ref]
        print(f"== reference: {args.ref_image}", flush=True)

    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline

    print(f"== loading {args.model}", flush=True)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae",
                                           torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(args.model, vae=vae,
                                           torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift)
    pipe.enable_model_cpu_offload()

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    print(f"== generating: {args.num_frames}f {args.width}x{args.height} "
          f"steps={args.steps} guidance={args.guidance} control={args.control} "
          f"cs={args.conditioning_scale} "
          f"ref={'yes' if ref_images else 'no'}", flush=True)
    result = pipe(
        video=control,
        reference_images=ref_images,
        prompt=args.prompt,
        negative_prompt=args.negative,
        conditioning_scale=args.conditioning_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
    ).frames[0]

    from diffusers.utils import export_to_video
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    export_to_video(result, args.out, fps=args.fps)
    print(f"V2V_OK {len(result)}f -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
