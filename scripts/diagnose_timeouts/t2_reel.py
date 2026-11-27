"""Concatenate chosen t2_failures.py clips into one MP4 with a caption burned into each.

    python scripts/diagnose_timeouts/t2_reel.py <out.mp4> <clip.mp4> "<caption>" [<clip.mp4> "<caption>" ...]
"""
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

out, pairs = sys.argv[1], list(zip(sys.argv[2::2], sys.argv[3::2]))
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
frames = []
for path, caption in pairs:
    for frame in iio.imread(path):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, frame.shape[1], 34], fill=(0, 0, 0))
        draw.text((10, 6), caption, font=font, fill=(255, 255, 255))
        frames.append(np.asarray(image))
    frames += [frames[-1]] * 25  # half-second hold on the last frame
iio.imwrite(out, np.stack(frames), fps=50, macro_block_size=8)
print(f"{out}: {len(pairs)} clips, {len(frames)} frames")
