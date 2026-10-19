"""Fail before encoding a video if frames are missing, corrupt, or mismatched."""
from pathlib import Path
from PIL import Image
import numpy as np

with np.load('frames.npz') as d: expected=len(d['ft'])
for i in range(expected):
    p=Path('frames')/f'f_{i:04d}.png'
    with Image.open(p) as im:
        if im.size!=(1040,1220): raise ValueError(f'Unexpected dimensions: {p}')
        im.verify()
print(f'verified {expected} complete animation frames')
