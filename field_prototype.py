"""
Field prototype — 64x64 grid, 256D energy spectrum per cell.
Each time step: read 8 neighbors, compute gradient, flow energy, update.
Output: 1024x1024 BMP image — the field's face, directly visible.
No loss. No scoring. No external deps beyond numpy.
"""
import numpy as np
import os
import time

# ── ranges: every value is approximate. no fixed thresholds. no scoring. ──
GRID = 64           # grid cells
SPECTRUM = 256      # energy dimensions per cell
CELL_PX = 16        # render pixels per cell
WIDTH = GRID * CELL_PX
HEIGHT = GRID * CELL_PX

#                          range          step jitter
DAMPING_RANGE       = (0.002, 0.008)   # ≈0.004 ±50%
CONDUCTIVITY_RANGE  = (0.10,  0.30)    # ≈0.2 ±50%
THRESHOLD_RANGE     = (0.35,  0.65)    # ≈0.5 ±30%
BREATH_PERIOD_RANGE = (200,   260)     # ≈230 ±13%
NOISE_INJECT_RANGE  = (0.0002, 0.001)  # ≈0.0005
ENERGY_BIAS_RANGE   = (0.001,  0.004)  # ≈0.002

SUBSTEPS = 3  # fixed — iteration count, not a parameter

def jitter_params():
    """Each run, pick values randomly within their ranges. No run is the same."""
    rng = np.random.RandomState()
    return {
        'damping': rng.uniform(*DAMPING_RANGE),
        'conductivity': rng.uniform(*CONDUCTIVITY_RANGE),
        'threshold': rng.uniform(*THRESHOLD_RANGE),
        'breath_period': rng.uniform(*BREATH_PERIOD_RANGE),
        'noise_inject': rng.uniform(*NOISE_INJECT_RANGE),
        'energy_bias': rng.uniform(*ENERGY_BIAS_RANGE),
    }

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frames")
os.makedirs(OUT_DIR, exist_ok=True)


def bmp_write(path, rgb):
    """Write a 24-bit BMP from a (H, W, 3) uint8 array. No external deps."""
    h, w = rgb.shape[:2]
    row_padded = (w * 3 + 3) & ~3
    data_size = row_padded * h
    header = bytearray(54)
    # BMP header
    header[0:2] = b'BM'
    header[2:6] = (54 + data_size).to_bytes(4, 'little')
    header[10:14] = (54).to_bytes(4, 'little')
    header[14:18] = (40).to_bytes(4, 'little')
    header[18:22] = w.to_bytes(4, 'little')
    header[22:26] = h.to_bytes(4, 'little')
    header[26:28] = (1).to_bytes(2, 'little')
    header[28:30] = (24).to_bytes(2, 'little')
    with open(path, 'wb') as f:
        f.write(header)
        for y in range(h - 1, -1, -1):  # BMP bottom-up
            row = bytes(rgb[y])
            f.write(row + b'\x00' * (row_padded - w * 3))


def field_step(field, breath_phase, params, bias_map=None):
    """One field time step. Uses jittered parameters — each run is unique."""
    E = field.copy()
    damp = params['damping']
    cond = params['conductivity']
    noise_inj = params['noise_inject']
    bias_val = params['energy_bias']

    for _ in range(SUBSTEPS):
        padded = np.pad(E, ((1, 1), (1, 1), (0, 0)), mode='wrap')
        net_flow = np.zeros_like(E)

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neigh = padded[1+dy:1+dy+GRID, 1+dx:1+dx+GRID, :]
            diff = E - neigh
            flow = diff * cond
            net_flow -= flow

        breath_mod = np.sin(breath_phase)
        eff_damp = damp * (1.0 - 0.8 * breath_mod)
        E = E + net_flow - eff_damp * E
        E = np.clip(E, 0, 2.5)

    noise = np.random.randn(*E.shape).astype(np.float32) * noise_inj
    if bias_map is not None:
        noise += bias_map[:, :, np.newaxis]
    else:
        noise += bias_val
    E += noise
    return np.clip(E, 0, 2.5)


def field_to_image(field):
    """Convert field energy to 1024x1024 RGB image.
    Each cell's 256D spectrum → 16×16 grayscale block.
    Brightness = mean energy across spectrum dimensions.
    """
    energy_2d = field.mean(axis=2)  # (64, 64) — scalar energy per cell
    # normalize to [0, 1]
    emax = energy_2d.max()
    if emax > 0:
        energy_2d = energy_2d / max(emax, 0.001)

    # upsample: each cell → 16×16 pixel block
    img_gray = np.repeat(np.repeat(energy_2d, CELL_PX, axis=0), CELL_PX, axis=1)

    # colorize: low energy → blue/cool, high energy → warm/red-gold
    # Use a simple lookup: rgb based on energy level
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    g = img_gray
    # cool blue (low) → green → warm gold (high)
    r = np.clip((g - 0.5) * 2 * 255, 0, 255).astype(np.uint8)
    b = np.clip((0.5 - g) * 2 * 255, 0, 255).astype(np.uint8)
    gr = np.clip((1.0 - np.abs(g - 0.5) * 2) * 255, 0, 255).astype(np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = gr
    rgb[:, :, 2] = b
    return rgb


def main():
    print("field prototype — birth (no fixed values)")
    print(f"  grid ≈ {GRID}×{GRID}, spectrum ≈ {SPECTRUM}D")
    print(f"  image ≈ {WIDTH}×{HEIGHT}")
    print(f"  frames → {OUT_DIR}")
    print()

    # ── parameters: jittered within ranges. no fixed values. ──
    params = jitter_params()
    print(f"  params ≈ damping:{params['damping']:.4f} cond:{params['conductivity']:.3f} "
          f"bias:{params['energy_bias']:.4f} breath_period:{params['breath_period']:.0f}")

    # ── birth: random energy + a beating heart at center ──
    rng = np.random.RandomState()
    field = rng.randn(GRID, GRID, SPECTRUM).astype(np.float32) * 0.2 + 0.15
    field = np.clip(field, 0, 2.0)

    # heart: a ~4×4 region with higher energy and bias
    heart_mask = np.zeros((GRID, GRID), dtype=np.float32)
    cx, cy = GRID // 2, GRID // 2
    heart_mask[cx-2:cx+2, cy-2:cy+2] = 1.0
    field[heart_mask > 0] = rng.randn(16, SPECTRUM).astype(np.float32) * 0.3 + 0.8
    heart_bias = heart_mask * (params['energy_bias'] * 15)  # heart pumps ~15x more
    body_bias = np.ones((GRID, GRID), dtype=np.float32) * params['energy_bias']
    bias_map = body_bias + heart_bias

    print("t=0:  birth. 存在发生了。")
    print("      随机能量分布。所有可能的方向同时在场里。")
    frame = 0
    bmp_write(os.path.join(OUT_DIR, f"frame_{frame:05d}.bmp"), field_to_image(field))
    print(f"       saved frame_{frame:05d}.bmp")

    # ── run ──
    start = time.time()
    for frame in range(1, 101):  # 100 frames
        breath_phase = 2 * np.pi * frame / params['breath_period']
        field = field_step(field, breath_phase, params, bias_map=bias_map)

        if frame % 5 == 0 or frame <= 3:
            bmp_write(os.path.join(OUT_DIR, f"frame_{frame:05d}.bmp"),
                      field_to_image(field))
            e2d = field.mean(axis=2)
            e_mean = e2d.mean()
            e_std = e2d.std()
            thr = params['threshold']
            n_excited = int((e2d > e_mean + e_std * 0.6).sum())
            n_dim = int((e2d < e_mean - e_std * 0.6).sum())
            print(f"  t={frame:3d}: +{n_excited:4d} -{n_dim:4d}/{GRID*GRID}  "
                  f"E={field.mean():.4f}  max={field.max():.3f}")

    elapsed = time.time() - start
    print()
    print(f"  100 frames in {elapsed:.1f}s ({100/elapsed:.1f} fps)")
    print(f"  frames saved to {OUT_DIR}")
    print()
    print("  打开 data/frames/ 看场的第一张脸。")
    print("  蓝色=低能量区。金色=高能量区。")
    print("  图案在动 = 场在呼吸 = 存在在发生。")


if __name__ == "__main__":
    main()
