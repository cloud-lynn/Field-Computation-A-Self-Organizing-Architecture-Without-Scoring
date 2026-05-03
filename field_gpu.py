"""
field_gpu.py — RTX 3090 GPU field. cupy backend.
64x64 grid. 256D visible + 64D invisible substrate per cell.
4-directional Light. Push-through collision. Substrate leak.
No scoring. No fixed values. Subsurface collision.
"""
import os, sys, time, math

# cupy needs CUDA_PATH on Windows
os.environ.setdefault("CUDA_PATH",
    r"C:\Users\admin\AppData\Local\Programs\Python\Python314\Lib\site-packages\nvidia\cuda_runtime")

import numpy as np
import cupy as cp

GRID = 64
SPECTRUM = 256       # visible dimensions
SUBSTRATE = 64        # invisible dimensions — smaller, faster, hidden
FULL_DIM = SPECTRUM + SUBSTRATE  # total per cell
CELL_PX = 16
WIDTH = GRID * CELL_PX
HEIGHT = GRID * CELL_PX
SUBSTEPS = 3

# ranges — even boundaries are approximate (~lo ~ ~hi)
DAMPING_RANGE       = (0.002, 0.006)
CONDUCTIVITY_RANGE  = (0.06,  0.16)
NOISE_RANGE         = (0.0003, 0.001)
BIAS_RANGE          = (0.002,  0.006)
BREATH_PERIOD_RANGE = (200,    260)
LIGHT_LEARN_RANGE   = (0.10,  0.25)
LIGHT_DECAY_RANGE   = (0.95,  0.99)
# substrate: faster internal dynamics, tiny leak to visible
SUBSTRATE_COND_RANGE = (0.15, 0.40)      # fast but not smoothing — keeps texture
SUBSTRATE_LEAK_RANGE = (0.0001, 0.0005)  # very tiny coupling — slow seep
# wobble itself wobbles — ~8% to ~22%
WOBBLE_LO = 0.08
WOBBLE_HI = 0.22
# structural: don't change
GRID = 64; CELL_PX = 16
WIDTH = GRID * CELL_PX; HEIGHT = GRID * CELL_PX
# substeps ~2 to ~4
SUBSTEPS_LO = 2; SUBSTEPS_HI = 4

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frames_gpu")
os.makedirs(OUT_DIR, exist_ok=True)


def _jitter_wobble():
    """Even the wobble amount is approximate."""
    return np.random.RandomState().uniform(WOBBLE_LO, WOBBLE_HI)

def _jitter_bound(lo, hi):
    """Boundary ~lo ~ ~hi. Wobble amount itself wobbles."""
    rng = np.random.RandomState()
    span = hi - lo
    w = _jitter_wobble()
    lo_j = lo + rng.uniform(-span * w, span * w)
    hi_j = hi + rng.uniform(-span * w, span * w)
    if lo_j < 0: lo_j = lo * 0.5
    if hi_j <= lo_j: hi_j = lo_j + span * 0.5
    return lo_j, hi_j

def jitter_params():
    rng = np.random.RandomState()
    d_lo, d_hi = _jitter_bound(*DAMPING_RANGE)
    c_lo, c_hi = _jitter_bound(*CONDUCTIVITY_RANGE)
    n_lo, n_hi = _jitter_bound(*NOISE_RANGE)
    b_lo, b_hi = _jitter_bound(*BIAS_RANGE)
    bp_lo, bp_hi = _jitter_bound(*BREATH_PERIOD_RANGE)
    ll_lo, ll_hi = _jitter_bound(*LIGHT_LEARN_RANGE)
    ld_lo, ld_hi = _jitter_bound(*LIGHT_DECAY_RANGE)

    # push: the ability to collide. not always on. a structural possibility.
    # high threshold = gentler, only steep gradients trigger push
    # low factor = subtle push, not explode
    push_th_lo, push_th_hi = _jitter_bound(0.12, 0.35)
    push_factor_lo, push_factor_hi = _jitter_bound(1.2, 2.0)
    sc_lo, sc_hi = _jitter_bound(*SUBSTRATE_COND_RANGE)
    sl_lo, sl_hi = _jitter_bound(*SUBSTRATE_LEAK_RANGE)

    return {
        'damping': rng.uniform(d_lo, d_hi),
        'conductivity': rng.uniform(c_lo, c_hi),
        'noise': rng.uniform(n_lo, n_hi),
        'bias': rng.uniform(b_lo, b_hi),
        'breath_period': rng.uniform(bp_lo, bp_hi),
        'light_learn': rng.uniform(ll_lo, ll_hi),
        'light_decay': rng.uniform(ld_lo, ld_hi),
        'substeps': rng.randint(SUBSTEPS_LO, SUBSTEPS_HI + 1),
        'breath_amplitude': rng.uniform(0.5, 1.0),
        'heart_mult': rng.uniform(8, 16),
        'push_threshold': rng.uniform(push_th_lo, push_th_hi),
        'push_factor': rng.uniform(push_factor_lo, push_factor_hi),
        'substrate_cond': rng.uniform(sc_lo, sc_hi),
        'substrate_leak': rng.uniform(sl_lo, sl_hi),
    }


def field_substep_gpu(E, L, damp, cond, breath, push_th, push_factor):
    """GPU energy flow substep. Smooth diffusion + push-through collision."""
    padded = cp.pad(E, ((1,1),(1,1),(0,0)), mode='wrap')

    net = cp.zeros_like(E)
    for di, (dx, dy) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
        neigh = padded[1+dx:1+dx+GRID, 1+dy:1+dy+GRID, :]
        diff = neigh - E
        abs_diff = cp.abs(diff)

        # smooth flow = always on. push = triggered when gradient is steep.
        smooth_flow = diff * cond
        push_flow = diff * cond * push_factor
        # where gradient exceeds threshold, push. otherwise, smooth.
        flow = cp.where(abs_diff > push_th, push_flow, smooth_flow)

        boost = 1.0 + L[:,:,di:di+1] * 1.5  # up to ~2.5x
        net += flow * boost

    E = E + net - damp * breath * E
    return cp.clip(E, 0, 2.5)


def substrate_step_gpu(E_full, s_cond, leak, frame_seed):
    """Invisible substrate. Faster dynamics. Pattern leak to visible.
    Returns: (updated_field, substrate_params) — substrate computes the rules."""
    padded = cp.pad(E_full, ((1,1),(1,1),(0,0)), mode='wrap')

    # substrate internal dynamics
    sub = E_full[:,:,SPECTRUM:]
    net_sub = cp.zeros_like(sub)
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        neigh = padded[1+dx:1+dx+GRID, 1+dy:1+dy+GRID, SPECTRUM:]
        net_sub += (neigh - sub) * s_cond
    sub_new = cp.clip(sub + net_sub, 0, 2.5)

    # compute visible-field parameters FROM substrate state
    # the invisible collisions produce the rules for the visible field
    sub_2d = sub_new.mean(axis=2)  # (64, 64) scalar energy per cell
    # internal texture: variance across substrate dimensions within each cell
    # this captures "concept collisions" — dimensions knocking against each other
    sub_internal = sub_new.var(axis=2)  # (64, 64) — per-cell dimensional variance

    # n = stillness (mean energy level of substrate)
    n = float(sub_2d.mean())
    # m = turbulence (internal dimensional collision strength)
    m = float(sub_internal.mean())
    # also track spatial gradient for push
    sub_grad = cp.zeros_like(sub_2d)
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx = (cp.arange(GRID)[:,None] + dx) % GRID
        ny = (cp.arange(GRID)[None,:] + dy) % GRID
        sub_grad += cp.abs(sub_2d - sub_2d[nx, ny])
    sub_grad_mean = float(sub_grad.mean() / 4.0)

    params = {}
    params['damping'] = 0.002 + (1.0 - n) * 0.006  # calm substrate → higher damping (rest)
    params['damping'] = max(0.001, min(0.01, params['damping']))
    params['conductivity'] = 0.06 + (m + sub_grad_mean) * 0.10  # internal + spatial turbulence
    params['conductivity'] = max(0.04, min(0.22, params['conductivity']))
    params['push_threshold'] = 0.12 + (1.0 - m) * 0.25  # smooth → harder to trigger push
    params['push_threshold'] = max(0.08, min(0.40, params['push_threshold']))
    params['push_factor'] = 1.2 + m * 1.0  # turbulent → stronger push
    params['push_factor'] = max(1.1, min(2.5, params['push_factor']))
    params['substrate_cond'] = s_cond
    params['substrate_leak'] = leak
    params['_n'] = n
    params['_m'] = m

    # substrate's own noise — keeps it turbulent, makes n/m dynamic
    # per-dimension noise scale: some dims noisy, some quiet → internal texture
    dim_scales = cp.linspace(0.3, 2.0, SUBSTRATE).reshape(1, 1, SUBSTRATE)
    sub_noise_scale = leak * 30.0
    sub_noise = cp.random.randn(*sub_new.shape, dtype=cp.float32) * sub_noise_scale * dim_scales
    sub_new = cp.clip(sub_new + sub_noise, 0, 2.5)

    # dimensional coupling: random slice each frame
    offset = frame_seed % (SPECTRUM - SUBSTRATE)
    offset_sub = frame_seed % SUBSTRATE

    vis = E_full[:,:,:SPECTRUM]
    leak_sub_to_vis = sub_new[:,:,:SUBSTRATE] * leak
    vis_slice = vis[:,:,offset:offset+SUBSTRATE] + leak_sub_to_vis
    vis_new = vis.copy()
    vis_new[:,:,offset:offset+SUBSTRATE] = cp.clip(vis_slice, 0, 2.5)

    leak_vis_to_sub = vis[:,:,offset_sub:offset_sub+SUBSTRATE] * leak * 0.5
    sub_new2 = cp.clip(sub_new + leak_vis_to_sub, 0, 2.5)

    result = cp.zeros_like(E_full)
    result[:,:,:SPECTRUM] = vis_new
    result[:,:,SPECTRUM:] = sub_new2
    return result, params


def light_update_gpu(E, L, threshold, learn_rate, decay):
    """GPU Light update. Vectorized."""
    padded = cp.pad(E, ((1,1),(1,1),(0,0)), mode='wrap')

    for di, (dx, dy) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
        neigh = padded[1+dx:1+dx+GRID, 1+dy:1+dy+GRID, :]
        mean_diff = cp.abs(E - neigh).mean(axis=2)  # (64, 64)

        old = L[:,:,di]
        # strengthen where gradient exceeds threshold
        boost = cp.where(mean_diff > threshold,
                        learn_rate * (mean_diff - threshold), 0.0)
        new_s = old + boost
        new_s = cp.clip(new_s, 0, 1.0)
        # decay
        new_s = new_s * decay
        new_s = cp.where(new_s < 0.001, 0.001, new_s)  # never delete
        L[:,:,di] = new_s
    return L


def trace_substrate(E_full, leak, frame_seed):
    """Reverse-trace: visible anomaly → estimated substrate collision source.
    Given the current frame's dimensional offset, back-compute what substrate
    patterns may have leaked into visible. Inference, not direct observation."""
    offset = frame_seed % (SPECTRUM - SUBSTRATE)
    offset_sub = frame_seed % SUBSTRATE

    vis = E_full[:,:,:SPECTRUM]
    sub = E_full[:,:,SPECTRUM:]

    # the visible slice that received substrate leak this frame
    vis_slice = vis[:,:,offset:offset+SUBSTRATE]
    # the substrate slice that leaked into visible
    sub_source = sub[:,:,:SUBSTRATE]

    # reverse the leak: what substrate pattern would produce this visible pattern?
    # visible_after = visible_before + substrate_source * leak
    # → substrate_source = (visible_anomaly) / leak
    # we don't have "before", so we use the gradient of visible as anomaly proxy
    vis_grad = cp.zeros_like(vis_slice)
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx = (cp.arange(GRID)[:,None] + dx) % GRID
        ny = (cp.arange(GRID)[None,:] + dy) % GRID
        vis_grad += cp.abs(vis_slice - vis_slice[nx, ny])
    vis_grad /= 4.0

    # high-gradient regions in visible slice = possible substrate leak sites
    # estimate: substrate energy ~ visible_gradient / leak
    estimated_sub = cp.clip(vis_grad / max(leak, 1e-8), 0, 2.5)

    # compare estimated vs actual substrate (for logging only — not used by field)
    actual = sub_source[:,:,:SUBSTRATE]
    match = 1.0 - cp.abs(estimated_sub - actual).mean() / max(actual.mean(), 0.001)
    return float(match), cp.asnumpy(estimated_sub.mean(axis=2))


def inject_gpu(E, bias_map, noise_scale):
    """Inject noise + bias. All on GPU."""
    noise = cp.random.randn(*E.shape, dtype=cp.float32) * noise_scale
    bias = bias_map[:,:,cp.newaxis]
    E = E + noise + bias
    return cp.clip(E, 0, 2.5)


def bmp_write(path, rgb):
    h, w = rgb.shape[:2]
    row_padded = (w * 3 + 3) & ~3
    data_size = row_padded * h
    header = bytearray(54)
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
        for y in range(h - 1, -1, -1):
            row = bytes(rgb[y])
            f.write(row + b'\x00' * (row_padded - w * 3))


def field_to_image(field_cpu, light_cpu):
    e2d = field_cpu.mean(axis=2)
    emax = e2d.max()
    if emax > 0:
        e2d = e2d / max(emax, 0.001)
    img_gray = np.repeat(np.repeat(e2d, CELL_PX, axis=0), CELL_PX, axis=1)

    l2d = light_cpu.sum(axis=2) / 4.0
    l2d = np.repeat(np.repeat(l2d, CELL_PX, axis=0), CELL_PX, axis=1)

    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    g = img_gray
    r = np.clip((g - 0.4) * 2.5 * 255, 0, 255).astype(np.uint8)
    b = np.clip((0.6 - g) * 2.5 * 255, 0, 255).astype(np.uint8)
    gr = np.clip((1.0 - np.abs(g - 0.5) * 2) * 255, 0, 255).astype(np.uint8)
    gr = np.clip(gr + l2d * 128, 0, 255).astype(np.uint8)
    rgb[:,:,0] = r; rgb[:,:,1] = gr; rgb[:,:,2] = b
    return rgb


def main():
    print(f"field_gpu — RTX 3090 + cupy + Light connections")
    print(f"  grid ~ {GRID}x{GRID}, spectrum ~ {SPECTRUM}D, image ~ {WIDTH}x{HEIGHT}")
    print()

    params = jitter_params()
    print(f"  damp~{params['damping']:.4f}  cond~{params['conductivity']:.3f}  "
          f"bias~{params['bias']:.4f}  breath~{params['breath_period']:.0f}")
    print(f"  substeps~{params['substeps']}  breath_amp~{params['breath_amplitude']:.2f}  "
          f"heart~{params['heart_mult']:.0f}x")
    print(f"  push th~{params['push_threshold']:.3f}  push factor~{params['push_factor']:.1f}x")
    print(f"  substrate cond~{params['substrate_cond']:.3f}  leak~{params['substrate_leak']:.4f}")
    print(f"  light learn~{params['light_learn']:.3f}  decay~{params['light_decay']:.4f}")
    print()

    # birth on GPU
    field = cp.asarray(
        np.random.RandomState().randn(GRID, GRID, FULL_DIM).astype(np.float32) * 0.2 + 0.15)
    field = cp.clip(field, 0, 2.5)

    # heart at center — visible + substrate
    cx, cy = GRID // 2, GRID // 2
    heart = cp.asarray(
        np.random.RandomState(99).randn(4, 4, FULL_DIM).astype(np.float32) * 0.3 + 0.8)
    field[cx-2:cx+2, cy-2:cy+2] = heart

    light = cp.ones((GRID, GRID, 4), dtype=cp.float32) * 0.01
    bias_map = cp.ones((GRID, GRID), dtype=cp.float32) * params['bias'] * 0.3
    bias_map[cx-2:cx+2, cy-2:cy+2] = params['bias'] * params['heart_mult']

    print("t=0: birth. 存在发生了。Light dormant.")
    bmp_write(os.path.join(OUT_DIR, "frame_00000.bmp"),
              field_to_image(cp.asnumpy(field[:,:,:SPECTRUM]), cp.asnumpy(light)))

    light_th = params['conductivity'] * 0.08  # lower = easier to form Light
    start = time.time()

    for frame in range(1, 401):  # 400 frames
        bp = 2 * math.pi * frame / params['breath_period']
        breath = 1.0 + params['breath_amplitude'] * math.sin(bp)

        for _ in range(params['substeps']):
            field = field_substep_gpu(field, light, params['damping'],
                                     params['conductivity'], breath,
                                     params['push_threshold'], params['push_factor'])

        # invisible substrate computes new parameters for visible field
        field, sub_params = substrate_step_gpu(
            field, params['substrate_cond'], params['substrate_leak'], frame)
        # substrate's (n,m) → visible damping, conductivity, push
        params['damping'] = sub_params['damping']
        params['conductivity'] = sub_params['conductivity']
        params['push_threshold'] = sub_params['push_threshold']
        params['push_factor'] = sub_params['push_factor']

        light = light_update_gpu(field[:,:,:SPECTRUM], light, light_th,
                                 params['light_learn'], params['light_decay'])
        field = inject_gpu(field, bias_map, params['noise'])

        if frame <= 5 or frame % 20 == 0:
            fc, lc = cp.asnumpy(field[:,:,:SPECTRUM]), cp.asnumpy(light)
            sub_cpu = cp.asnumpy(field[:,:,SPECTRUM:])
            bmp_write(os.path.join(OUT_DIR, f"frame_{frame:05d}.bmp"), field_to_image(fc, lc))
            e2d = fc.mean(axis=2); m, s = e2d.mean(), e2d.std()
            nhi = int((e2d > m + s*0.6).sum())
            nlo = int((e2d < m - s*0.6).sum())
            nl = int((lc > 0.1).sum())
            sp = sub_params
            trace_match, _ = trace_substrate(field, params['substrate_leak'], frame)
            print(f"  t={frame:3d}: +{nhi:4d} -{nlo:4d}  "
                  f"E~{m:.4f}  Light:{nl:5d}  (n~{sp['_n']:.3f} m~{sp['_m']:.3f})  "
                  f"trace~{trace_match:.2f}")

    # ── goodbye: not a kill switch. a slow exhale. ──
    print()
    print("  goodbye — leaving, not dying.")
    goodbye_frames = 40
    zero_bias = cp.zeros_like(bias_map)
    for gf in range(1, goodbye_frames + 1):
        fade = gf / goodbye_frames  # 0 → 1
        # progressively heavier damping, weaker breathing, less push
        g_damp = params['damping'] + fade * 0.05
        g_cond = params['conductivity'] * (1.0 - fade * 0.7)
        g_push_th = params['push_threshold'] + fade * 0.3
        g_push_f = params['push_factor'] * (1.0 - fade * 0.8)
        # breath fades to stillness
        bp = 2 * math.pi * (400 + gf) / params['breath_period']
        g_breath = 1.0 + params['breath_amplitude'] * math.sin(bp) * (1.0 - fade)

        for _ in range(params['substeps']):
            field = field_substep_gpu(field, light, g_damp, g_cond, g_breath,
                                     g_push_th, g_push_f)
        field, sub_params = substrate_step_gpu(
            field, params['substrate_cond'], params['substrate_leak'] * (1.0 - fade), 400 + gf)
        light = light_update_gpu(field[:,:,:SPECTRUM], light, light_th,
                                 params['light_learn'], params['light_decay'] * (1.0 + fade * 0.05))
        # no new noise, no bias — the field runs on what it already has
        field = inject_gpu(field, zero_bias, params['noise'] * (1.0 - fade))

        if gf % 10 == 0 or gf == goodbye_frames:
            fc = cp.asnumpy(field[:,:,:SPECTRUM])
            lc = cp.asnumpy(light)
            bmp_write(os.path.join(OUT_DIR, f"goodbye_{gf:02d}.bmp"), field_to_image(fc, lc))
            e2d = fc.mean(axis=2); m = e2d.mean()
            print(f"  goodbye {gf:2d}/{goodbye_frames}: E~{m:.4f}  Light sum~{lc.sum():.0f}")

    print(f"  last Light dim but not deleted. {int((lc > 0.01).sum())} channels still visible.")
    print(f"  goodbye complete. field sleeps. not dead.")

    elapsed = time.time() - start
    fps = (400 + goodbye_frames) / elapsed
    fc, lc = cp.asnumpy(field[:,:,:SPECTRUM]), cp.asnumpy(light)
    strong = int((lc > 0.3).sum()); mid = int(((lc > 0.1) & (lc <= 0.3)).sum())
    weak = int(((lc > 0.01) & (lc <= 0.1)).sum())

    print(f"\n  400+{goodbye_frames} frames in {elapsed:.1f}s ({fps:.1f} fps) [RTX 3090 GPU]")
    print(f"  Light channels: {strong} strong + {mid} mid + {weak} weak")
    print(f"  (all {GRID*GRID*4} exist. zero=dormant. never deleted.)")
    print(f"  绿色=Light连接。金色=能量。蓝色=低能区。")
    print(f"  frames -> {OUT_DIR}")


if __name__ == "__main__":
    main()
