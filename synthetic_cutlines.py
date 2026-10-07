"""Reproducible synthetic resist images and cutline CD measurements.

Run: .venv/bin/python synthetic_cutlines.py
Coordinates are zero-based (x=column, y=row), with y increasing downwards.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from dataclasses import asdict

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.special import expit

ROOT = Path(__file__).resolve().parent


def load_model_api():
    namespace = {"__name__": __name__}
    notebook = json.loads((ROOT / "黑盒优化.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
        if cell["id"] == "resist-model":
            break
    return namespace


def synthetic_make_ib(image, b, sign):
    """Assumed synthetic formula, not a calibrated physical resist law."""
    return np.maximum(image - b if sign == "positive" else b - image, 0.0)


def cutline_cd(prediction, x0, y0, x1, y1, threshold, sample_step=0.25):
    """Return CD and edge distances along a cutline with exactly two crossings.

    Sample bilinearly, then linearly interpolate threshold crossings. Reject
    missing/ambiguous edges rather than silently changing the measured feature.
    """
    prediction = np.asarray(prediction, dtype=float)
    if prediction.ndim != 2 or not np.all(np.isfinite(prediction)):
        raise ValueError("prediction must be a finite 2-D array")
    if not np.all(np.isfinite([x0, y0, x1, y1, threshold, sample_step])):
        raise ValueError("cutline parameters must be finite")
    height, width = prediction.shape
    if not (0 <= x0 <= width - 1 and 0 <= x1 <= width - 1
            and 0 <= y0 <= height - 1 and 0 <= y1 <= height - 1):
        raise ValueError("cutline endpoints must lie inside the image")
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length == 0 or sample_step <= 0:
        raise ValueError("cutline length and sample_step must be positive")
    distance = np.linspace(0, length, int(np.ceil(length / sample_step)) + 1)
    fraction = distance / length
    profile = map_coordinates(
        prediction, [y0 + fraction * (y1 - y0), x0 + fraction * (x1 - x0)],
        order=1, prefilter=False,
    )
    delta = profile - threshold
    # Exact threshold samples (including plateaus) count once per contiguous run.
    if delta[0] == 0 or delta[-1] == 0:
        raise ValueError("threshold crossing at a cutline endpoint")
    edges = []
    i = 0
    while i < len(delta) - 1:
        if delta[i] == 0:
            end = i
            while end + 1 < len(delta) and delta[end + 1] == 0:
                end += 1
            if end != i:
                raise ValueError("ambiguous threshold plateau")
            if delta[i - 1] * delta[i + 1] < 0:
                edges.append(float(distance[i]))
            i = end + 1
            continue
        if delta[i] * delta[i + 1] < 0:
            edges.append(float(distance[i] - delta[i] *
                               (distance[i + 1] - distance[i]) / (delta[i + 1] - delta[i])))
        i += 1
    if len(edges) != 2:
        raise ValueError(f"expected exactly two threshold crossings, got {len(edges)}")
    return edges[1] - edges[0], edges[0], edges[1]


def bind_cutline_gauges(gauges):
    """Freeze CSV gauge definitions and return CD residuals for make_objective."""
    frozen = [dict(g) for g in gauges]
    def residuals(prediction):
        result = []
        for gauge in frozen:
            try:
                cd, _, _ = cutline_cd(prediction, *[
                    float(gauge[key]) for key in
                    ("x0_px", "y0_px", "x1_px", "y1_px", "threshold", "sample_step_px")
                ])
            except ValueError as exc:
                raise ValueError(f"{gauge['gauge_id']}: {exc}") from exc
            result.append(cd - float(gauge["measured_cd_px"]))
        return np.array(result)
    return residuals


def generate(output_dir=ROOT / "data" / "cutline_case", table_output=None):
    api = load_model_api()
    Term, build = api["Term"], api["build"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261007)
    terms = []
    for i in range(50):
        k = (i // 2) % 2
        # Signed negative-branch weights create a clear bright-feature contour.
        sign = "positive" if i % 2 == 0 else "negative"
        c = float(rng.uniform(0.02, 0.08) if k == 0 else rng.uniform(0.5, 2.0))
        terms.append(Term(sign, float(rng.uniform(.15, .85)), k,
                          int(rng.choice([1, 2])), float(rng.uniform(.5, 4)),
                          3.0, c if sign == "positive" else -c)) # type: ignore

    yy, xx = np.mgrid[:320, :320]
    image = np.zeros((320, 320)) + .03
    cutlines = []
    for tile in range(25):
        cy, cx = (tile // 5) * 64 + 32, (tile % 5) * 64 + 32
        half_x, half_y = rng.uniform(8, 18, 2)
        blur = float(rng.uniform(.8, 2.2))
        angle = float(rng.uniform(-.25, .25))
        u = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        v = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        kind = "ellipse" if tile % 3 == 0 else "rectangle"
        if kind == "ellipse":
            shape = expit((1 - np.sqrt((u / half_x)**2 + (v / half_y)**2))
                          * min(half_x, half_y) / blur)
        else:
            shape = expit((half_x - np.abs(u)) / blur) * expit((half_y - np.abs(v)) / blur)
        image += float(rng.uniform(.8, .95)) * shape
        for axis in ("horizontal", "vertical"):
            # 25 features * 2 orientations * 80 distinct offsets = 4000 gauges.
            for offset in np.linspace(-.45, .45, 80):
                if axis == "horizontal":
                    line = (cx - 27, cy + offset * half_y, cx + 27, cy + offset * half_y)
                else:
                    line = (cx + offset * half_x, cy - 27, cx + offset * half_x, cy + 27)
                cutlines.append((tile, kind, axis, line))

    # The delivered 16-bit PNG is the authoritative input, including quantization.
    input_path = output_dir / "input_image.png"
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 65535).astype(np.uint16)).save(input_path)
    image = np.array(Image.open(input_path), dtype=np.float64) / 65535.0
    model = build(terms, make_Ib=synthetic_make_ib) # type: ignore
    prediction = model(image)
    np.save(output_dir / "ideal_response.npy", prediction)
    # Fixed threshold halfway between analytical uniform-dark/uniform-bright limits.
    dark = sum(t.c * t.b for t in terms if t.k == 0 and t.sign == "negative")
    bright = sum(t.c * (1 - t.b) for t in terms if t.k == 0 and t.sign == "positive")
    threshold = float((dark + bright) / 2)
    rows = []
    for i, (tile, kind, axis, line) in enumerate(cutlines):
        cd, e0, e1 = cutline_cd(prediction, *line, threshold) # type: ignore
        x0, y0, x1, y1 = map(float, line)
        length = np.hypot(x1 - x0, y1 - y0)
        rows.append(dict(
            gauge_id=f"G{i + 1:03d}", image_file=input_path.name,
            feature_id=f"F{tile + 1:02d}", feature_type=kind, orientation=axis,
            x0_px=x0, y0_px=y0, x1_px=x1, y1_px=y1,
            threshold=threshold, sample_step_px=.25,
            edge0_x_px=x0 + e0 / length * (x1 - x0),
            edge0_y_px=y0 + e0 / length * (y1 - y0),
            edge1_x_px=x0 + e1 / length * (x1 - x0),
            edge1_y_px=y0 + e1 / length * (y1 - y0),
            measured_cd_px=cd, noise_std_px=0.0,
        ))
    table_path = Path(table_output) if table_output else output_dir / "gauges.csv"
    with table_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    log = dict(
        description="Synthetic 50-term ideal model; noiseless cutline CD targets",
        seed=20261007, term_count=50, gauge_count=len(rows),
        input_image="input_image.png", input_decode="uint16 / 65535.0",
        image_shape=list(image.shape),
        input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        coordinates="zero-based pixel centers; x=column, y=row; y increases downwards",
        physical_pixel_size_nm=None, cd_unit="pixel", s_unit="pixel",
        threshold=threshold,
        threshold_rule="0.5 * (uniform dark response + uniform bright response); fixed for all candidate weights",
        sample_step_px=.25, interpolation="bilinear profile sampling; linear threshold crossing",
        edge_rule="exactly two crossings within cutline; otherwise raise ValueError",
        noise_std_px=0.0,
        make_Ib=dict(positive="maximum(image-b,0)", negative="maximum(b-image,0)"),
        model="sum(c_i * maximum(fftconvolve(gradient_transform(Ib)**n,G,mode='same'),0)**(1/n))",
        gradient="k iterations of dx**2+dy**2 using np.gradient with pixel spacing=1",
        gaussian="normalized isotropic Gaussian; sigma=s pixels; radius=ceil(p*s); zero-padded convolution",
        parameter_assumptions="b in [0.15,0.85], k in {0,1}, n in {1,2}, s in [0.5,4], p=3; "
                              "abs(c) in [0.02,0.08] for k=0, [0.5,2] for k=1; negative branch c<0",
        terms=[dict(term_id=i, **asdict(term)) for i, term in enumerate(terms)],
    )
    (output_dir / "ideal_parameters.log").write_text(json.dumps(log, indent=2, allow_nan=False) + "\n")
    render_preview(image, prediction, rows, threshold, output_dir)
    residuals = bind_cutline_gauges(rows)(prediction)
    assert np.max(np.abs(residuals)) < 1e-12
    print(f"Generated {len(terms)} terms, {len(rows)} cutlines; ideal CD RMS=0 px")
    print(f"CD range: {min(r['measured_cd_px'] for r in rows):.4f} .. "
          f"{max(r['measured_cd_px'] for r in rows):.4f} px; threshold={threshold:.8f}")
    return rows


def render_preview(image, prediction, rows, threshold, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    preview_rows = rows[::20]  # Show 200 of 4000 cutlines to keep the overlay legible.
    axes[0].set_title(f"Synthetic input / {len(rows)} cutlines ({len(preview_rows)} shown)")
    axes[1].imshow(prediction, cmap="viridis")
    axes[1].contour(prediction, levels=[threshold], colors="white", linewidths=.8)
    axes[1].set_title(f"Ideal resist response / contour T={threshold:.5f}")
    for row in preview_rows:
        axes[0].plot([row['x0_px'], row['x1_px']], [row['y0_px'], row['y1_px']],
                     color="#ff8c42" if row['orientation'] == 'horizontal' else '#32d9ef',
                     alpha=.55, linewidth=.45)
        axes[1].plot([row['edge0_x_px'], row['edge1_x_px']],
                     [row['edge0_y_px'], row['edge1_y_px']], '.', color='#ffb454', markersize=1.5)
    for tile in range(25):
        cy, cx = (tile // 5) * 64 + 32, (tile % 5) * 64 + 32
        axes[0].text(cx - 28, cy - 26, f"F{tile + 1:02d}", fontsize=7, color="white")
    for axis in axes:
        axis.set_xlabel("x (pixel)")
        axis.set_ylabel("y (pixel)")
    fig.savefig(output_dir / "cutlines_preview.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "cutline_case")
    parser.add_argument("--table-output", type=Path)
    args = parser.parse_args()
    generate(args.output_dir, args.table_output)
