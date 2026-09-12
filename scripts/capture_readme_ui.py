"""Render a clean README hero from the synthetic butterfly pattern."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from butterfly_saxs.pipeline import synthetic_butterfly


def _ellipse_xy(a: float, b: float, angle_deg: float, n: int = 361) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * math.pi, n)
    angle = math.radians(angle_deg)
    x = a * np.cos(t)
    y = b * np.sin(t)
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


image, qmap = synthetic_butterfly(
    (256, 256),
    q0=28.0,
    width=2.2,
    ellipticity=2.0,
    angle_deg=28.0,
    return_qmap=True,
)
qx = np.asarray(qmap["qx"], dtype=float)
qy = np.asarray(qmap["qy"], dtype=float)
extent = (float(qx.min()), float(qx.max()), float(qy.min()), float(qy.max()))
# synthetic_butterfly: q(δ=0)=q0*ellipticity, q(δ=90°)=q0
plus_x, plus_y = _ellipse_xy(56.0, 28.0, 28.0)
minus_x, minus_y = _ellipse_xy(56.0, 28.0, -28.0)

fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=160)
ax.imshow(
    np.log10(np.asarray(image, dtype=float) + 1e-4),
    origin="lower",
    extent=extent,
    cmap="magma",
    aspect="equal",
)
ax.plot(plus_x, plus_y, color="white", lw=1.2, label="measured ellipse (+)")
ax.plot(minus_x, minus_y, color="white", lw=1.2, ls="--", label="mirror ellipse (−)")
ax.set_xlabel(r"$q_x$ (pixel-q)")
ax.set_ylabel(r"$q_y$ (pixel-q)")
ax.set_title("Synthetic butterfly  ·  origin-centred double ellipse")
ax.legend(loc="upper right", frameon=True, fancybox=False, fontsize=8)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
fig.tight_layout()
target = Path(__file__).resolve().parents[1] / "docs" / "assets" / "refinement-ui.png"
target.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(target, dpi=160)
plt.close(fig)
if not target.exists() or target.stat().st_size < 10_000:
    raise SystemExit(f"failed to write {target}")
print(target, target.stat().st_size)
