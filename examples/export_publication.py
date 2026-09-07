"""Export an existing native result as a reproducible publication artboard.

Run with the GUI extra installed, on the main Python thread. The destination
must be new. No fitting or editing of the source result occurs here.
"""
from __future__ import annotations

import argparse

from butterfly_saxs import build_lamellar_scene, load_lamellar_sources
from butterfly_saxs.publication import PublicationFigureSpec, PublicationStyle, export_publication_figure
from butterfly_saxs.ui.lamellar_state import source_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Native single result JSON/NPZ or batch manifest")
    parser.add_argument("destination", help="New output directory")
    parser.add_argument("--index", type=int, default=0, help="Zero-based index in the native sequence")
    parser.add_argument("--width", type=float, default=183., help="Artboard width in mm")
    parser.add_argument("--height", type=float)
    parser.add_argument("--template", choices=("structure", "evidence"), default="structure")
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--transparent", action="store_true")
    args = parser.parse_args()
    sources = load_lamellar_sources(args.source)
    if not 0 <= args.index < len(sources):
        parser.error("index is outside the actual source sequence")
    source = sources[args.index]
    scene = build_lamellar_scene(source, {"mode": args.mode})
    if not scene.metadata["available"]:
        parser.error(scene.message)
    narrow = args.width < 120
    height = args.height or ((90. if narrow else 126.) if args.template == "evidence" else (70. if narrow else 95.))
    spec = PublicationFigureSpec(template=args.template, width_mm=args.width, height_mm=height,
                                 dpi=args.dpi, background="transparent" if args.transparent else "white")
    observed, qx, qy = source_images(source)
    outputs = export_publication_figure(scene, args.destination, spec=spec, style=PublicationStyle(),
        observed=observed, qx=qx, qy=qy, progress=lambda current, total: print(f"{current}/{total}", flush=True))
    print(outputs["pdf"])


if __name__ == "__main__":
    main()
