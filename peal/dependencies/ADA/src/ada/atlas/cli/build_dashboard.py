from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.visualization.dashboard import build_e0_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a self-contained HTML dashboard for ADA atlas E0 results.")
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--joined-csv", required=True, type=Path)
    parser.add_argument("--manifest-csv", default=None, type=Path)
    parser.add_argument("--cache-metadata-json", default=None, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_e0_dashboard(
        metrics_json=args.metrics_json,
        joined_csv=args.joined_csv,
        manifest_csv=args.manifest_csv,
        cache_metadata_json=args.cache_metadata_json,
        output_html=args.output_html,
    )
    print(json.dumps({"dashboard_html": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
