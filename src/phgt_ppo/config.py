"""Load the public, non-industrial demonstration configuration."""

from pathlib import Path
import yaml


def load_demo_config(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[2] / 'configs/demo.yaml'
    with path.open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)
