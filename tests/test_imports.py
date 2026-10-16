"""Import the standalone package without the private project."""

import importlib
from phgt_ppo.config import load_demo_config


def test_core_imports():
    for name in ('models','losses','physics','features','preprocessing','random_utils','reward','ppo'):
        assert importlib.import_module('phgt_ppo.'+name)


def test_safe_config():
    cfg = load_demo_config()
    assert set(cfg) == {'forecast','ppo'}
    assert cfg['forecast']['dimension'] == 128
    assert len(cfg['ppo']['hidden']) == 3
