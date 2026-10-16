"""Release candidates contain source and explanations, never experiment assets."""

from pathlib import Path
import ast
import re

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ONLY = {'PUBLIC_RELEASE_REPORT.md','PROFILE_LINK_SNIPPET.md'}


def public_files():
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in ('.local-review','.venv','venv','.git'):
            continue
        if any(part in ('__pycache__','.pytest_cache') for part in relative.parts):
            continue
        if path.is_file() and relative.as_posix() not in LOCAL_ONLY:
            yield path


def test_no_data_weights_or_old_results():
    blocked = {'.csv','.mat','.npy','.npz','.h5','.hdf5','.parquet','.pt','.pth','.ckpt','.pkl','.pickle','.joblib','.safetensors','.log'}
    assert all(path.suffix.lower() not in blocked for path in public_files())
    assert all(not (ROOT/name).exists() for name in ('Data','DataExperiment','outputs','checkpoints','.git'))


def test_local_review_files_are_ignored():
    lines = (ROOT/'.gitignore').read_text().splitlines()
    assert all(name in lines for name in LOCAL_ONLY)
    assert '.local-review/' in lines


def test_local_readme_links_exist():
    text = (ROOT/'README.md').read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if '://' not in target and not target.startswith('#'):
            assert (ROOT/target.split('#')[0]).exists()


def test_no_private_paths_or_network_clients():
    for path in public_files():
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/]',text)
        assert not re.search(r'/(?:Users|home|mnt)/[A-Za-z]',text)
        if path.suffix == '.py':
            tree = ast.parse(text)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module.split('.')[0])
            assert not set(imports) & {'requests','urllib','httpx','socket'}


def test_graphical_reference_names_and_status():
    text = (ROOT/'README.md').read_text(encoding='utf-8')
    expected = ('08-jms-slag-removal-motivation.svg',
                '08-jms-slag-removal-framework.svg',
                '08-jms-slag-removal-state-transition.svg',
                '08-jms-slag-removal-ppo.svg')
    embedded = re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
    assert tuple(url.rsplit('/',1)[-1] for url in embedded) == expected
    assert all(url.startswith('https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/') for url in embedded)
    assert '**GitHub profile entry:** [8]' in text
    assert '**Status:** Submitted (unpublished)' in text
    assert 'LINK_NOT_LIVE_VERIFIED' in text and 'TO_VERIFY' in text
    assert '<details>' not in text
