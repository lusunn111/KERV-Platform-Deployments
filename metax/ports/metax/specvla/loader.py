"""Isolated loading of byte-preserved, pinned SpecVLA code on vendor PyTorch.

Only namespace routing is shared by both variants. boundary_fixed changes one
predicate in utils.py at compile time. No KERV speculation/drafter implementation
is imported. Original source files and entry remain available for audit.
"""
import importlib
import importlib.abc
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / 'upstream_original'
PREDICATE = 'if (input_ids.shape[1]-input_len-1+accept_length)>6:'
FIXED = 'if (input_ids.shape[1]-input_len-1+accept_length)>=6:'

def verify_original():
    manifest = json.loads((HERE / 'provenance.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Upstream source changed: {name}')
    return manifest

def adapted_source(relative, variant):
    source = (ORIGINAL / relative).read_text()
    prefix = f'_kerv_specvla_{variant}'
    # Upstream absolute imports point to the independently loaded original drafter.
    source = source.replace('from openvla.', f'from {prefix}.openvla.')
    source = source.replace('from local_transformers', f'from {prefix}.openvla.local_transformers')
    if relative == 'openvla/prismatic/extern/hf/modeling_speculation.py':
        # Checkpoint storages were serialized on CUDA; route loading to CPU in
        # both variants before the one explicit model.to(cuda) deployment step.
        assert source.count('torch.load(load_model_path)') == 1
        source = source.replace('torch.load(load_model_path)',
                                "torch.load(load_model_path, map_location='cpu')")
    if relative == 'openvla/prismatic/extern/hf/utils.py':
        assert source.count(PREDICATE) == 1
        if variant == 'boundary_fixed':
            source = source.replace(PREDICATE, FIXED)
    return source

class SourceLoader(importlib.abc.Loader):
    def __init__(self, relative, variant):
        self.relative, self.variant = relative, variant
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        path = ORIGINAL / self.relative
        module.__file__ = str(path)
        exec(compile(adapted_source(self.relative, self.variant), str(path), 'exec'), module.__dict__)

class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, prefix, variant):
        self.prefix, self.variant = prefix, variant
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.prefix + '.'):
            return None
        relative = fullname[len(self.prefix)+1:].replace('.', '/')
        if (ORIGINAL / relative / '__init__.py').is_file():
            return importlib.util.spec_from_loader(fullname, SourceLoader(relative+'/__init__.py', self.variant), is_package=True)
        if (ORIGINAL / relative).is_dir():
            return importlib.util.spec_from_loader(fullname, loader=None, is_package=True)
        if (ORIGINAL / (relative + '.py')).is_file():
            return importlib.util.spec_from_loader(fullname, SourceLoader(relative+'.py', self.variant))
        return None

def load(variant):
    if variant not in {'upstream_original', 'boundary_fixed'}:
        raise ValueError(variant)
    verify_original()
    prefix = f'_kerv_specvla_{variant}'
    if prefix not in sys.modules:
        package = types.ModuleType(prefix); package.__path__ = []
        sys.modules[prefix] = package
        sys.meta_path.insert(0, Finder(prefix, variant))
    return tuple(importlib.import_module(prefix + '.openvla.prismatic.extern.hf.' + name)
                 for name in ['modeling_prismatic', 'modeling_speculation', 'utils'])
