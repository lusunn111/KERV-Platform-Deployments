"""Read-only storage budget for the proposed calibrated mixed quantizer.

Uses safetensors headers, not a model load. This is a design estimate, not a
quantization run or a GPU peak measurement. No checkpoint data is modified.
"""
import argparse
import json
import math
from pathlib import Path
import re
import struct


def header(path):
    with path.open('rb') as stream:
        count = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < count < 100_000_000:
            raise ValueError('Invalid safetensors header length')
        return json.loads(stream.read(count))


def budget(checkpoint, group_size, protected_fraction):
    if group_size not in (32, 64, 128) or not 0 <= protected_fraction <= 1:
        raise ValueError('Unsupported group size or protected fraction')
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    files = {name: header(checkpoint / name) for name in set(index['weight_map'].values())}
    projections = []
    for name, shard in sorted(index['weight_map'].items()):
        match = re.fullmatch(r'language_model.model.layers.(\d+).mlp.(gate_proj|up_proj|down_proj).weight', name)
        if not match:
            continue
        item = files[shard][name]
        if item['dtype'] != 'BF16' or len(item['shape']) != 2:
            raise ValueError('Expected BF16 projection ' + name)
        layer = int(match[1]); out_features, in_features = item['shape']
        n = out_features * in_features
        groups = math.ceil(in_features / group_size)
        # Column groups shared across all output rows, with a strict ceil rule.
        k = math.ceil(protected_fraction * groups)
        protected_columns = min(in_features, k * group_size)
        # Full INT8 backing plus BF16 replacements conservatively double stores
        # protected groups. Keep both scales and indices in the storage account.
        retained = layer < 2 or layer >= 28
        parts = ({'native_bf16': 2*n} if retained else {
            'int8_weights': n,
            'fp32_group_scales': 4*out_features*groups,
            'fp32_channel_inverse_scales': 4*in_features,
            'protected_group_indices_int32': 4*k,
            'protected_original_bf16': 2*out_features*protected_columns,
        })
        projections.append({'name': name, 'shape': item['shape'], 'mode': 'BF16' if retained else 'W8_BF16_GROUPS',
                            'group_size': group_size, 'protected_groups': 0 if retained else k,
                            'protected_columns': 0 if retained else protected_columns,
                            'before_bytes': 2*n, 'after_bytes': sum(parts.values()), 'components': parts})
    if len(projections) != 96:
        raise ValueError('Expected 96 verifier MLP projections')
    components = {}
    for row in projections:
        for key, value in row['components'].items():
            components[key] = components.get(key, 0) + value
    before = sum(x['before_bytes'] for x in projections)
    after = sum(x['after_bytes'] for x in projections)
    return {'status': 'DESIGN_ESTIMATE_ONLY', 'group_size': group_size, 'requested_protected_group_fraction': protected_fraction,
            'bf16_mlp_projections': sum(x['mode'] == 'BF16' for x in projections),
            'mixed_mlp_projections': sum(x['mode'] != 'BF16' for x in projections),
            'mlp_bytes_before': before, 'mlp_bytes_after': after, 'estimated_static_saving_bytes': before-after,
            'components_bytes': components,
            'largest_single_decoded_projection_bytes': max(x['before_bytes'] for x in projections if x['mode'] != 'BF16'),
            'excludes': ['unchanged attention/vision/drafter/embedding/head/norm/KV',
                         'allocator rounding and workspaces', 'activation and temporary buffers',
                         'checkpoint container/JSON overhead', 'no activation statistics collected by this tool'],
            'projections': projections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output file')
    results = [budget(args.checkpoint, g, p) for g in (32, 64) for p in (0, .02, .05)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    for result in results:
        print('group', result['group_size'], 'protected', result['requested_protected_group_fraction'],
              'static_saving_GB', round(result['estimated_static_saving_bytes']/1e9, 6))


if __name__ == '__main__':
    main()
