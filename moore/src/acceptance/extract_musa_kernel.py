"""Extract a MUSA offload image and one exact symbol for local inspection.

Only reads its input. Extracted vendor code stays in the requested local output
directory and is not part of a source patch or upstream submission.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct


def sections(data):
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
    if header[0][:6] != b'\x7fELF\x02\x01':
        raise ValueError('Expected little-endian ELF64')
    result = [struct.unpack_from('<IIQQQQIIQQ', data, header[6]+i*header[11])
              for i in range(header[12])]
    names_section = result[header[13]]
    names = data[names_section[4]:names_section[4]+names_section[5]]
    return [(names[s[0]:names.find(b'\0', s[0])].decode(), s) for s in result]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--target', default='musa-mtgpu-mt-musa--mp_31')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.library.open('rb') as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        _, fat = next(x for x in sections(data) if x[0] == '.mt_fatbin')
        base = fat[4]
        magic = b'__CLANG_OFFLOAD_BUNDLE__'
        if data[base:base+len(magic)] != magic: raise ValueError('Unknown bundle format')
        count, = struct.unpack_from('<Q', data, base+len(magic))
        pos = base+len(magic)+8
        image = None
        for _ in range(count):
            offset, size, length = struct.unpack_from('<QQQ', data, pos)
            pos += 24
            target = data[pos:pos+length].decode()
            pos += length
            if target == args.target: image = data[base+offset:base+offset+size]
        if image is None: raise ValueError('Requested target absent')
        original_sha = hashlib.sha256(data).hexdigest()
    image_sections = sections(image)
    found = None
    for _, section in image_sections:
        if section[1] != 11: continue  # SHT_DYNSYM
        strings = image_sections[section[6]][1]
        names = image[strings[4]:strings[4]+strings[5]]
        for pos in range(section[4], section[4]+section[5], section[9]):
            name, info, other, index, address, size = struct.unpack_from('<IBBHQQ', image, pos)
            if names[name:names.find(b'\0', name)].decode() != args.symbol: continue
            owner = image_sections[index][1]
            start = owner[4]+address-owner[3]
            found = (address, size, image[start:start+size])
            break
    if found is None: raise ValueError('Requested dynamic symbol absent')
    address, size, kernel = found
    (args.output/'device.mubin').write_bytes(image)
    (args.output/'kernel.bin').write_bytes(kernel)
    report = {'library':str(args.library), 'library_sha256':original_sha,
        'target':args.target, 'symbol':args.symbol, 'address':hex(address),
        'size_bytes':size, 'kernel_sha256':hashlib.sha256(kernel).hexdigest()}
    (args.output/'metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
