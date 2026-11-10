"""Download a pinned public muDNN DEB for isolated, local comparison.

Read the official package index, verify its advertised package SHA-256, and
extract vendor libraries into a flat directory. Does not install any package
or run package-maintenance scripts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request


BASE = 'https://dl.mthreads.com/repo/repository/ubuntu2204/'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--package', default='libmudnn3-musa-5-2')
    p.add_argument('--version', default='3.4.0.0')
    p.add_argument('--library-prefix', default='libmudnn')
    p.add_argument('--resume-deb', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(BASE+'dists/jammy/main/binary-amd64/Packages.gz', timeout=30) as r:
        index = gzip.decompress(r.read()).decode()
    (a.output/'Packages').write_text(index)
    matches = []
    for block in index.split('\n\n'):
        fields = dict(line.split(': ', 1) for line in block.splitlines()
                      if ': ' in line and not line.startswith(' '))
        if fields.get('Package') == a.package and fields.get('Version') == a.version:
            matches.append(fields)
    if len(matches) != 1:
        raise ValueError('Expected one exact package/version; got '+str(len(matches)))
    fields = matches[0]
    (a.output/'package.json').write_text(json.dumps(fields, indent=2)+'\n')
    deb = a.output/Path(fields['Filename']).name
    if a.resume_deb:
        # Resume a stopped download into a new artifact; final full-file hash
        # verifies both the copied prefix and every independently fetched range.
        shutil.copyfile(a.resume_deb, deb)
        begin = deb.stat().st_size
        size = int(fields['Size'])
        if begin > size:
            raise ValueError('Resume file is larger than the package')
        with deb.open('r+b') as out:
            out.truncate(size)
            def fetch_range(bounds):
                start, end = bounds
                op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                request = urllib.request.Request(BASE+fields['Filename'],
                    headers={'Range':f'bytes={start}-{end}'})
                with op.open(request, timeout=45) as response:
                    expected = f'bytes {start}-{end}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Server did not honor the requested range')
                    position = start
                    while position <= end:
                        chunk = response.read(min(1024*1024, end-position+1))
                        if not chunk:
                            raise ValueError('Truncated HTTP range')
                        written = os.pwrite(out.fileno(), chunk, position)
                        if written != len(chunk):
                            raise IOError('Short local write')
                        position += written
                return end-start+1
            ranges = [(s, min(s+8*1024*1024, size)-1)
                      for s in range(begin, size, 8*1024*1024)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                total = begin
                for fetched in pool.map(fetch_range, ranges):
                    total += fetched
                    print('downloaded', total, '/', size, flush=True)
    else:
        total = 0
        last = time.monotonic()
        with opener.open(BASE+fields['Filename'], timeout=45) as r, deb.open('wb') as out:
            while True:
                chunk = r.read(1024*1024)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
                if time.monotonic()-last > 20:
                    print('downloaded', total, '/', fields['Size'], flush=True)
                    last = time.monotonic()
    h = hashlib.sha256()
    with deb.open('rb') as source:
        for chunk in iter(lambda:source.read(4*1024*1024), b''):
            h.update(chunk)
    if deb.stat().st_size != int(fields['Size']) or h.hexdigest() != fields['SHA256']:
        raise ValueError('Package size/SHA-256 mismatch')
    print('verified package', h.hexdigest(), flush=True)
    lib = a.output/'lib'
    lib.mkdir()
    inventory = []
    links = []
    with subprocess.Popen(['dpkg-deb', '--fsys-tarfile', str(deb)], stdout=subprocess.PIPE) as proc:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for entry in archive:
                name = Path(entry.name).name
                if not name.startswith(a.library_prefix) or '.so' not in name:
                    continue
                if entry.isfile():
                    with archive.extractfile(entry) as source, (lib/name).open('wb') as out:
                        shutil.copyfileobj(source, out, length=4*1024*1024)
                    inventory.append({'file':name, 'bytes':entry.size})
                    print('extracted', name, entry.size, flush=True)
                elif entry.issym():
                    links.append((name, Path(entry.linkname).name))
        if proc.wait() != 0:
            raise RuntimeError('dpkg-deb extraction failed')
    for name, target in links:
        (lib/name).symlink_to(target)
    (a.output/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    print('ready', lib.resolve(), flush=True)


if __name__ == '__main__':
    main()
