"""Rehash the frozen checkpoint dependencies for the precision follow-up."""
import argparse
from pathlib import Path
from acceptance.common import load,dump,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();root=Path('/root/KERV')
    source=root/'outputs/calibrated_quantization/20261026T045030Z_v2/inventory/weight_manifest.json'
    expected=load(source);actual={}
    for rel,entry in expected.items():
        path=root/rel;record={'bytes':path.stat().st_size,'sha256':sha256(path)}
        assert record==entry,rel
        actual[rel]=record;print('VERIFIED',rel,flush=True)
    dump(a.campaign/'inventory/weight_manifest.json',actual)
    dump(a.campaign/'reports/dependencies_checked.json',{'status':'PASS','files':len(actual),'source_manifest':str(source),'rehash_performed':True})


if __name__=='__main__':main()
