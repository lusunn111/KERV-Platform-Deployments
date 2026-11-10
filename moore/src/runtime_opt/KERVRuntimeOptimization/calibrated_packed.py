"""Wide calibration workspace and checked export to packed W10/W12 storage."""
import copy
import torch
from .calibrated_int8 import CalibratedInt8Linear
from .packed_intn import PackedIntLinear,pack_codes
from .packed_pow2 import PackedPow2Linear


class CalibrationIntNLinear(CalibratedInt8Linear):
    def __init__(self,k,n,group_size,protected_count,bits):
        super().__init__(k,n,group_size,protected_count)
        self.qweight=torch.zeros(n,k,dtype=torch.int16);self.precision_bits=bits

    def payload(self):raise RuntimeError('Calibration workspace: export with pack_calibrated()')


def pack_calibrated(module):
    if not isinstance(module,CalibrationIntNLinear):raise ValueError('Expected wide calibration workspace')
    if module.qweight.device.type!='cpu':raise ValueError('Export the completed workspace on CPU')
    if not bool((module.inverse_channel_scales==1).all()):raise ValueError('Packed export requires identity inverse scales')
    power2=module.calibration['scale_kind']=='pow2';cls=PackedPow2Linear if power2 else PackedIntLinear
    q=cls(module.in_features,module.out_features,module.precision_bits,module.group_size,module.protected_groups.numel())
    q.packed.copy_(torch.from_numpy(pack_codes(module.qweight.numpy(),module.precision_bits)))
    if power2:
        exponents=module.scales.log2().round().to(torch.int32)
        if bool(((exponents < -126)|(exponents > 127)).any()):raise ValueError('Scale exponent out of range')
        reconstructed=torch.ldexp(torch.ones_like(module.scales),exponents)
        if not torch.equal(reconstructed,module.scales):raise ValueError('Non-power-of-two calibration scale')
        q.scales.copy_(exponents.to(torch.int8))
    else:q.scales.copy_(module.scales)
    q.protected_groups.copy_(module.protected_groups);q.protected_values.copy_(module.protected_values);q.prepare()
    if module.bias is not None:q.bias=module.bias.clone()
    q.calibration=copy.deepcopy(module.calibration)
    if not torch.equal(q.dequantize().view(torch.int16),module.dequantize().view(torch.int16)):
        raise RuntimeError('Packed export changed calibrated BF16 weight bits')
    return q.eval()
