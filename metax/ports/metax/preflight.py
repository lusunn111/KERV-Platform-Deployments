"""Run on the single visible MetaX device, before model deployment."""
import json
import time
import torch

assert getattr(torch.version, 'maca', None), 'Vendor PyTorch required'
assert torch.cuda.device_count() == 1
properties = torch.cuda.get_device_properties(0)
assert properties.name == 'MetaX C550'
a = torch.arange(256 * 256, device='cuda', dtype=torch.float32).reshape(256,256).remainder(17).to(torch.bfloat16)
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
b = a @ a.T
end.record()
torch.cuda.synchronize()
torch.testing.assert_close(b.cpu(), a.cpu() @ a.cpu().T, rtol=0, atol=0)
print(json.dumps(dict(torch=torch.__version__, torch_path=torch.__file__, maca=torch.version.maca,
    device=str(properties), logical='cuda:0', bf16_matmul_exact=True,
    device_event_ms=start.elapsed_time(end),
    allocated_peak_bytes=torch.cuda.max_memory_allocated(),
    reserved_peak_bytes=torch.cuda.max_memory_reserved()), indent=2))
