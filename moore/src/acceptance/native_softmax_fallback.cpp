// Exercise the existing musified ATen kernel without the muDNN dispatcher.
#include <torch/extension.h>
#include <ATen/ops/_softmax_native.h>
#include <c10/core/DeviceGuard.h>

namespace {
struct NativeSoftmax : at::native::structured_softmax_cuda_out {
  const at::Tensor& output;
  explicit NativeSoftmax(const at::Tensor& value) : output(value) {}
  const at::Tensor& maybe_get_output(int64_t index) override {
    TORCH_CHECK(index==0);return output;
  }
};

at::Tensor forward(const at::Tensor& input,int64_t dim) {
  c10::DeviceGuard guard(input.device());
  auto x=input.contiguous();auto out=at::empty_like(x);
  NativeSoftmax kernel(out);kernel.impl(x,dim,false,out);
  return out;
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
