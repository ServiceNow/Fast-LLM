#include <torch/extension.h>

at::Tensor embedding_dense_backward(const at::Tensor &grad_output, const at::Tensor &indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq){
    return at::native::embedding_dense_backward_cuda(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("embedding_dense_backward", embedding_dense_backward, "embedding_dense_backward");
}
