// Portable host-side topology construction for KERV verification trees.
// The published Python facade references this operator but omitted its source.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include <tuple>
#include <vector>

using at::Tensor;
using TreeResult = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;

TreeResult build_verification_subtree(const Tensor& parent_indices, int64_t max_paths) {
  TORCH_CHECK(parent_indices.device().is_cpu() && parent_indices.scalar_type() == at::kLong,
              "parent indices must be an int64 CPU tensor");
  TORCH_CHECK(parent_indices.dim() == 1 && parent_indices.numel() > 0,
              "parent indices must be a nonempty vector");
  TORCH_CHECK(max_paths >= 0, "max_paths must be nonnegative");
  const auto parents = parent_indices.contiguous();
  const auto* parent = parents.data_ptr<int64_t>();
  const auto nodes = parents.numel();
  TORCH_CHECK(parent[0] == 0, "root parent must be zero");
  std::vector<bool> has_child(nodes, false);
  for (int64_t i = 1; i < nodes; ++i) {
    TORCH_CHECK(parent[i] >= 0 && parent[i] < nodes && parent[i] != i,
                "invalid parent index");
    has_child[parent[i]] = true;
  }
  std::vector<int64_t> leaves;
  // Validate every ancestor chain, including any disconnected cyclic branch.
  for (int64_t i = 1; i < nodes; ++i) {
    int64_t cursor = i, traversed = 0;
    while (cursor != 0 && traversed++ < nodes) cursor = parent[cursor];
    TORCH_CHECK(cursor == 0, "parent indices contain a cycle");
    if (!has_child[i]) leaves.push_back(i);
  }
  if (leaves.empty()) leaves.push_back(0);
  const auto full_paths = static_cast<int64_t>(leaves.size());
  if (max_paths > 0 && max_paths < full_paths) leaves.resize(max_paths);
  std::vector<std::vector<int64_t>> paths;
  std::vector<bool> selected(nodes, false);
  int64_t width = 0;
  for (const auto leaf : leaves) {
    std::vector<int64_t> path;
    for (int64_t cursor = leaf;; cursor = parent[cursor]) {
      path.push_back(cursor);
      selected[cursor] = true;
      if (cursor == 0) break;
    }
    std::reverse(path.begin(), path.end());
    width = std::max(width, static_cast<int64_t>(path.size()));
    paths.push_back(std::move(path));
  }
  std::vector<int64_t> kept_nodes, mapping(nodes, -1);
  for (int64_t i = 0; i < nodes; ++i) if (selected[i]) {
    mapping[i] = static_cast<int64_t>(kept_nodes.size());
    kept_nodes.push_back(i);
  }
  const auto count = static_cast<int64_t>(kept_nodes.size());
  const auto long_options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  auto kept = at::tensor(kept_nodes, long_options);
  auto retrieve = at::full({static_cast<int64_t>(paths.size()), width}, -1, long_options);
  auto mask = at::zeros({1, 1, count, count}, at::TensorOptions().dtype(at::kFloat));
  auto positions = at::zeros({1, count}, long_options);
  auto* retrieve_data = retrieve.data_ptr<int64_t>();
  auto* mask_data = mask.data_ptr<float>();
  auto* position_data = positions.data_ptr<int64_t>();
  for (int64_t i = 0; i < count; ++i) {
    int64_t depth = 0;
    for (int64_t cursor = kept_nodes[i];; cursor = parent[cursor]) {
      mask_data[i * count + mapping[cursor]] = 1.f;
      if (cursor == 0) break;
      ++depth;
    }
    position_data[i] = depth;
  }
  for (size_t row = 0; row < paths.size(); ++row)
    for (size_t col = 0; col < paths[row].size(); ++col)
      retrieve_data[row * width + col] = mapping[paths[row][col]];
  auto stats = at::tensor({nodes, count, full_paths, static_cast<int64_t>(paths.size())}, long_options);
  return {kept, retrieve, mask, positions, stats};
}

Tensor pack_verification_tokens(const Tensor& tokens, const Tensor& kept) {
  TORCH_CHECK(tokens.dim() == 2 && kept.dim() == 1, "expected token matrix and index vector");
  return tokens.index_select(1, kept.to(tokens.device(), at::kLong));
}

Tensor pack_verification_tokens_padded(const Tensor& tokens, const Tensor& kept, int64_t target) {
  auto packed = pack_verification_tokens(tokens, kept);
  TORCH_CHECK(target >= packed.size(1), "padding target is smaller than selected nodes");
  auto output = tokens.narrow(1, 0, 1).expand({tokens.size(0), target}).clone();
  output.narrow(1, 0, packed.size(1)).copy_(packed);
  return output;
}

TORCH_LIBRARY_FRAGMENT(flagos_kerv, m) {
  m.def("build_verification_subtree(Tensor parent_indices, int max_paths) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("pack_verification_tokens(Tensor draft_tokens, Tensor kept_indices) -> Tensor");
  m.def("pack_verification_tokens_padded(Tensor draft_tokens, Tensor kept_indices, int target_nodes) -> Tensor");
}
TORCH_LIBRARY_IMPL(flagos_kerv, CPU, m) {
  m.impl("build_verification_subtree", build_verification_subtree);
}
TORCH_LIBRARY_IMPL(flagos_kerv, CompositeExplicitAutograd, m) {
  m.impl("pack_verification_tokens", pack_verification_tokens);
  m.impl("pack_verification_tokens_padded", pack_verification_tokens_padded);
}
