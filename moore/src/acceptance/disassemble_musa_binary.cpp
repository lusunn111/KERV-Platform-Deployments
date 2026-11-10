// Local, read-only adapter for the DisassembleBinary entry point shipped in
// libmusaasm.so. The ABI was inferred from the installed driver's call site;
// it is not a supported public API. Run as a separate CPU process.
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: disassemble_musa_binary LIB INPUT ARCH_ENUM\n";
    return 2;
  }
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) { std::cerr << dlerror() << '\n'; return 2; }
  using Disassemble = int (*)(std::ostream&, const uint32_t*, uint64_t*,
                              uint64_t, unsigned, bool, bool, uint64_t);
  auto disassemble = reinterpret_cast<Disassemble>(dlsym(library, "DisassembleBinary"));
  if (!disassemble) { std::cerr << dlerror() << '\n'; return 2; }
  std::ifstream file(argv[2], std::ios::binary);
  if (!file) return 2;
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
  if (bytes.empty() || bytes.size() % 4) return 2;
  std::vector<uint32_t> words(bytes.size() / 4 + 16, 0);
  std::copy(bytes.begin(), bytes.end(), reinterpret_cast<char*>(words.data()));
  uint64_t position = 0;
  int result = disassemble(std::cout, words.data(), &position, bytes.size()/4,
                           std::strtoul(argv[3], nullptr, 0), false, false, 0);
  std::cerr << "status=" << result << " decoded_words=" << position
            << " total_words=" << bytes.size()/4 << '\n';
  return result ? 1 : 0;
}
