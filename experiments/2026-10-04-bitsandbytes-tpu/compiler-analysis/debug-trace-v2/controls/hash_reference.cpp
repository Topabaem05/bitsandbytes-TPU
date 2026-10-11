// Independent stdlib-only reference for pinned torch lazy hash.cpp.
// Requires little-endian, 16-byte native unsigned __int128.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using U = unsigned __int128;
std::string text(U h) {
  std::ostringstream s;
  s << std::hex << static_cast<uint64_t>(h >> 64)
    << std::setfill('0') << std::setw(16) << static_cast<uint64_t>(h);
  return s.str();
}
U digest(const std::vector<uint8_t>& b) {
  const U m = static_cast<uint64_t>(0xc6a4a7935bd1e995ULL);
  U h = static_cast<uint64_t>(0xc2b2ae3d27d4eb4fULL) ^ (b.size() * m);
  for (size_t pos = 0; pos < b.size(); pos += sizeof(U)) {
    U k = 0;
    std::memcpy(&k, b.data()+pos, std::min(sizeof(U), b.size()-pos));
    k *= m; k ^= k >> 47; k *= m;
    h ^= k; h *= m;
  }
  h ^= h >> 47; h *= m; h ^= h >> 47;
  return h;
}
int main(int argc, char** argv) {
  uint16_t endian = 1;
  if (sizeof(U) != 16 || *reinterpret_cast<uint8_t*>(&endian) != 1) return 3;
  if (argc == 3 && std::string(argv[1]) == "--format") {
    U h = 0;
    for (char c : std::string(argv[2])) {
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return 2;
      h = (h << 4) | (c <= '9' ? c - '0' : c - 'a' + 10);
    }
    std::cout << text(h) << '\n'; return 0;
  }
  if (argc != 2) return 2;
  std::string hex = argv[1];
  if (hex.size() % 2) return 2;
  std::vector<uint8_t> b;
  for (size_t i = 0; i < hex.size(); i += 2) {
    unsigned n; std::istringstream s(hex.substr(i,2)); s >> std::hex >> n;
    if (!s || n > 255) return 2;
    b.push_back(static_cast<uint8_t>(n));
  }
  std::cout << text(digest(b)) << '\n';
}
