// Small SHA-256 implementation. The algorithm follows FIPS 180-4.
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
struct Sha256 {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  unsigned char b[64];
  size_t n = 0;
  uint64_t bytes = 0;
  static uint32_t r(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
  }
  void block() {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(b[4 * i]) << 24) | (uint32_t(b[4 * i + 1]) << 16) |
             (uint32_t(b[4 * i + 2]) << 8) | b[4 * i + 3];
    for (int i = 16; i < 64; ++i)
      w[i] = w[i - 16] +
             (r(w[i - 15], 7) ^ r(w[i - 15], 18) ^ (w[i - 15] >> 3)) +
             w[i - 7] + (r(w[i - 2], 17) ^ r(w[i - 2], 19) ^ (w[i - 2] >> 10));
    uint32_t a = h[0], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6],
             q = h[7], z = h[1];
    for (int i = 0; i < 64; ++i) {
      uint32_t t = q + (r(e, 6) ^ r(e, 11) ^ r(e, 25)) + ((e & f) ^ (~e & g)) +
                   k[i] + w[i];
      uint32_t u =
          (r(a, 2) ^ r(a, 13) ^ r(a, 22)) + ((a & z) ^ (a & c) ^ (z & c));
      q = g;
      g = f;
      f = e;
      e = d + t;
      d = c;
      c = z;
      z = a;
      a = t + u;
    }
    h[0] += a;
    h[1] += z;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += q;
  }
  void add(const void *p, size_t s) {
    auto x = static_cast<const unsigned char *>(p);
    bytes += s;
    while (s) {
      size_t m = 64 - n < s ? 64 - n : s;
      memcpy(b + n, x, m);
      n += m;
      x += m;
      s -= m;
      if (n == 64) {
        block();
        n = 0;
      }
    }
  }
  std::string end() {
    uint64_t bits = bytes * 8;
    unsigned char q = 0x80;
    add(&q, 1);
    q = 0;
    while (n != 56)
      add(&q, 1);
    unsigned char len[8];
    for (int i = 0; i < 8; ++i)
      len[7 - i] = bits >> (8 * i);
    add(len, 8);
    char s[65];
    for (int i = 0; i < 8; ++i)
      snprintf(s + 8 * i, 9, "%08x", h[i]);
    return s;
  }
};
inline std::string digest(const void *p, size_t n) {
  Sha256 s;
  s.add(p, n);
  return s.end();
}
