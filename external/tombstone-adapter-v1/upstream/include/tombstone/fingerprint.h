/* SPDX-License-Identifier: Apache-2.0 */
#ifndef TOMBSTONE_FINGERPRINT_H
#define TOMBSTONE_FINGERPRINT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TB_FINGERPRINT_ABI_VERSION 1u
#define TB_SHA256_DIGEST_SIZE 32u
#define TB_SHA256_HEX_SIZE 65u

/* Stable v1 status values; never use errno or OpenSSL-specific error codes. */
typedef enum tb_status {
    TB_OK = 0,
    TB_ERR_ARGUMENT = 1,
    TB_ERR_LENGTH = 2,
    TB_ERR_MEMORY = 3,
    TB_ERR_CRYPTO = 4,
    TB_ERR_STATE = 5,
    TB_ERR_MISMATCH = 6, /* supplied evidence or digest mismatch */
    TB_ERR_FORMAT = 7 /* malformed/unsupported receipt v1 */
} tb_status;

typedef struct tb_sha256_ctx tb_sha256_ctx;

/* Version query for future FFI compatibility checks. */
unsigned int tb_fingerprint_abi_version(void);

/*
 * Hash exactly length bytes of data; data == NULL is permitted iff length == 0.
 * out must point to at least TB_SHA256_DIGEST_SIZE writable bytes, and must not
 * overlap any input bytes. On failure, out remains untouched.
 * Inputs are neither retained nor modified; no filesystem I/O occurs.
 */
tb_status tb_sha256_bytes(const void *data, size_t length,
                          unsigned char out[TB_SHA256_DIGEST_SIZE]);

/*
 * Streaming hashing: init assigns *out_ctx only on success, otherwise NULL.
 * Updates accept NULL with zero length. The sum of all lengths must fit the
 * standard SHA-256 64-bit bit counter (at most UINT64_MAX / 8 bytes).
 * Updates after final, or a second final, return TB_ERR_STATE.
 * Any cryptographic backend error puts a context in a failed state.
 * Final writes exactly 32 bytes on success, otherwise leaves out untouched.
 * The caller owns the context and must call tb_sha256_free, including on error.
 * Do not use the same context concurrently from multiple threads.
 */
tb_status tb_sha256_new(tb_sha256_ctx **out_ctx);
tb_status tb_sha256_update(tb_sha256_ctx *ctx, const void *data, size_t length);
tb_status tb_sha256_final(tb_sha256_ctx *ctx,
                          unsigned char out[TB_SHA256_DIGEST_SIZE]);
void tb_sha256_free(tb_sha256_ctx *ctx);

/* Lowercase ASCII hex plus a terminal NUL. No allocation or I/O. */
tb_status tb_sha256_hex(const unsigned char digest[TB_SHA256_DIGEST_SIZE],
                        char out[TB_SHA256_HEX_SIZE]);

#ifdef __cplusplus
}
#endif
#endif /* TOMBSTONE_FINGERPRINT_H */
