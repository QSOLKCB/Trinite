/* SPDX-License-Identifier: Apache-2.0 */
#include "tombstone/fingerprint.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>

#if CHAR_BIT != 8
#error "TOMBSTONE requires eight-bit bytes"
#endif

/* SHA-256 encodes its message length in a 64-bit BIT count. */
#define TB_MAX_SHA256_BYTES (UINT64_MAX / UINT64_C(8))

struct tb_sha256_ctx {
    EVP_MD_CTX *backend;
    uint64_t total;
    int finished;
    int failed;
};

unsigned int tb_fingerprint_abi_version(void)
{
    return TB_FINGERPRINT_ABI_VERSION;
}

tb_status tb_sha256_new(tb_sha256_ctx **out_ctx)
{
    tb_sha256_ctx *ctx;
    if (out_ctx == NULL) {
        return TB_ERR_ARGUMENT;
    }
    *out_ctx = NULL;
    ctx = (tb_sha256_ctx *)calloc(1u, sizeof(*ctx));
    if (ctx == NULL) {
        return TB_ERR_MEMORY;
    }
    ctx->backend = EVP_MD_CTX_new();
    if (ctx->backend == NULL) {
        free(ctx);
        return TB_ERR_MEMORY;
    }
    if (EVP_DigestInit_ex(ctx->backend, EVP_sha256(), NULL) != 1) {
        EVP_MD_CTX_free(ctx->backend);
        free(ctx);
        return TB_ERR_CRYPTO;
    }
    *out_ctx = ctx;
    return TB_OK;
}

tb_status tb_sha256_update(tb_sha256_ctx *ctx, const void *data, size_t length)
{
    if (ctx == NULL) {
        return TB_ERR_ARGUMENT;
    }
    if (ctx->finished || ctx->failed) {
        return TB_ERR_STATE;
    }
    if (length != 0u && data == NULL) {
        return TB_ERR_ARGUMENT;
    }
    /* Check before calling the backend: giant bogus lengths must not be read. */
    if ((uint64_t)length > TB_MAX_SHA256_BYTES - ctx->total) {
        return TB_ERR_LENGTH;
    }
    if (length == 0u) {
        return TB_OK;
    }
    if (EVP_DigestUpdate(ctx->backend, data, length) != 1) {
        ctx->failed = 1;
        return TB_ERR_CRYPTO;
    }
    ctx->total += (uint64_t)length;
    return TB_OK;
}

tb_status tb_sha256_final(tb_sha256_ctx *ctx,
                          unsigned char out[TB_SHA256_DIGEST_SIZE])
{
    unsigned char tmp[TB_SHA256_DIGEST_SIZE];
    unsigned int size = 0;
    if (ctx == NULL || out == NULL) {
        return TB_ERR_ARGUMENT;
    }
    if (ctx->finished || ctx->failed) {
        return TB_ERR_STATE;
    }
    if (EVP_DigestFinal_ex(ctx->backend, tmp, &size) != 1 ||
        size != TB_SHA256_DIGEST_SIZE) {
        ctx->failed = 1;
        return TB_ERR_CRYPTO;
    }
    ctx->finished = 1;
    memcpy(out, tmp, TB_SHA256_DIGEST_SIZE);
    return TB_OK;
}

void tb_sha256_free(tb_sha256_ctx *ctx)
{
    if (ctx != NULL) {
        EVP_MD_CTX_free(ctx->backend);
        free(ctx);
    }
}

tb_status tb_sha256_bytes(const void *data, size_t length,
                          unsigned char out[TB_SHA256_DIGEST_SIZE])
{
    tb_sha256_ctx *ctx = NULL;
    tb_status status;
    if (out == NULL || (length != 0u && data == NULL)) {
        return TB_ERR_ARGUMENT;
    }
    if ((uint64_t)length > TB_MAX_SHA256_BYTES) {
        return TB_ERR_LENGTH;
    }
    status = tb_sha256_new(&ctx);
    if (status != TB_OK) {
        return status;
    }
    status = tb_sha256_update(ctx, data, length);
    if (status == TB_OK) {
        status = tb_sha256_final(ctx, out);
    }
    tb_sha256_free(ctx);
    return status;
}

tb_status tb_sha256_hex(const unsigned char digest[TB_SHA256_DIGEST_SIZE],
                        char out[TB_SHA256_HEX_SIZE])
{
    static const char digits[] = "0123456789abcdef";
    size_t i;
    if (digest == NULL || out == NULL) {
        return TB_ERR_ARGUMENT;
    }
    for (i = 0u; i < TB_SHA256_DIGEST_SIZE; ++i) {
        out[2u * i] = digits[digest[i] >> 4];
        out[2u * i + 1u] = digits[digest[i] & 0x0fu];
    }
    out[TB_SHA256_HEX_SIZE - 1u] = '\0';
    return TB_OK;
}
