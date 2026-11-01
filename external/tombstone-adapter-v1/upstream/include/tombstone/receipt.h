/* SPDX-License-Identifier: Apache-2.0 */
#ifndef TOMBSTONE_RECEIPT_H
#define TOMBSTONE_RECEIPT_H
#include "tombstone/deletion.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TB_RECEIPT_ABI_VERSION 1u
#define TB_RECEIPT_JSON_MAX 1024u

/* In-memory state, NOT a wire-format struct. Hashes are raw 32-byte arrays. */
typedef struct tb_receipt_v1 {
    tb_evidence_v1 evidence;
    int has_previous;                  /* 0: genesis; 1: predecessor hash */
    unsigned char previous_hash[TB_SHA256_DIGEST_SIZE];
    unsigned char receipt_hash[TB_SHA256_DIGEST_SIZE];
} tb_receipt_v1;

unsigned int tb_receipt_abi_version(void);

/* Constructs receipt from caller-supplied evidence. No independent proof of
 * historical edits. NULL previous_hash marks genesis. Output unchanged on
 * error. Invalid evidence -> TB_ERR_FORMAT. */
tb_status tb_receipt_create(const tb_evidence_v1 *evidence,
                            const unsigned char previous_hash[TB_SHA256_DIGEST_SIZE],
                            tb_receipt_v1 *out);

/* Compact RFC 8785 canonical JSON for restricted receipt-v1 schema.
 * Byte count returned in *written; no newline or terminating NUL.
 * Input/output must not overlap. Output unchanged on failure. */
tb_status tb_receipt_encode(const tb_receipt_v1 *receipt,
                            char *buffer, size_t capacity, size_t *written);

/* Strict bounded JSON parser; supports arbitrary key order/whitespace and
 * ASCII \u escapes. Rejects duplicate/unknown keys (including escaped
 * duplicates), invalid numbers, non-ASCII/surrogates, incorrect receipt
 * digests and invalid version/type. Output unchanged on failure. */
tb_status tb_receipt_parse(const char *json, size_t length, tb_receipt_v1 *out);

/* Verify internal hashes, genesis and predecessor links; optionally require
 * externally trusted final hash. An unanchored chain is locally consistent,
 * NOT proof against whole-chain rewriting, authorship or time. */
tb_status tb_receipt_verify_chain(const tb_receipt_v1 *receipts, size_t count,
                                  const unsigned char trusted_head[TB_SHA256_DIGEST_SIZE]);
#ifdef __cplusplus
}
#endif
#endif
