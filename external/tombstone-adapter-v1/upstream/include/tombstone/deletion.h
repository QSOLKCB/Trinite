/* SPDX-License-Identifier: Apache-2.0 */
#ifndef TOMBSTONE_DELETION_H
#define TOMBSTONE_DELETION_H

#include "tombstone/fingerprint.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TB_EVIDENCE_ABI_VERSION 1u
/* The proposed receipt v1 JSON schema uses JSON-safe integer sizes. */
#define TB_EVIDENCE_MAX_BYTES UINT64_C(9007199254740991)

typedef enum tb_evidence_kind {
    TB_EVIDENCE_UNCHANGED = 1,
    TB_EVIDENCE_OBSERVED_TRANSITION = 2,
    TB_EVIDENCE_EXPLICIT_DELETE = 3
} tb_evidence_kind;

/*
 * The C99 engine owns these values, but the caller owns the struct.
 * All digests are 32 RAW bytes; never interpret as NUL-terminated text.
 * For non-deletions, deleted is all zero and MUST NOT be interpreted as a
 * removed span. Only kind == TB_EVIDENCE_EXPLICIT_DELETE means it is valid.
 * This struct is not a persistent wire format (Phase 3 adds receipts).
 */
typedef struct tb_evidence_v1 {
    tb_evidence_kind kind;
    size_t before_size;
    size_t after_size;
    unsigned char before_sha256[TB_SHA256_DIGEST_SIZE];
    unsigned char after_sha256[TB_SHA256_DIGEST_SIZE];
    size_t deleted_size;
    size_t deleted_offset;
    unsigned char deleted_sha256[TB_SHA256_DIGEST_SIZE];
} tb_evidence_v1;

unsigned int tb_evidence_abi_version(void);

/*
 * Compare complete caller-supplied before/after buffers. No deletion span is
 * inferred from their difference, even when a text diff seems obvious.
 * Equal bytes => UNCHANGED; otherwise OBSERVED_TRANSITION.
 */
tb_status tb_observe_transition(const void *before, size_t before_size,
                                const void *after, size_t after_size,
                                tb_evidence_v1 *out);

/*
 * Validate a caller-witnessed exact deletion splice:
 * before == prefix || removed || suffix
 * after  == prefix || suffix
 * where prefix length is offset and removed_size > 0.
 *
 * This proves the supplied buffers have that splice relationship. It does NOT
 * prove when, where, by whom, or whether that historical edit occurred.
 * Multiple offsets may be valid in repeated data; the claimed offset is
 * checked but not independently authenticated.
 *
 * Success => EXPLICIT_DELETE and SHA-256 for before/after/removed.
 * Errors: ARGUMENT for null/nonzero buffers, zero removed span or null out;
 * LENGTH for out-of-range sizes/offset; MISMATCH for invalid splice.
 * On error, out remains completely untouched. No input is changed, retained,
 * logged, or saved by TOMBSTONE. All input buffers and out MUST be disjoint.
 * This function performs no filesystem mutation, secure erasure or I/O.
 */
tb_status tb_witness_delete(const void *before, size_t before_size,
                            const void *removed, size_t removed_size,
                            size_t offset,
                            const void *after, size_t after_size,
                            tb_evidence_v1 *out);

#ifdef __cplusplus
}
#endif
#endif /* TOMBSTONE_DELETION_H */
