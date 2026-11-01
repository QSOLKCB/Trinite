/* SPDX-License-Identifier: Apache-2.0 */
#include "tombstone/receipt.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define DOMAIN "TOMBSTONE-RECEIPT-V1\n"
#define MAXN TB_EVIDENCE_MAX_BYTES
#define F_AFTER 1u
#define F_BEFORE 2u
#define F_DELETED 4u
#define F_FORMAT 8u
#define F_KIND 16u
#define F_PREV 32u
#define F_HASH 64u
#define F_VERSION 128u
#define F_ALL 255u

typedef struct tb_parser { const char *s; size_t n, i; } tb_parser;

unsigned int tb_receipt_abi_version(void) { return TB_RECEIPT_ABI_VERSION; }

static int zero32(const unsigned char *p)
{
    unsigned int i;
    for (i = 0; i < 32u; ++i) if (p[i]) return 0;
    return 1;
}
static int equal32(const unsigned char *a, const unsigned char *b)
{
    return memcmp(a, b, 32u) == 0;
}
static int good_evidence(const tb_evidence_v1 *e)
{
    if ((uint64_t)e->before_size > MAXN || (uint64_t)e->after_size > MAXN ||
        (uint64_t)e->deleted_size > MAXN || (uint64_t)e->deleted_offset > MAXN)
        return 0;
    if (e->kind == TB_EVIDENCE_UNCHANGED)
        return e->before_size == e->after_size &&
               equal32(e->before_sha256, e->after_sha256) &&
               e->deleted_size == 0 && e->deleted_offset == 0 &&
               zero32(e->deleted_sha256);
    if (e->kind == TB_EVIDENCE_OBSERVED_TRANSITION)
        return (e->before_size != e->after_size ||
                !equal32(e->before_sha256, e->after_sha256)) &&
               e->deleted_size == 0 && e->deleted_offset == 0 &&
               zero32(e->deleted_sha256);
    if (e->kind == TB_EVIDENCE_EXPLICIT_DELETE)
        return e->deleted_size > 0 && e->deleted_size <= e->before_size &&
               e->after_size == e->before_size - e->deleted_size &&
               e->deleted_offset <= e->before_size - e->deleted_size;
    return 0;
}
static int good_receipt(const tb_receipt_v1 *r)
{
    return (r->has_previous == 0 || r->has_previous == 1) &&
           (!r->has_previous ? zero32(r->previous_hash) : 1) &&
           good_evidence(&r->evidence);
}
static int hex32(const unsigned char *d, char b[TB_SHA256_HEX_SIZE])
{
    return tb_sha256_hex(d,b) == TB_OK;
}
/* For this restricted v1 schema, fixed ASCII key/value strings and JSON-safe
 * nonnegative integers make sorted/minimal decimal output RFC 8785 JCS.
 * This is deliberately NOT a general-purpose JSON canonicalizer. */
static tb_status format_json(const tb_receipt_v1 *r, int with_hash,
                             char *buffer, size_t capacity, size_t *size)
{
    char before[65], after[65], deleted[65], prev[65], own[65];
    char dfield[192], pfield[96], hfield[96];
    const tb_evidence_v1 *e = &r->evidence;
    const char *kind;
    int k;
    if (!hex32(e->before_sha256,before) || !hex32(e->after_sha256,after) ||
        !hex32(e->deleted_sha256,deleted) ||
        !hex32(r->previous_hash,prev) || !hex32(r->receipt_hash,own))
        return TB_ERR_CRYPTO;
    if (e->kind == TB_EVIDENCE_EXPLICIT_DELETE) {
        kind = "explicit-delete";
        k = snprintf(dfield,sizeof(dfield),
            "{\"offset\":%" PRIu64 ",\"sha256\":\"%s\",\"size\":%" PRIu64 "}",
            (uint64_t)e->deleted_offset, deleted, (uint64_t)e->deleted_size);
        if (k < 0 || (size_t)k >= sizeof(dfield)) return TB_ERR_LENGTH;
    } else {
        kind = e->kind == TB_EVIDENCE_UNCHANGED ? "unchanged" : "observed-transition";
        memcpy(dfield,"null",5);
    }
    if (r->has_previous) {
        k = snprintf(pfield,sizeof(pfield),"\"%s\"",prev);
        if (k < 0 || (size_t)k >= sizeof(pfield)) return TB_ERR_LENGTH;
    } else memcpy(pfield,"null",5);
    if (with_hash) {
        k = snprintf(hfield,sizeof(hfield),",\"receipt_hash\":\"%s\"",own);
        if (k < 0 || (size_t)k >= sizeof(hfield)) return TB_ERR_LENGTH;
    } else hfield[0] = '\0';
    k = snprintf(buffer,capacity,
      "{\"after\":{\"sha256\":\"%s\",\"size\":%" PRIu64 "},"
      "\"before\":{\"sha256\":\"%s\",\"size\":%" PRIu64 "},"
      "\"deleted\":%s,\"format\":\"tombstone.receipt\","
      "\"kind\":\"%s\",\"previous_receipt_hash\":%s%s,\"version\":1}",
      after,(uint64_t)e->after_size,before,(uint64_t)e->before_size,
      dfield,kind,pfield,hfield);
    if (k < 0 || (size_t)k >= capacity) return TB_ERR_LENGTH;
    *size = (size_t)k;
    return TB_OK;
}
static tb_status calculate_hash(const tb_receipt_v1 *r, unsigned char out[32])
{
    char buffer[TB_RECEIPT_JSON_MAX];
    tb_sha256_ctx *ctx = NULL;
    tb_status st;
    size_t n;
    if (!good_receipt(r)) return TB_ERR_FORMAT;
    st = format_json(r,0,buffer,sizeof(buffer),&n);
    if (st != TB_OK) return st;
    st = tb_sha256_new(&ctx);
    if (st != TB_OK) return st;
    st = tb_sha256_update(ctx,DOMAIN,sizeof(DOMAIN)-1u);
    if (st == TB_OK) st = tb_sha256_update(ctx,buffer,n);
    if (st == TB_OK) st = tb_sha256_final(ctx,out);
    tb_sha256_free(ctx);
    return st;
}
tb_status tb_receipt_create(const tb_evidence_v1 *e,
                            const unsigned char previous_hash[32],tb_receipt_v1 *out)
{
    tb_receipt_v1 tmp;
    tb_status st;
    if (e == NULL || out == NULL) return TB_ERR_ARGUMENT;
    if (!good_evidence(e)) return TB_ERR_FORMAT;
    memset(&tmp,0,sizeof(tmp));
    tmp.evidence = *e;
    if (previous_hash != NULL) {
        tmp.has_previous=1;
        memcpy(tmp.previous_hash,previous_hash,32);
    }
    st = calculate_hash(&tmp,tmp.receipt_hash);
    if (st == TB_OK) *out=tmp;
    return st;
}
tb_status tb_receipt_encode(const tb_receipt_v1 *r,char *buffer,
                            size_t capacity,size_t *written)
{
    char tmp[TB_RECEIPT_JSON_MAX];
    unsigned char hash[32];
    size_t n;
    tb_status st;
    if (!r || !buffer || !written) return TB_ERR_ARGUMENT;
    st = calculate_hash(r,hash);
    if (st != TB_OK) return st;
    if (!equal32(hash,r->receipt_hash)) return TB_ERR_MISMATCH;
    st = format_json(r,1,tmp,sizeof(tmp),&n);
    if (st != TB_OK) return st;
    if (n > capacity) return TB_ERR_LENGTH;
    memcpy(buffer,tmp,n);
    *written=n;
    return TB_OK;
}
/* Schema-aware JSON parser. Reject unknown/duplicate fields and non-ASCII.
 * Decode ASCII \u escapes so escaped duplicate keys are also rejected. */
static void ws(tb_parser *p)
{
    while(p->i<p->n && (p->s[p->i]==' ' || p->s[p->i]=='\t' ||
          p->s[p->i]=='\r' || p->s[p->i]=='\n')) ++p->i;
}
static int take(tb_parser *p, char c)
{
    ws(p);
    if (p->i<p->n && p->s[p->i]==c) { ++p->i; return 1; }
    return 0;
}
static int lit(tb_parser *p,const char *s)
{
    size_t n=strlen(s);
    ws(p);
    if(n<=p->n-p->i && memcmp(p->s+p->i,s,n)==0) {p->i+=n;return 1;}
    return 0;
}
static int string(tb_parser *p,char *dst,size_t cap)
{
    size_t j=0;
    if (!take(p,'"')) return 0;
    while(p->i<p->n) {
        unsigned int c=(unsigned char)p->s[p->i++];
        if(c=='"') { if(j>=cap) return 0;dst[j]='\0';return 1; }
        if(c=='\\') {
            unsigned int v=0; unsigned int k;
            if(p->i>=p->n) return 0;
            c=(unsigned char)p->s[p->i++];
            if(c=='u') {
                if(p->n-p->i<4u) return 0;
                for(k=0;k<4u;++k) {
                    unsigned int d=(unsigned char)p->s[p->i++];
                    if(d>='0'&&d<='9') v=v*16u+d-'0';
                    else if(d>='a'&&d<='f') v=v*16u+d-'a'+10u;
                    else if(d>='A'&&d<='F') v=v*16u+d-'A'+10u;
                    else return 0;
                }
                if(v<32u || v>127u) return 0;
                c=v;
            } else if(c=='"'||c=='\\'||c=='/') { }
            else return 0;
        } else if(c<32u || c>=127u) return 0;
        if(j+1u>=cap) return 0;
        dst[j++]=(char)c;
    }
    return 0;
}
static int number(tb_parser *p,size_t *out)
{
    uint64_t x=0;
    unsigned int digit;
    ws(p);
    if(p->i>=p->n || p->s[p->i]<'0' || p->s[p->i]>'9') return 0;
    if(p->s[p->i]=='0' && p->i+1u<p->n &&
       p->s[p->i+1u]>='0' && p->s[p->i+1u]<='9') return 0;
    do {
        digit=(unsigned int)(p->s[p->i++]-'0');
        if(x>(MAXN-digit)/10u) return 0;
        x=x*10u+digit;
    } while(p->i<p->n && p->s[p->i]>='0' && p->s[p->i]<='9');
    if(x>SIZE_MAX) return 0;
    *out=(size_t)x;
    return 1;
}
static int digest(tb_parser *p,unsigned char out[32])
{
    char str[66]; size_t i;
    if(!string(p,str,sizeof(str)) || strlen(str)!=64u) return 0;
    for(i=0;i<32u;++i) {
        unsigned int k,v=0;
        for(k=0;k<2u;++k) {
            unsigned int c=(unsigned char)str[2u*i+k];
            if(c>='0'&&c<='9') v=v*16u+c-'0';
            else if(c>='a'&&c<='f') v=v*16u+c-'a'+10u;
            else return 0;
        }
        out[i]=(unsigned char)v;
    }
    return 1;
}
static int object_fp(tb_parser *p,size_t *size,unsigned char hash[32],
                     size_t *offset,int with_offset)
{
    unsigned int mask=0,required=with_offset?7u:3u;
    if(!take(p,'{')) return 0;
    for(;;) {
        char key[40]; unsigned int bit;
        if(!string(p,key,sizeof(key)) || !take(p,':')) return 0;
        if(strcmp(key,"sha256")==0) bit=1u;
        else if(strcmp(key,"size")==0) bit=2u;
        else if(with_offset && strcmp(key,"offset")==0) bit=4u;
        else return 0;
        if(mask&bit) return 0;
        mask|=bit;
        if(bit==1u) { if(!digest(p,hash)) return 0; }
        else if(bit==2u) { if(!number(p,size)) return 0; }
        else { if(!number(p,offset)) return 0; }
        if(take(p,'}')) break;
        if(!take(p,',')) return 0;
    }
    return mask==required;
}
tb_status tb_receipt_parse(const char *json,size_t len,tb_receipt_v1 *out)
{
    tb_parser p;
    tb_receipt_v1 tmp;
    unsigned int mask=0;
    unsigned char correct[32];
    tb_status st;
    if(!json || !out) return TB_ERR_ARGUMENT;
    if(len>16384u) return TB_ERR_LENGTH;
    p.s=json;p.n=len;p.i=0u;
    memset(&tmp,0,sizeof(tmp));
    if(!take(&p,'{')) return TB_ERR_FORMAT;
    for(;;) {
        char key[48],value[80]; unsigned int bit;
        if(!string(&p,key,sizeof(key)) || !take(&p,':')) return TB_ERR_FORMAT;
        if(strcmp(key,"after")==0) bit=F_AFTER;
        else if(strcmp(key,"before")==0) bit=F_BEFORE;
        else if(strcmp(key,"deleted")==0) bit=F_DELETED;
        else if(strcmp(key,"format")==0) bit=F_FORMAT;
        else if(strcmp(key,"kind")==0) bit=F_KIND;
        else if(strcmp(key,"previous_receipt_hash")==0) bit=F_PREV;
        else if(strcmp(key,"receipt_hash")==0) bit=F_HASH;
        else if(strcmp(key,"version")==0) bit=F_VERSION;
        else return TB_ERR_FORMAT;
        if(mask&bit) return TB_ERR_FORMAT;
        mask|=bit;
        if(bit==F_AFTER) {
            if(!object_fp(&p,&tmp.evidence.after_size,tmp.evidence.after_sha256,NULL,0)) return TB_ERR_FORMAT;
        } else if(bit==F_BEFORE) {
            if(!object_fp(&p,&tmp.evidence.before_size,tmp.evidence.before_sha256,NULL,0)) return TB_ERR_FORMAT;
        } else if(bit==F_DELETED) {
            if(!lit(&p,"null")) {
                if(!object_fp(&p,&tmp.evidence.deleted_size,tmp.evidence.deleted_sha256,&tmp.evidence.deleted_offset,1)) return TB_ERR_FORMAT;
                if(tmp.evidence.deleted_size==0u) return TB_ERR_FORMAT;
            }
        } else if(bit==F_FORMAT) {
            if(!string(&p,value,sizeof(value)) || strcmp(value,"tombstone.receipt")) return TB_ERR_FORMAT;
        } else if(bit==F_KIND) {
            if(!string(&p,value,sizeof(value))) return TB_ERR_FORMAT;
            if(strcmp(value,"explicit-delete")==0) tmp.evidence.kind=TB_EVIDENCE_EXPLICIT_DELETE;
            else if(strcmp(value,"observed-transition")==0) tmp.evidence.kind=TB_EVIDENCE_OBSERVED_TRANSITION;
            else if(strcmp(value,"unchanged")==0) tmp.evidence.kind=TB_EVIDENCE_UNCHANGED;
            else return TB_ERR_FORMAT;
        } else if(bit==F_PREV) {
            if(!lit(&p,"null")) {
                if(!digest(&p,tmp.previous_hash)) return TB_ERR_FORMAT;
                tmp.has_previous=1;
            }
        } else if(bit==F_HASH) {
            if(!digest(&p,tmp.receipt_hash)) return TB_ERR_FORMAT;
        } else {
            size_t version;
            if(!number(&p,&version) || version!=1u) return TB_ERR_FORMAT;
        }
        if(take(&p,'}')) break;
        if(!take(&p,',')) return TB_ERR_FORMAT;
    }
    ws(&p);
    if(p.i!=p.n || mask!=F_ALL || !good_receipt(&tmp)) return TB_ERR_FORMAT;
    st=calculate_hash(&tmp,correct);
    if(st!=TB_OK) return st;
    if(!equal32(correct,tmp.receipt_hash)) return TB_ERR_MISMATCH;
    *out=tmp;
    return TB_OK;
}
tb_status tb_receipt_verify_chain(const tb_receipt_v1 *receipts,size_t count,
                                  const unsigned char trusted_head[32])
{
    unsigned char computed[32];
    size_t i;
    tb_status st;
    if(!receipts || count==0u || count>SIZE_MAX/sizeof(*receipts)) return TB_ERR_ARGUMENT;
    for(i=0;i<count;++i) {
        if(i==0u) {
            if(receipts[i].has_previous) return TB_ERR_MISMATCH;
        } else if(!receipts[i].has_previous ||
                  !equal32(receipts[i].previous_hash,receipts[i-1u].receipt_hash))
            return TB_ERR_MISMATCH;
        st=calculate_hash(&receipts[i],computed);
        if(st!=TB_OK) return st;
        if(!equal32(computed,receipts[i].receipt_hash)) return TB_ERR_MISMATCH;
    }
    if(trusted_head && !equal32(trusted_head,receipts[count-1u].receipt_hash))
        return TB_ERR_MISMATCH;
    return TB_OK;
}
