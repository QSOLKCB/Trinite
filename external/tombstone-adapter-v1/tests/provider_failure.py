"""Inject actual OpenSSL failures into the real C99-linked binary on Linux.

Test-only LD_PRELOAD, never a production provider or a splice classifier.
"""
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())
request = json.dumps({"schema": "trinite.tombstone-request.v1", "before": [0, 255],
                      "removed": [255], "after": [0], "offset": 1}).encode()
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    # Allocation failures at each of the three SHA calls. Real C99 cleanup/error
    # propagation executes, no modification to imported production source.
    source = root / "failure.c"
    source.write_text('''#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
EVP_MD_CTX *EVP_MD_CTX_new(void) {
    static int calls;
    int site = atoi(getenv("FAIL_SITE"));
    if (++calls == site && strcmp(getenv("FAIL_KIND"), "allocation") == 0) return NULL;
    EVP_MD_CTX *(*real_new)(void) = dlsym(RTLD_NEXT, "EVP_MD_CTX_new");
    return real_new();
}
int EVP_DigestFinal_ex(EVP_MD_CTX *ctx, unsigned char *md, unsigned int *size) {
    static int calls;
    if (++calls == atoi(getenv("FAIL_SITE")) && strcmp(getenv("FAIL_KIND"), "crypto") == 0) return 0;
    int (*real_final)(EVP_MD_CTX *, unsigned char *, unsigned int *) = dlsym(RTLD_NEXT, "EVP_DigestFinal_ex");
    return real_final(ctx, md, size);
}
''')
    library = root / "failure.so"
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library), "-ldl"], check=True, timeout=30)
    for kind, site in ((kind, site) for kind in ("allocation", "crypto") for site in (1, 2, 3)):
        env = dict(os.environ, LD_PRELOAD=str(library), FAIL_SITE=str(site), FAIL_KIND=kind)
        result = subprocess.run([binary, "evaluate-privileged"], input=request, capture_output=True, timeout=5, env=env)
        assert (result.returncode, result.stdout, result.stderr) == (3, b"", b"infrastructure failure\n"), (site, result)
        observed = subprocess.run([binary, "observe"], input=request, capture_output=True, timeout=5, env=env)
        assert observed.returncode == 0 and "label" not in json.loads(observed.stdout)
print("real C99: 3 allocation + 3 crypto failure sites remain infrastructure; observation calls no provider")
