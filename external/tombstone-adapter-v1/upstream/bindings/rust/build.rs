use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(program: &str, args: &[&str]) {
    let status = Command::new(program)
        .args(args)
        .status()
        .unwrap_or_else(|error| panic!("could not launch {program}: {error}"));
    assert!(status.success(), "{program} failed with {status}");
}

fn compile(compiler: &str, root: &Path, out: &Path, source: &str, name: &str) {
    let include = root.join("include");
    let src = root.join(source);
    let obj = out.join(name);
    run(
        compiler,
        &[
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-I",
            include.to_str().expect("UTF-8 include path"),
            "-c",
            src.to_str().expect("UTF-8 C path"),
            "-o",
            obj.to_str().expect("UTF-8 object path"),
        ],
    );
}

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());

    run("pkg-config", &["--exists", "openssl"]);
    for path in [
        "src/crypto/fingerprint.c",
        "src/core/deletion.c",
        "src/encoding/receipt.c",
        "include/tombstone/fingerprint.h",
        "include/tombstone/deletion.h",
        "include/tombstone/receipt.h",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=AR");

    compile(&cc, &root, &out, "src/crypto/fingerprint.c", "fingerprint.o");
    compile(&cc, &root, &out, "src/core/deletion.c", "deletion.o");
    compile(&cc, &root, &out, "src/encoding/receipt.c", "receipt.o");

    let archive = out.join("libtombstone_native.a");
    run(
        &ar,
        &[
            "rcsD",
            archive.to_str().expect("UTF-8 archive path"),
            out.join("fingerprint.o").to_str().expect("UTF-8 object path"),
            out.join("deletion.o").to_str().expect("UTF-8 object path"),
            out.join("receipt.o").to_str().expect("UTF-8 object path"),
        ],
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=tombstone_native");
    println!("cargo:rustc-link-lib=crypto");
}
