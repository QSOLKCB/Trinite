#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Read-only TT-003-FIT-001 evidence review. Outputs aggregates, never weights."""
import argparse
import os
import pathlib
import sys
import zipfile

from teacher_fit_review_v1 import core


def open_output_directory(parent: pathlib.Path) -> int:
    """Walk the canonical absolute parent from / without following ancestors.

    O_NOFOLLOW on a single final component is insufficient: an attacker can
    replace an earlier directory with a symlink between resolve() and open().
    Every component is opened relative to the preceding directory descriptor.
    Confirm that the resulting handle still names the checked parent.
    """
    if not parent.is_absolute() or '..' in parent.parts:
        raise core.EvidenceError('noncanonical output parent')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    expected = parent.stat()
    fd = os.open('/', flags)
    try:
        for component in parent.parts[1:]:
            following = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = following
        actual = os.fstat(fd)
        if (actual.st_dev != expected.st_dev or actual.st_ino != expected.st_ino or
                os.readlink(f'/proc/self/fd/{fd}') != str(parent)):
            raise core.EvidenceError('output parent changed during descriptor traversal')
        return fd
    except BaseException:
        os.close(fd)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=pathlib.Path)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--archive-sha256', help='External sha256:<64 lowercase hex>')
    sources.add_argument('--sidecar', type=pathlib.Path)
    parser.add_argument('--out', type=pathlib.Path,
                        help='New local canonical JSON file, outside archive/checkout')
    args = parser.parse_args()
    try:
        anchor = (args.archive_sha256 if args.archive_sha256 else
                  core.parse_anchor(args.sidecar.read_bytes()))
        result = core.review(args.archive, anchor)
        raw = core.canonical(result)
        if args.out is not None:
            # Resolve all parent aliases *before* verifying checkout exclusion.
            # Open via the verified directory descriptor to avoid alias swaps
            # between policy check and file creation.
            parent = args.out.parent.resolve(strict=True)
            target = parent / args.out.name
            root = pathlib.Path(__file__).resolve().parents[1]
            if (target.is_relative_to(root) or
                    target == args.archive.resolve(strict=True) or
                    (args.sidecar is not None and
                     target == args.sidecar.resolve(strict=True)) or
                    target.exists() or target.is_symlink()):
                raise core.EvidenceError('unsafe/out-of-scope output destination')
            directory = open_output_directory(parent)
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                descriptor = os.open(target.name, flags, 0o600, dir_fd=directory)
                with os.fdopen(descriptor, 'wb') as out:
                    out.write(raw)
            finally:
                os.close(directory)
        else:
            sys.stdout.buffer.write(raw)
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print('TT-003-FIT-001-EVIDENCE-REVIEW rejected: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
