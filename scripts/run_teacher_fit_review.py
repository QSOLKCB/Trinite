#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Read-only TT-003-FIT-001 evidence review. Outputs aggregates, never weights."""
import argparse
import pathlib
import sys
import zipfile

from teacher_fit_review_v1 import core


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
            target = args.out.absolute()
            root = pathlib.Path(__file__).resolve().parents[1]
            if (target == args.archive.absolute() or
                    target.is_relative_to(root) or
                    (args.sidecar is not None and target == args.sidecar.absolute()) or
                    target.is_symlink() or not target.parent.is_dir()):
                raise core.EvidenceError('unsafe/out-of-scope output destination')
            with target.open('xb') as out:
                out.write(raw)
        else:
            sys.stdout.buffer.write(raw)
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print('TT-003-FIT-001-EVIDENCE-REVIEW rejected: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
