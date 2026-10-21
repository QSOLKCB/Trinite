"""Offline quarantine only. Never runs a teacher, downloads assets or trains a student."""
import sys
if not sys.flags.isolated or not sys.flags.no_site:
    raise SystemExit('Use python -I -S scripts/run_teacher_intake.py ...')
import argparse
import os
from pathlib import Path
import tempfile
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
sys.dont_write_bytecode=True
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from teacher_intake_v1.spec import check_assets, request, validate_request
from teacher_intake_v1.intake import seal, verify
LIMIT=8*1024*1024


def read(path):
    if path.is_symlink() or not path.is_file():
        raise ContractError('regular non-symlink artifact required')
    with path.open('rb') as stream:
        raw=stream.read(LIMIT+1)
    if len(raw)>LIMIT:
        raise ContractError('artifact exceeds8MiB')
    return raw


def publish(path,raw):
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.link(name,path)
    finally:
        os.unlink(name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','seal','verify'))
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--asset-root',type=Path)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--request-identity')
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--packet',type=Path)
    parser.add_argument('--packet-identity')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    attempt=None
    try:
        if args.action=='verify':
            if args.packet is None or args.packet_identity is None:
                raise ContractError('verify requires packet and independently retained identity')
            raw=read(args.packet)
            if identity(raw)!=args.packet_identity:
                raise ContractError('packet identity mismatch')
            packet=verify(parse_json(raw,canonical=True))
            print(json_bytes({'verified':True,'admission':packet['admission'],
                              'student_execution':packet['student_execution']}).decode(),end='')
            return 0
        if args.output is None:
            raise ContractError('fresh output path required')
        if args.output.exists() or args.output.is_symlink():
            raise ContractError('output exists; preserve it and select a fresh path')
        if args.action=='freeze':
            if args.manifest is None or args.asset_root is None:
                raise ContractError('freeze requires manifest and local asset root')
            manifest=parse_json(read(args.manifest),canonical=True)
            check_assets(manifest,args.asset_root)
            completed=json_bytes(request(manifest))
        else:
            if args.request is None or args.request_identity is None or args.capture is None:
                raise ContractError('seal requires frozen request, independent identity and capture')
            request_raw=read(args.request);capture_raw=read(args.capture)
            # Retain even structurally invalid bounded captures before validation.
            attempt=args.output.with_name(args.output.name+'.attempt-'+uuid.uuid4().hex)
            attempt.mkdir()
            publish(attempt/'request.json',request_raw)
            publish(attempt/'capture.json',capture_raw)
            if identity(request_raw)!=args.request_identity:
                raise ContractError('request identity mismatch')
            frozen=validate_request(parse_json(request_raw,canonical=True))
            completed=json_bytes(seal(frozen,parse_json(capture_raw,canonical=True)))
            if len(completed)>LIMIT:
                raise ContractError('quarantine packet exceeds8MiB')
            publish(attempt/'prepared-packet.json',completed)
            publish(attempt/'preparation.json',json_bytes({'phase':'prepared_not_published',
                    'packet_identity':identity(completed),'admission':'see_prepared_packet'}))
        # The atomic final packet is itself the complete outcome, not a success
        # receipt written before a later report publication can fail.
        publish(args.output,completed)
        print(identity(completed))
        if attempt is not None:
            print(attempt)
        return 0
    except (ContractError,OSError,ValueError) as error:
        if attempt is not None:
            try:
                publish(attempt/'failure.json',json_bytes({'phase':'failed','error':str(error)[:512],
                        'admission':'NOT_ADMITTED','student_execution':'BLOCKED'}))
            except OSError:
                pass
        print(str(error)+(('; retained inputs: '+str(attempt)) if attempt else ''),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
