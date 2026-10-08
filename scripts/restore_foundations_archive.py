"""Restore exact review files from a bounded, losslessly deduplicated ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

SCHEMA='trinite.lossless-review-archive.v1'


def digest(raw):return 'sha256:'+hashlib.sha256(raw).hexdigest()


def pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('duplicate archive index key')
        result[key]=value
    return result


def restore(archive,destination,expected_identity):
    archive=Path(archive);destination=Path(destination)
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',expected_identity):raise ValueError('invalid archive identity')
    with archive.open('rb') as stream:raw=stream.read(64*1024*1024+1)
    if len(raw)>64*1024*1024 or digest(raw)!=expected_identity:raise ValueError('archive size/identity mismatch')
    if destination.exists() or destination.is_symlink():raise ValueError('destination must be new')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries=z.infolist();names=[v.filename for v in entries]
        if len(names)>4096 or len(set(names))!=len(names):raise ValueError('duplicate/oversized ZIP inventory')
        info=z.getinfo('restore-index.json')
        if info.file_size>1024*1024:raise ValueError('oversized restore index')
        index=json.loads(z.read(info),object_pairs_hook=pairs)
        if type(index) is not dict or set(index)!={'schema','files'} or index['schema']!=SCHEMA:
            raise ValueError('invalid restore index')
        files=index['files']
        if type(files) is not dict or not files or len(files)>4096:raise ValueError('invalid file inventory')
        folded_files={name.casefold() for name in files}
        objects={};folded=set();total=0
        for name,record in files.items():
            p=PurePosixPath(name)
            if ('\\' in name or '\0' in name or p.is_absolute() or '..' in p.parts
                    or p.as_posix()!=name or not (name=='members.json' or name.startswith(('run/','producer/')))
                    or name.casefold() in folded or len(name)>2048
                    or any(str(parent).casefold() in folded_files for parent in p.parents)):
                raise ValueError('unsafe/colliding restore path')
            folded.add(name.casefold())
            if (type(record) is not dict or set(record)!={'bytes','identity'}
                    or type(record['bytes']) is not int or not 0<=record['bytes']<=32*1024*1024
                    or type(record['identity']) is not str
                    or not re.fullmatch(r'sha256:[0-9a-f]{64}',record['identity'])):
                raise ValueError('invalid file receipt')
            total+=record['bytes'];key='objects/sha256/'+record['identity'][7:]
            if key in objects and objects[key]!=record['bytes']:raise ValueError('contradictory object size')
            objects[key]=record['bytes']
        if total>512*1024*1024 or set(names)!={'restore-index.json',*objects}:
            raise ValueError('incomplete/unexpected or oversized archive')
        payloads={}
        for key,size in objects.items():
            entry=z.getinfo(key)
            if entry.file_size!=size:raise ValueError('object size mismatch')
            payload=z.read(entry)
            if digest(payload)!='sha256:'+key.rsplit('/',1)[1]:raise ValueError('object hash mismatch')
            payloads[key]=payload
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='trinite-review-',dir=destination.parent) as directory:
        stage=Path(directory)/'restored';stage.mkdir()
        for name,record in files.items():
            path=stage/name;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(payloads['objects/sha256/'+record['identity'][7:]])
        if destination.exists() or destination.is_symlink():raise ValueError('destination became occupied')
        stage.rename(destination)
    return {'archive_identity':expected_identity,'restored_files':len(files),'restored_bytes':total}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path);parser.add_argument('destination',type=Path)
    parser.add_argument('--archive-identity',required=True)
    args=parser.parse_args();print(json.dumps(restore(args.archive,args.destination,args.archive_identity),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
