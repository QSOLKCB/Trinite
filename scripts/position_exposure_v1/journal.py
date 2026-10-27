"""Bounded exclusive raw prediction journal, including incomplete scoring attempts."""
import os

from trinite.contracts import ContractError, json_bytes


class PredictionJournal:
    def __init__(self, store, name, context):
        if '/' in name or not name.endswith('.jsonl'):
            raise ContractError('journal requires a jsonl basename')
        store.check()
        self.store = store
        self.context = context
        self.size = 0
        fd = os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=store.fd)
        self.stream = os.fdopen(fd,'wb')

    def append(self, prediction):
        raw = json_bytes({**self.context,'prediction':prediction})
        if self.size+len(raw)>16*1024*1024 or self.store.bytes+len(raw)>32*1024*1024:
            raise ContractError('prediction journal byte bound exceeded')
        self.store.check()
        self.stream.write(raw);self.stream.flush();os.fsync(self.stream.fileno())
        self.size += len(raw);self.store.bytes += len(raw)
        self.store.check()

    def close(self):
        self.stream.close()
