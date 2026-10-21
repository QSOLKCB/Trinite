"""Retain every attempt; admit only the explicitly approved formal subset."""
import base64
import binascii
from trinite.contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json
from .spec import COUNT, PROTOCOL, validate_request


def counter(n,a,b):
    value=a
    for _ in range(b):
        value+=1
        if value==n:
            value=0
    return value


def proposal(raw, n, a):
    if len(raw)>PROTOCOL['accepted_response_bytes']:
        raise ContractError('response exceeds acceptance bound')
    value=exact_keys(parse_json(raw), {'n','a','b'}, 'proposal')
    integer(value['n'],'n',n,n)
    for key in ('a','b'):
        integer(value[key],key,0,n-1)
    integer(value['a'],'slot a',a,a)
    return value


def seal(frozen, capture):
    validate_request(frozen)
    rid=identity(json_bytes(frozen))
    exact_keys(capture, {'schema','request_identity','attempts'}, 'capture')
    if capture['schema']!='trinite.teacher-capture.v1' or capture['request_identity']!=rid:
        raise ContractError('capture request binding mismatch')
    attempts=capture['attempts']
    if type(attempts) is not list or len(attempts)!=COUNT:
        raise ContractError('all32 fixed attempts, including failures, are required')
    receipts=[]; items=[]; seen=set()
    for i,attempt in enumerate(attempts):
        exact_keys(attempt, {'index','outcome','raw_base64','final_start','final_end'}, 'attempt')
        integer(attempt['index'],'attempt index',i,i)
        if attempt['outcome'] not in ('completed','timeout','runtime_error'):
            raise ContractError('unknown attempt outcome')
        encoded=attempt['raw_base64']
        if type(encoded) is not str or len(encoded)>87384:
            raise ContractError('raw response retention bound exceeded')
        try:
            raw=base64.b64decode(encoded,validate=True)
        except (binascii.Error,ValueError) as error:
            raise ContractError('invalid raw base64') from error
        if base64.b64encode(raw).decode('ascii')!=encoded or len(raw)>65536:
            raise ContractError('noncanonical or oversized raw response encoding')
        start=integer(attempt['final_start'],'final start',0,len(raw))
        end=integer(attempt['final_end'],'final end',start,len(raw))
        final=raw[start:end]
        receipt={'index':i,'outcome':attempt['outcome'],'raw_identity':identity(raw),
                 'raw_bytes':len(raw),'final_identity':identity(final),'decision':'rejected','reason':attempt['outcome']}
        if attempt['outcome']=='completed':
            try:
                value=proposal(final,frozen['slots'][i]['n'],frozen['slots'][i]['a'])
                n,a,b=(value[k] for k in ('n','a','b'))
                family=(n,min(a,b),max(a,b))
                if family in seen:
                    raise ContractError('duplicate commutative family')
                answer=counter(n,a,b)
                if answer!=(a+b)%n:
                    raise ContractError('independent oracle mismatch')
                seen.add(family)
                item={'source_id':'trinite.teacher-proposed-formal-inputs.v1',
                      'generator_revision':identity(json_bytes(frozen['source'])), 'seed':0,
                      'split':'train', 'split_procedure':'teacher slots only moduli5/7/9 train; heldout11/13 never proposed',
                      'formal_spec':value, 'family':list(family),
                      'prompt':f'addmod[{n:02d},{a:02d},{b:02d}]=', 'answer':f'{answer:02d}',
                      'verifier':{'outcome':'exact','producer':'bounded cyclic counter',
                                  'independent_check':'integer modulo'},
                      'lineage':{'request_identity':rid,'attempt':i,'raw_identity':identity(raw),
                                 'teacher_origin':'selection of numeric operands; never erased by oracle relabelling'},
                      'rights_basis':'explicit owner-approved bounded teacher-origin numeric-input policy; no prose admission',
                      'admission':'ADMITTED_FORMAL_ONLY',
                      'policy':'trinite.teacher-origin-formal-policy.v1',
                      'reviewer':'Trent Slade / QSOLKCB', 'review_date':'2026-10-21'}
                item['identity']=identity(json_bytes(item));items.append(item)
                receipt.update(decision='formally_valid',reason='exact bounded numeric proposal')
            except ContractError as error:
                receipt['reason']=str(error)
        receipts.append(receipt)
    return {'schema':'trinite.teacher-quarantine.v1', 'request':frozen,'request_identity':rid,
            'capture':capture,'capture_identity':identity(json_bytes(capture)),
            'receipts':receipts,'items':items,'items_identity':identity(json_bytes(items)),
            'counts':{'attempts':COUNT,'valid_unique':len(items),'rejected':COUNT-len(items)},
            'intake_feasible':len(items)>=16,
            'admission':'ADMITTED_FORMAL_ONLY' if len(items)>=16 else 'BLOCKED_UNDERFILLED',
            'raw_admission':'QUARANTINED_NOT_TRAINING_DATA',
            'student_execution':'SEPARATE_FROZEN_CPU_REQUEST' if len(items)>=16 else 'BLOCKED',
            'scope':'formal-only scoped admission; no teacher quality, legal guarantee, learning or execution claim'}


def verify(packet):
    if type(packet) is not dict or 'request' not in packet or 'capture' not in packet:
        raise ContractError('quarantine packet required')
    if json_bytes(packet)!=json_bytes(seal(packet['request'],packet['capture'])):
        raise ContractError('quarantine packet does not exactly regenerate')
    return packet
