"""Immutable reports-only placement/device-result checks; no model/backend execution."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/music-placement-result-v1'
LIMIT = 1024 * 1024
ORIGINAL_HASHES = {'cpu/audit.stdout': 'sha256:a8a8492cc01bbca8593efe97a43e2fabb0f208b6a4617d3eb50a103c2f6b0b7f', 'cpu/control-17/attempt.json': 'sha256:6dcdd1cf8f4bdbd8434a079cdd5494cdbe57c365a13bdc7e4d1830222c1dbef2', 'cpu/control-17/completion.json': 'sha256:991728d785210d11aa1b9c37c086ad19068a5b93984febd211e6714e587bbd74', 'cpu/control-17/diagnostic-predictions.jsonl': 'sha256:83589d201a2ed03121569d955723a4c5122b666d997811eda061651259c8f4b0', 'cpu/control-17/report.json': 'sha256:b8fb39837a5ef5c716a6e67ae6979a29918e1218a5468cbf60159576c9cc052a', 'cpu/control-29/attempt.json': 'sha256:5ac9ad2b5cb3eb94a481bc6411e41be93f5b52a20896ff7eb7049d4a6e4a0aa7', 'cpu/control-29/completion.json': 'sha256:3759ea8127bd620817fc940abed1eb3f13c9788bc7f2a6cde03efd1dea3525a6', 'cpu/control-29/diagnostic-predictions.jsonl': 'sha256:40c42c1bf4af61393591348b45ab5b78e74a1d916bc6960192b5e4516904cb68', 'cpu/control-29/report.json': 'sha256:293aeb8b589433abdc041dd358a8e9d6f95d5ff053c4542b31f8899afbb62771', 'cpu/development/attempt.json': 'sha256:89ef1632937fc38c0498fdd1154c42b9a32a32e15ff4236103c1775bda17515d', 'cpu/development/completion.json': 'sha256:76fa72c9839384e146947a46c1aa1a9b96a84f264a2fc90c9bd81cd66ffae4a0', 'cpu/development/predictions.jsonl': 'sha256:6f5b4d236d1f8535e34cae27d4ea60066dde529a6cd2fed3b37d97ce4325ea88', 'cpu/development/report.json': 'sha256:4ec924ac3338f212752a36bee7814d623a5753d36de004126b38487f60e18d5a', 'cpu/development-gate.json': 'sha256:5471942d8aea2a01327cf78ace3f024d11b9d06096be0b7bdf9965a6657adb56', 'cpu/matrix.json': 'sha256:a25c8b1a2a0b66bb536c5fb02c7ed2fa359bdb62845b36fd445eb4b1bf21896e', 'cpu/swap-17/attempt.json': 'sha256:292c01a61189fe480459d1618b44a1c219cd88cd5383740236220c5f8655ea42', 'cpu/swap-17/completion.json': 'sha256:aeb1bbbd99a43d4d55e3443659084a64b8dea97eeb79bda9b2c0e9a5feb3181c', 'cpu/swap-17/diagnostic-predictions.jsonl': 'sha256:652b51f2b8e7b68858dfc7c4082301d278b3bfb9854c6fcdcf21ee6d5eafd692', 'cpu/swap-17/report.json': 'sha256:cb0f0af5d2b93728d3a88aca06dbb972a264de80c710897f1bbcd6402e96a4fb', 'cpu/swap-29/attempt.json': 'sha256:9ccb14f0e85aed4b0fbfeb7eb583d02534f29d31f92662a25a008bbb27bd554f', 'cpu/swap-29/completion.json': 'sha256:9cc79d31d7a9d69210f69c2a7ad214e5ef1f7fe74f1fb606fe2413ced4d22e1d', 'cpu/swap-29/diagnostic-predictions.jsonl': 'sha256:dc106535e2c2b5493bf8ffb684ad82282ca760e437064484cd48ccbab9ffd283', 'cpu/swap-29/report.json': 'sha256:00f7679561308532a34f41b62365c3d1e2b8f1a1876001797a96fa6fd2940101', 'device/audit.stdout': 'sha256:a2712ca2eb45f1fc7aea67e54e37263df6be5ad8068bb7f0c3cbf3ef5eec8798', 'device/cpu/attempt.json': 'sha256:b332c08ac7dfe13ca256fc8c2555452110b17e850893f869d01e189d27831ede', 'device/cpu/completion.json': 'sha256:3a90046fd19a6615542d5194515122c6e5e73df9f0774fb0eb575a30ec690144', 'device/cpu/diagnostic-predictions.jsonl': 'sha256:41eb9b8d0dd70c7e852bb45a55c27656df2a42d5dd4f78afa6974a73fe248c58', 'device/cpu/report.json': 'sha256:d46773c26352984fb2b4f164a858a57af5388f42bfb145498a145d65507db037', 'device/cuda/attempt.json': 'sha256:6c223cc64bac1bb64a345e0a16d43014ae440c45a3b2021afe07750fe64f1746', 'device/cuda/completion.json': 'sha256:2839ce6019d4c35687abb82fa68718b059f8989fe95d7169fa80bd60a787c31a', 'device/cuda/diagnostic-predictions.jsonl': 'sha256:ffa403e17ed1eb6826e5d00f508655e243145974c6b7b48e0e1535bc6781f182', 'device/cuda/report.json': 'sha256:473abcd161d879e1c65af29a04892fe43b91cac4c5490a84a5400b80e7764e6a', 'device/development/attempt.json': 'sha256:11e7a0e765522d0dd0542fbba16fd2adaff4234151e175090418d5b9ff63317d', 'device/development/completion.json': 'sha256:e4c5834daa3262c9a1bc72d05440964fbebe7ebc8efb03ff4678290828dbaba8', 'device/development/predictions.jsonl': 'sha256:7f33603c154d8c22c6d7075ec01329bb75775cd37642fe67d3a0bf24ca948e55', 'device/development/report.json': 'sha256:021b03aee77e0356fae4f393ae63f58194a2e151d69f497dfca272c0f01a6f4a', 'device/development-gate.json': 'sha256:24bb11366ff5df48dd0af43e0ac01ee641abc763a13b042812e53f2387f49910', 'device/matrix.json': 'sha256:e6641be61e71d9b1db8919d0e224d95c66bb2b1ec46038d2c8eec340b5c37bb5', 'device/probe/attempt.json': 'sha256:6a41f1845bf97ad7e486939d870e90f956b119ac328088ca0f972fba774a4982', 'device/probe/comparison.json': 'sha256:4da425bb13cef1d5bf758bd68f319b57dd93d656fc891350ced1ccb2bf585c23', 'device/probe/completion.json': 'sha256:bd377f6164dd1ac7cef44a3b8a38ea47362770ad2a5907a1b2024e05c4412aa3', 'device/probe/report.json': 'sha256:5483eebb2086daf76c3780024a9ceaf6f95022b1b9d5574bf7f046d24698734b', 'source-SHA256SUMS': 'sha256:9178b5006ac3a0ea008fdc0b070f599790df45dcbd983b8c093627de074e0f6f'}
FIXTURE_HASHES = {'README.md': 'sha256:a127c3b0d932dd8faa019c37ba3b6907df18ee678a3ae4a11716a1690b0ff213', 'cpu/audit.stdout': 'sha256:a8a8492cc01bbca8593efe97a43e2fabb0f208b6a4617d3eb50a103c2f6b0b7f', 'cpu/control-17/attempt.json': 'sha256:6dcdd1cf8f4bdbd8434a079cdd5494cdbe57c365a13bdc7e4d1830222c1dbef2', 'cpu/control-17/completion.json': 'sha256:991728d785210d11aa1b9c37c086ad19068a5b93984febd211e6714e587bbd74', 'cpu/control-17/diagnostic-predictions.jsonl': 'sha256:83589d201a2ed03121569d955723a4c5122b666d997811eda061651259c8f4b0', 'cpu/control-17/report.json': 'sha256:b8fb39837a5ef5c716a6e67ae6979a29918e1218a5468cbf60159576c9cc052a', 'cpu/control-29/attempt.json': 'sha256:5ac9ad2b5cb3eb94a481bc6411e41be93f5b52a20896ff7eb7049d4a6e4a0aa7', 'cpu/control-29/completion.json': 'sha256:3759ea8127bd620817fc940abed1eb3f13c9788bc7f2a6cde03efd1dea3525a6', 'cpu/control-29/diagnostic-predictions.jsonl': 'sha256:40c42c1bf4af61393591348b45ab5b78e74a1d916bc6960192b5e4516904cb68', 'cpu/control-29/report.json': 'sha256:293aeb8b589433abdc041dd358a8e9d6f95d5ff053c4542b31f8899afbb62771', 'cpu/development/attempt.json': 'sha256:89ef1632937fc38c0498fdd1154c42b9a32a32e15ff4236103c1775bda17515d', 'cpu/development/completion.json': 'sha256:76fa72c9839384e146947a46c1aa1a9b96a84f264a2fc90c9bd81cd66ffae4a0', 'cpu/development/predictions.jsonl': 'sha256:6f5b4d236d1f8535e34cae27d4ea60066dde529a6cd2fed3b37d97ce4325ea88', 'cpu/development/report.json': 'sha256:4ec924ac3338f212752a36bee7814d623a5753d36de004126b38487f60e18d5a', 'cpu/development-gate.json': 'sha256:5471942d8aea2a01327cf78ace3f024d11b9d06096be0b7bdf9965a6657adb56', 'cpu/matrix.json': 'sha256:a25c8b1a2a0b66bb536c5fb02c7ed2fa359bdb62845b36fd445eb4b1bf21896e', 'cpu/swap-17/attempt.json': 'sha256:292c01a61189fe480459d1618b44a1c219cd88cd5383740236220c5f8655ea42', 'cpu/swap-17/completion.json': 'sha256:aeb1bbbd99a43d4d55e3443659084a64b8dea97eeb79bda9b2c0e9a5feb3181c', 'cpu/swap-17/diagnostic-predictions.jsonl': 'sha256:652b51f2b8e7b68858dfc7c4082301d278b3bfb9854c6fcdcf21ee6d5eafd692', 'cpu/swap-17/report.json': 'sha256:cb0f0af5d2b93728d3a88aca06dbb972a264de80c710897f1bbcd6402e96a4fb', 'cpu/swap-29/attempt.json': 'sha256:9ccb14f0e85aed4b0fbfeb7eb583d02534f29d31f92662a25a008bbb27bd554f', 'cpu/swap-29/completion.json': 'sha256:9cc79d31d7a9d69210f69c2a7ad214e5ef1f7fe74f1fb606fe2413ced4d22e1d', 'cpu/swap-29/diagnostic-predictions.jsonl': 'sha256:dc106535e2c2b5493bf8ffb684ad82282ca760e437064484cd48ccbab9ffd283', 'cpu/swap-29/report.json': 'sha256:00f7679561308532a34f41b62365c3d1e2b8f1a1876001797a96fa6fd2940101', 'device/audit.stdout': 'sha256:a2712ca2eb45f1fc7aea67e54e37263df6be5ad8068bb7f0c3cbf3ef5eec8798', 'device/cpu/attempt.json': 'sha256:b332c08ac7dfe13ca256fc8c2555452110b17e850893f869d01e189d27831ede', 'device/cpu/completion.json': 'sha256:3a90046fd19a6615542d5194515122c6e5e73df9f0774fb0eb575a30ec690144', 'device/cpu/diagnostic-predictions.jsonl': 'sha256:41eb9b8d0dd70c7e852bb45a55c27656df2a42d5dd4f78afa6974a73fe248c58', 'device/cpu/report.json': 'sha256:d46773c26352984fb2b4f164a858a57af5388f42bfb145498a145d65507db037', 'device/cuda/attempt.json': 'sha256:6c223cc64bac1bb64a345e0a16d43014ae440c45a3b2021afe07750fe64f1746', 'device/cuda/completion.json': 'sha256:2839ce6019d4c35687abb82fa68718b059f8989fe95d7169fa80bd60a787c31a', 'device/cuda/diagnostic-predictions.jsonl': 'sha256:ffa403e17ed1eb6826e5d00f508655e243145974c6b7b48e0e1535bc6781f182', 'device/cuda/report.json': 'sha256:473abcd161d879e1c65af29a04892fe43b91cac4c5490a84a5400b80e7764e6a', 'device/development/attempt.json': 'sha256:11e7a0e765522d0dd0542fbba16fd2adaff4234151e175090418d5b9ff63317d', 'device/development/completion.json': 'sha256:e4c5834daa3262c9a1bc72d05440964fbebe7ebc8efb03ff4678290828dbaba8', 'device/development/predictions.jsonl': 'sha256:7f33603c154d8c22c6d7075ec01329bb75775cd37642fe67d3a0bf24ca948e55', 'device/development/report.json': 'sha256:021b03aee77e0356fae4f393ae63f58194a2e151d69f497dfca272c0f01a6f4a', 'device/development-gate.json': 'sha256:24bb11366ff5df48dd0af43e0ac01ee641abc763a13b042812e53f2387f49910', 'device/matrix.json': 'sha256:e6641be61e71d9b1db8919d0e224d95c66bb2b1ec46038d2c8eec340b5c37bb5', 'device/probe/attempt.json': 'sha256:6a41f1845bf97ad7e486939d870e90f956b119ac328088ca0f972fba774a4982', 'device/probe/comparison.json': 'sha256:4da425bb13cef1d5bf758bd68f319b57dd93d656fc891350ced1ccb2bf585c23', 'device/probe/completion.json': 'sha256:bd377f6164dd1ac7cef44a3b8a38ea47362770ad2a5907a1b2024e05c4412aa3', 'device/probe/report.json': 'sha256:5483eebb2086daf76c3780024a9ceaf6f95022b1b9d5574bf7f046d24698734b', 'provenance-DERIVED.json': 'sha256:aa243e5fc0131c589cdbb8b6e8cf868015adb995cec81648554322243218398f', 'source-SHA256SUMS': 'sha256:9178b5006ac3a0ea008fdc0b070f599790df45dcbd983b8c093627de074e0f6f'}
MODULES = {'trinite': 'src/trinite/__init__.py', 'trinite.contracts': 'src/trinite/contracts.py', 'trinite.foundations_plan': 'src/trinite/foundations_plan.py', 'symbolic_music_v1': 'scripts/symbolic_music_v1/__init__.py', 'symbolic_music_v1.core': 'scripts/symbolic_music_v1/core.py', 'symbolic_music_v1.oracle': 'scripts/symbolic_music_v1/oracle.py', 'music_learning_v1': 'scripts/music_learning_v1/__init__.py', 'music_learning_v1.codec': 'scripts/music_learning_v1/codec.py', 'music_learning_v1.numerical': 'scripts/music_learning_v1/numerical.py', 'music_contrast_v1': 'scripts/music_contrast_v1/__init__.py', 'music_contrast_v1.spec': 'scripts/music_contrast_v1/spec.py', 'music_contrast_v1.reporting': 'scripts/music_contrast_v1/reporting.py', 'music_exposure_v1': 'scripts/music_exposure_v1/__init__.py', 'music_exposure_v1.spec': 'scripts/music_exposure_v1/spec.py', 'trinite.tokenizer': 'src/trinite/tokenizer.py', 'music_placement_v1': 'scripts/music_placement_v1/__init__.py', 'music_placement_v1.spec': 'scripts/music_placement_v1/spec.py', 'music_placement_v1.reporting': 'scripts/music_placement_v1/reporting.py', 'music_device_trial_v1': 'scripts/music_device_trial_v1/__init__.py', 'music_device_trial_v1.spec': 'scripts/music_device_trial_v1/spec.py'}
SOURCES = {'src/trinite/__init__.py': 'sha256:029d147c8a741f0cecffc90611d77c6a4047a019863d86ca40992c630a3524b6', 'src/trinite/contracts.py': 'sha256:36562611c6d60b4426a28e0974cda77b81a305186768f69d85bbbaf12a355d24', 'src/trinite/foundations_plan.py': 'sha256:f4a4cee055b97e6aa1217340b1a103cd561caeaca01e6b358e71eaedbab8a8d7', 'scripts/symbolic_music_v1/__init__.py': 'sha256:aea01a9d1bd6ae591665b9fcc3658189091a162565851d4c92579731f82405db', 'scripts/symbolic_music_v1/core.py': 'sha256:9c7446c8d275dbc02fe195eac7a660130cf70a6943fa31428c0d4c7f49bba6ff', 'scripts/symbolic_music_v1/oracle.py': 'sha256:3b6c3e00a2caca2a4d2e12d6ab93fda912197d638cec220175a8073b4ee09af0', 'scripts/music_learning_v1/__init__.py': 'sha256:352fce75f453ef87602ff6b451800c6eb765fa2edd2c79d5428013b6f05b5c9a', 'scripts/music_learning_v1/codec.py': 'sha256:5236fab68f2ff11ff4248d1187ddd0db5797138784665eeca4323ff9c192fa0b', 'scripts/music_learning_v1/numerical.py': 'sha256:214fa9dfd4490cedb7a55489331db1ad550b8b5dfdf5021294a117d470f20813', 'scripts/music_contrast_v1/__init__.py': 'sha256:3d67f788b234aba0de63cfe5368e9a9dac7d8d8220ecfc5120b5ce708694428d', 'scripts/music_contrast_v1/spec.py': 'sha256:34f051fe47feb48baff3ca636730d9348682812e2ffb4bd3e2a4c07b78cb3317', 'scripts/music_contrast_v1/reporting.py': 'sha256:5f15f9231e3b57e32410a8cd2eb7d66f49565fa13661a680c8327d03fba2b846', 'scripts/music_exposure_v1/__init__.py': 'sha256:36ac2634a27e9076d885362c988335f154cd071ae167b6e7c8d1a481df9f5c6d', 'scripts/music_exposure_v1/spec.py': 'sha256:c80b16e68ab85ebdfe06a40e406941b3e19ae5596ca7945babac9df8e35448d9', 'src/trinite/tokenizer.py': 'sha256:fa87f79f667da8cabeeae0e999add9f420102efbfb746a0c0b275b8703d845bb', 'scripts/music_placement_v1/__init__.py': 'sha256:03a38e3bfe0523473faf6de5fba8f80afb3ce561cfadf12631300b3d5a1b562e', 'scripts/music_placement_v1/spec.py': 'sha256:60660075dac0062ff334bac0db577cb539108f136b77a0ad927a00af8d3eef33', 'scripts/music_placement_v1/reporting.py': 'sha256:4bcfd7cddfe9f92cc2133117ec853588834a5895e0b62697c77f20e5655fa3ab', 'scripts/music_device_trial_v1/__init__.py': 'sha256:d8441a81360272a82f039ca80b6e7f44acd752497272ee7e242103e8d1db25ce', 'scripts/music_device_trial_v1/spec.py': 'sha256:4c71390dc5df4b99bd347ab04a251b5fe5a39b5e3d5240f84c640ebd601417c6'}
REFERENCE_SOURCES = {'scripts/music_device_trial_v1/controller.py': 'sha256:aa3de700f821e0aed221b1e1a0c5d6bac7005f41e7f70f038e79b600bf661c23'}


def digest(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def read(path):
    """Bounded no-follow read with ancestor and observed mutation rejection."""
    path = Path(os.path.abspath(path))
    if any(not stat.S_ISDIR(p.lstat().st_mode) for p in path.parents):
        raise ValueError('unsafe ancestor')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > LIMIT:
        raise ValueError('bounded regular file required')
    signature = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if signature(os.fstat(fd)) != signature(before):
            raise ValueError('file replaced at open')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(LIMIT + 1)
        if (len(raw) > LIMIT or signature(os.fstat(fd)) != signature(before)
                or signature(path.lstat()) != signature(before)):
            raise ValueError('file changed during read')
        return raw
    finally:
        os.close(fd)


def sources():
    """Authenticate the entire stdlib-only scorer closure before project imports.

    The interpreter, stdlib and this verifier are trusted. Reject ordinary
    filesystem/package-resolution shadows, not arbitrary runtime monkeypatches.
    """
    for relative, expected in REFERENCE_SOURCES.items():
        if digest(read(ROOT / relative)) != expected:
            raise ValueError('frozen timing reference source changed')
    for name, relative in MODULES.items():
        path = ROOT / relative
        if digest(read(path)) != SOURCES[relative]:
            raise ValueError('frozen scoring source changed')
        package = path.parent
        if (package/'__pycache__').exists():
            raise ValueError('scorer import cache forbidden')
        stem = path.stem
        if path.name != '__init__.py' and (package/stem).exists():
            raise ValueError('scorer package shadow forbidden')
        for member in package.iterdir():
            if member.stem == stem and member.suffix in ('.pyc','.pyo','.so','.pyd'):
                raise ValueError('scorer extension/cache shadow forbidden')
            if member.name.startswith(stem+'.') and member.suffix in ('.so','.pyd'):
                raise ValueError('scorer extension shadow forbidden')
    for name, relative in MODULES.items():
        path = ROOT / relative
        module = sys.modules.get(name)
        if module is not None and Path(getattr(module,'__file__','')).absolute() != path:
            raise ValueError('foreign scoring module')
    # Parent packages are listed before children; all package initializers have
    # been authenticated above before find_spec may execute them.
    for name, relative in MODULES.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin != str(ROOT/relative):
            raise ValueError('foreign scoring import resolution')


def snapshot(fixture):
    """Exact bounded membership; refuse links, special files and extra payloads."""
    fixture = Path(os.path.abspath(fixture))
    if any(not stat.S_ISDIR(p.lstat().st_mode) for p in fixture.parents):
        raise ValueError('unsafe ancestor')
    if not stat.S_ISDIR(fixture.lstat().st_mode):
        raise ValueError('non-link fixture directory required')
    expected_dirs = {p.as_posix() for name in FIXTURE_HASHES
                     for p in Path(name).parents if p != Path('.')}
    found_files, found_dirs = set(), set()
    for current, directories, files in os.walk(fixture, followlinks=False):
        for name in directories:
            p = Path(current) / name
            if not stat.S_ISDIR(p.lstat().st_mode):
                raise ValueError('non-link fixture directory required')
            relative = p.relative_to(fixture).as_posix()
            if relative not in expected_dirs:
                raise ValueError('fixture directory membership differs')
            found_dirs.add(relative)
        for name in files:
            found_files.add((Path(current) / name).relative_to(fixture).as_posix())
            if len(found_files) > len(FIXTURE_HASHES):
                raise ValueError('fixture membership differs')
    if found_files != set(FIXTURE_HASHES) or found_dirs != expected_dirs:
        raise ValueError('fixture membership differs')
    return {name: read(fixture / name) for name in FIXTURE_HASHES}

CELLS = ('control-17', 'swap-17', 'swap-29', 'control-29')
BACKENDS = ('cpu', 'cuda')
REQUEST_IDS = {'cpu':'sha256:df7706a21ca8bc5f2840cae6b4c239c6a82b8c0755045aeda38dbf4499bf4eeb',
               'device':'sha256:28220b74d1b0cb11e6b2cff48d784c0e687402c094c6917f32d0d24a4685c2a2'}
MATRIX_IDS = {'cpu':'sha256:a25c8b1a2a0b66bb536c5fb02c7ed2fa359bdb62845b36fd445eb4b1bf21896e',
              'device':'sha256:e6641be61e71d9b1db8919d0e224d95c66bb2b1ec46038d2c8eec340b5c37bb5'}
CLAIMS = {'fresh_confirmation':False,'learning_adequate':False,'phase5_ready':False,'release_ready':False}
SCOPE = 'source-bound bytes, safe tensor/progress structure and retained scores; no inference or optimizer replay'


def finite(value, maximum=None):
    number = float.fromhex(value)
    if (not math.isfinite(number) or number < 0 or number.hex() != value
            or (maximum is not None and number > maximum)):
        raise ValueError('invalid retained duration/error')
    return number


def score(before):
    """Recompute retained tokens and available links, not tensors or execution."""
    sources()
    from trinite.contracts import parse_json, json_bytes, exact_keys
    from music_placement_v1.reporting import matrix_summary as cpu_summary, score_cell as cpu_cell
    from music_device_trial_v1.spec import matrix_summary as device_summary, score_cell as device_cell
    from music_device_trial_v1.spec import TIMING_PHASES, PARITY_TOLERANCES
    sources()
    def parse(name): return parse_json(before[name], canonical=True)
    def raw_id(name): return digest(before[name])
    def same(a, b, label):
        if json_bytes(a) != json_bytes(b): raise ValueError(label+' differs')
    def records(name):
        return [parse_json(line,canonical=True) for line in before[name].splitlines(keepends=True)]
    summaries, inventories = {}, {}
    for lane in ('cpu','device'):
        prefix=lane+'/'
        cells=CELLS if lane=='cpu' else BACKENDS
        protocol='placement' if lane=='cpu' else 'device'
        request_id=REQUEST_IDS[lane]
        matrix=parse(prefix+'matrix.json');gate=parse(prefix+'development-gate.json');audit=parse(prefix+'audit.stdout')
        matrix_keys={'schema','request_identity','outcome','cells','development','gate_identity','failure','summary','verification_scope','files',*CLAIMS}
        if lane=='device':matrix_keys.update(('probe','parity','timing_comparison','controller_wall_seconds_hex','timing_scope'))
        exact_keys(matrix,matrix_keys,'matrix')
        if (raw_id(prefix+'matrix.json')!=MATRIX_IDS[lane] or matrix['request_identity']!=request_id
                or matrix['schema']!='trinite.music-'+protocol+'-matrix.v1' or matrix['outcome']!='completed'
                or matrix['failure'] is not None or matrix['verification_scope']!=SCOPE):
            raise ValueError('original matrix context differs')
        same({k:matrix[k] for k in CLAIMS},CLAIMS,'matrix claims')
        exact_keys(matrix['cells'],set(cells),'fixed cells')
        present={name[len(prefix):]:digest(raw) for name,raw in before.items() if name.startswith(prefix)
                 and name not in (prefix+'matrix.json',prefix+'audit.stdout')}
        omitted={cell+'/'+name for cell in cells for name in ('diagnostic-predictions.json','final-checkpoint.json','final.safetensors','request.json','worker.log')}
        omitted.update(('request.json','development/worker.log'))
        if lane=='device':omitted.update(('probe/probe.safetensors','probe/worker.log'))
        same(set_as_list(matrix['files']),sorted(set(present)|omitted),'matrix member inventory')
        same({name:matrix['files'][name] for name in present},present,'available matrix hashes')
        inventories[lane]={'present':sorted(present),'omitted':sorted(omitted)}
        diagnostic_records=[];initializations={}
        for cell in cells:
            child=prefix+cell+'/'
            journal=records(child+'diagnostic-predictions.jsonl');diagnostic_records.extend(journal)
            if lane=='cpu':
                arm,seed=cell.split('-');seed=int(seed);summary=cpu_cell(journal,arm,seed,'diagnostic')
            else:seed=17;summary=device_cell(journal,cell,'diagnostic')
            report=parse(child+'report.json')
            common={'schema','request_identity','admission_identity','step','target_tokens','initialization_identity','model_identity',
                    'update_seconds_hex','diagnostic_decodes','diagnostic_forward_calls','diagnostics','verification_scope',*CLAIMS}
            exact_keys(report,common|({'arm','seed','cell','diagnostic_summary'} if lane=='cpu' else {'backend','seed','timing_hex','diagnostic_greedy_seconds_hex'}),'cell report')
            same(report['diagnostics'],summary,'diagnostic score')
            if lane=='cpu':
                same(report['diagnostic_summary'],summary,'native diagnostic score')
                same({k:report[k] for k in ('arm','seed','cell')},{'arm':arm,'seed':seed,'cell':cell},'native cell')
            else:
                same({k:report[k] for k in ('backend','seed')},{'backend':cell,'seed':17},'device cell')
                exact_keys(report['timing_hex'],set(TIMING_PHASES),'phase timing')
                for value in report['timing_hex'].values():finite(value)
                if (report['timing_hex']['updates']!=report['update_seconds_hex']
                        or any(report['timing_hex'][k]!=(0.).hex() for k in ('compile','target_warmup'))
                        or finite(report['diagnostic_greedy_seconds_hex'])>finite(report['timing_hex']['diagnostics'])):
                    raise ValueError('timing phase boundary differs')
            same({k:report[k] for k in CLAIMS},CLAIMS,'cell claims')
            same({k:report[k] for k in ('schema','request_identity','step','target_tokens','diagnostic_decodes','diagnostic_forward_calls','verification_scope')},
                 {'schema':'trinite.music-'+protocol+'-cell.v1','request_identity':request_id,'step':640,'target_tokens':10240,
                  'diagnostic_decodes':80,'diagnostic_forward_calls':sum(len(r['generated_tokens']) for r in journal),'verification_scope':SCOPE},'cell context/accounting')
            finite(report['update_seconds_hex'],120);initializations[cell]=report['initialization_identity']
            completion={'schema':'trinite.music-'+protocol+'-completion.v1','request_identity':request_id,
                        ('cell' if lane=='cpu' else 'backend'):cell,'artifacts':{n:matrix['files'][cell+'/'+n] for n in
                         ('diagnostic-predictions.json','diagnostic-predictions.jsonl','final-checkpoint.json','final.safetensors','report.json')}}
            same(parse(child+'completion.json'),completion,'cell completion')
        if lane=='cpu':
            for seed in (17,29):
                if initializations['control-'+str(seed)]!=initializations['swap-'+str(seed)]:raise ValueError('paired initialization differs')
        elif initializations['cpu']!=initializations['cuda']:raise ValueError('backend initialization differs')
        expected_gate={'schema':'trinite.music-'+protocol+'-development-gate.v1','request_identity':request_id,
                       'cells':{cell:raw_id(prefix+cell+'/completion.json') for cell in cells},
                       'selection':'all four safe completed final640 cells; no training-fit gate' if lane=='cpu' else 'both safe final640 cells; no accuracy fit gate'}
        if lane=='device':expected_gate['probe_completion_identity']=raw_id(prefix+'probe/completion.json')
        same(gate,expected_gate,'gate');same(raw_id(prefix+'development-gate.json'),matrix['gate_identity'],'gate link')
        for cell in (*cells,'development',*(('probe',) if lane=='device' else ())):
            child=prefix+cell+'/'
            record=matrix[cell] if cell in ('development','probe') else matrix['cells'][cell]
            same(record,{'outcome':'completed','completion_identity':raw_id(child+'completion.json'),'attempt_identity':raw_id(child+'attempt.json'),
                 'files':{n[len(cell)+1:]:h for n,h in matrix['files'].items() if n.startswith(cell+'/')}},'attempt links')
            attempt=parse(child+'attempt.json')
            fields={'schema','action','cell','request_identity','outcome','worker_exit','completion_identity','detail','wall_seconds_hex'}
            if lane=='device':fields.update(('worker_wall_seconds_hex','postworker_admission_seconds_hex'))
            exact_keys(attempt,fields,'attempt')
            expected={'schema':'trinite.music-'+protocol+'-attempt.v1','action':'_'+cell if cell in ('development','probe') else '_train',
                      'cell':cell,'request_identity':request_id,'outcome':'completed','worker_exit':0,'completion_identity':record['completion_identity']}
            same({k:attempt[k] for k in expected},expected,'attempt context')
            if type(attempt['detail']) is not str:raise ValueError('attempt detail differs')
            seconds=finite(attempt['wall_seconds_hex'])
            if lane=='device' and abs(seconds-finite(attempt['worker_wall_seconds_hex'])-finite(attempt['postworker_admission_seconds_hex']))>1e-8:
                raise ValueError('attempt timing partition differs')
        development_records=records(prefix+'development/predictions.jsonl')
        summary=(cpu_summary if lane=='cpu' else device_summary)(diagnostic_records,development_records)
        same(matrix['summary'],summary,'matrix score')
        development=parse(prefix+'development/report.json')
        expected={'schema':'trinite.music-'+protocol+'-development.v1','request_identity':request_id,'gate_identity':matrix['gate_identity'],
                  'summary':summary,'generation':{'decodes':480 if lane=='cpu' else 240,
                  'forward_calls':sum(len(r['generated_tokens']) for r in development_records),'model_identities_unchanged':True,'training_performed':False},
                  'input_files_unchanged':True,'model_states_unchanged':True,'verification_scope':SCOPE,**CLAIMS}
        if lane=='device':
            for field in ('timing_hex','greedy_seconds_hex'):
                exact_keys(development[field],set(BACKENDS),'development timing')
                expected[field]=development[field]
                for value in development[field].values():finite(value)
            for b in BACKENDS:
                if finite(development['greedy_seconds_hex'][b])>finite(development['timing_hex'][b]):raise ValueError('nested greedy exceeds pipeline')
        same(development,expected,'development report')
        same(parse(prefix+'development/completion.json'),{'schema':'trinite.music-'+protocol+'-development-completion.v1',
              'request_identity':request_id,'gate_identity':matrix['gate_identity'],
              'artifacts':{n:raw_id(prefix+'development/'+n) for n in ('predictions.jsonl','report.json')}},'development completion')
        expected_audit={'verified':True,'outcome':'completed','matrix_identity':MATRIX_IDS[lane],'scope':SCOPE,
                        'inference_replayed':False,'optimizer_replayed':False,**CLAIMS}
        if lane=='device':
            finite(audit['audit_wall_seconds_hex']);expected_audit['audit_wall_seconds_hex']=audit['audit_wall_seconds_hex']
            finite(matrix['controller_wall_seconds_hex'])
            if matrix['timing_scope']!='controller run including workers, admission, serialization and inline audits; excludes Python controller startup and later standalone audit':
                raise ValueError('controller timing scope differs')
            check_probe(parse,raw_id,matrix,PARITY_TOLERANCES,initializations,same,exact_keys)
            same(matrix['timing_comparison'],timing_summary(parse),'retained timing comparison')
        same(audit,expected_audit,'supplied audit receipt')
        summaries[lane]=summary
    manifest={}
    for line in before['source-SHA256SUMS'].decode('ascii').splitlines():
        checksum,name=line.split('  ',1)
        if name in manifest:raise ValueError('duplicate manifest member')
        manifest[name]='sha256:'+checksum
    for name in ORIGINAL_HASHES:
        if name!='source-SHA256SUMS' and manifest.get(name)!=digest(before[name]):raise ValueError('original manifest link differs')
    provenance=parse('provenance-DERIVED.json')
    check_provenance(provenance,inventories,same)
    sources()
    return summaries


def set_as_list(value):
    return sorted(value)


def timing_summary(parse):
    """Same retained-clock arithmetic as pinned v1 controller; no runtime import."""
    result={'by_backend':{},'scope':'fixed CPU then CUDA pair; cold process wall includes spawn/import/runtime/context and output handling; phase clocks synchronize CUDA; no significance claim'}
    for backend in BACKENDS:
        report=parse('device/'+backend+'/report.json');attempt=parse('device/'+backend+'/attempt.json')
        worker=finite(attempt['worker_wall_seconds_hex']);phases=sum(finite(v) for v in report['timing_hex'].values())
        if phases>worker+1e-6:raise ValueError('named phases exceed worker wall')
        result['by_backend'][backend]={'worker_wall_seconds_hex':attempt['worker_wall_seconds_hex'],
            'postworker_admission_seconds_hex':attempt['postworker_admission_seconds_hex'],'phase_seconds_hex':report['timing_hex'],
            'diagnostic_greedy_seconds_hex_nested':report['diagnostic_greedy_seconds_hex'],
            'unattributed_startup_admission_handling_seconds_hex':max(0.,worker-phases).hex()}
    cpu=finite(result['by_backend']['cpu']['worker_wall_seconds_hex']);cuda=finite(result['by_backend']['cuda']['worker_wall_seconds_hex'])
    result['cuda_over_cpu_worker_wall_ratio_hex']=(cuda/cpu).hex() if cpu>0 else None
    result['overhead_attempt_seconds_hex']={n:parse('device/'+n+'/attempt.json')['wall_seconds_hex'] for n in ('probe','development')}
    result['development_decode_seconds_hex']=parse('device/development/report.json')['timing_hex']
    result['development_greedy_seconds_hex_nested']=parse('device/development/report.json')['greedy_seconds_hex']
    return result


def check_probe(parse,raw_id,matrix,tolerances,initializations,same,exact_keys):
    report=parse('device/probe/report.json');comparison=parse('device/probe/comparison.json');completion=parse('device/probe/completion.json')
    same(completion,{'schema':'trinite.music-device-probe-completion.v1','request_identity':REQUEST_IDS['device'],
         'artifacts':{n:matrix['files']['probe/'+n] for n in ('probe.safetensors','report.json','comparison.json')}},'probe completion')
    exact_keys(report,{'schema','budget','by_backend','payload_identity','probe_input_identity','tolerances'},'probe report')
    same(report['budget'],{'forward_calls':2,'target_tokens':32,'updates':2,'visits':16},'probe budget')
    same(report['tolerances'],tolerances,'frozen probe tolerance')
    exact_keys(report['by_backend'],set(BACKENDS),'probe backends')
    for backend in BACKENDS:
        value=report['by_backend'][backend];finite(value['seconds_hex'])
        same(value,{'backend':backend,'forward_calls':1,'forward_shape':[8,67,259],'initialization_identity':initializations[backend],
             'seconds_hex':value['seconds_hex'],'target_tokens':16,'updates':1,'visits':8},'probe backend')
    if report['schema']!='trinite.music-device-probe.v1' or report['payload_identity']!=completion['artifacts']['probe.safetensors']:
        raise ValueError('probe payload link differs')
    exact_keys(comparison,{'schema','scope','passed','initial_bytes_equal','metrics','probe_payload_identity','probe_report_identity',
                          'final_training_equality_required','inference_replayed','optimizer_replayed'},'probe comparison')
    if (comparison['schema']!='trinite.music-device-parity.v1' or comparison['passed'] is not True or comparison['initial_bytes_equal'] is not True
            or comparison['probe_payload_identity']!=report['payload_identity'] or comparison['probe_report_identity']!=raw_id('device/probe/report.json')
            or any(comparison[k] is not False for k in ('final_training_equality_required','inference_replayed','optimizer_replayed'))
            or comparison['scope']!='retained finite initial/one-step tensor comparison; not actual execution, timing or hardware attestation'):
        raise ValueError('probe reported context differs')
    params=('blocks.0.attention_norm.weight','blocks.0.down.weight','blocks.0.feed_forward_norm.weight','blocks.0.k.weight','blocks.0.o.weight',
            'blocks.0.q.weight','blocks.0.up.weight','blocks.0.v.weight','final_norm.weight','position_embedding','token_embedding')
    expected={kind+'.'+p for kind in ('exp_avg','exp_avg_sq','gradient','parameter') for p in params}|{'gradient_norm','logits','loss'}
    exact_keys(comparison['metrics'],expected,'47 probe metrics')
    for name,metric in comparison['metrics'].items():
        exact_keys(metric,{'atol_hex','rtol_hex','max_absolute_error_hex','max_reference_magnitude_hex','passed'},'probe metric')
        kind=name.split('.')[0];same({k:metric[k] for k in ('atol_hex','rtol_hex')},tolerances[kind],'probe metric tolerance')
        error=finite(metric['max_absolute_error_hex']);magnitude=finite(metric['max_reference_magnitude_hex'])
        # Necessary summary consistency only. Missing per-element tensors prevent
        # independently proving the original elementwise allclose result.
        if metric['passed'] is not True or error>finite(metric['atol_hex'])+finite(metric['rtol_hex'])*magnitude:
            raise ValueError('probe recorded error/tolerance inconsistent')
    same(matrix['parity'],comparison,'matrix probe comparison')


def check_provenance(provenance,inventories,same):
    expected_keys={'schema','classification','producer_commit','producer_tree','source_archive_sha256',
                   'included_original_files','omitted_handoff_files','lanes','actual_execution_attested','full_run_custody',
                   'inference_replayed','optimizer_replayed','fresh_confirmation','hardware_attested','timing_attested',
                   'probe_tensor_comparison_replayed'}
    if type(provenance) is not dict or set(provenance)!=expected_keys:
        raise ValueError('derived provenance fields differ')
    same({k:provenance[k] for k in ('schema','classification','producer_commit','producer_tree','source_archive_sha256')},
         {'schema':'trinite.music-placement-result-provenance.v1','classification':'DERIVED reports-only provenance; not a producer receipt',
          'producer_commit':'798cf05a1ade30b4877177f251f71435913293a8','producer_tree':'fa8cce93b08f158f37e0f9b5e52a5113208ef19b',
          'source_archive_sha256':'23a20e25510f7a2c7c06aa289103b52c315118d5d53cf3b2da3b9428cd80931d'},'derived source provenance')
    same(provenance['included_original_files'],sorted(n for n in ORIGINAL_HASHES if n!='source-SHA256SUMS'),'derived original inventory')
    same(provenance['omitted_handoff_files'],['HANDOFF.md','INVENTORY.txt','REPORT.md'],'omitted handoff inventory')
    same(provenance['lanes'],{lane:{'original_matrix_identity':MATRIX_IDS[lane],
         'original_request_identity_owner_receipt':REQUEST_IDS[lane],'matrix_bound_present':inventories[lane]['present'],
         'matrix_bound_omitted':inventories[lane]['omitted']} for lane in ('cpu','device')},'derived lane inventory')
    for key in ('actual_execution_attested','full_run_custody','inference_replayed','optimizer_replayed','fresh_confirmation',
                'hardware_attested','timing_attested','probe_tensor_comparison_replayed'):
        if provenance[key] is not False:raise ValueError('derived verification scope differs')


def verify(fixture=FIXTURE):
    before=snapshot(fixture)
    if {name:digest(raw) for name,raw in before.items()}!=FIXTURE_HASHES:
        raise ValueError('preserved fixture bytes changed')
    summary=score(before)
    if snapshot(fixture)!=before:raise ValueError('fixture changed during verification')
    sources()
    return {'schema':'trinite.music-placement-retained-check.v1','verified':True,
        'scope':'supplied byte integrity, available links, retained-token scores and reported timing arithmetic only',
        'execution_outcome_in_supplied_report':{'cpu':'completed','device':'completed'},
        'interpretation':'mixed/negative development; one descriptive same-wheel CPU/CUDA pair',
        'decodes_in_journal':summary['cpu']['decodes']+summary['device']['decodes'],
        'generated_tokens':summary['cpu']['decode_forwards']+summary['device']['generated_tokens'],
        'available_matrix_bound_files':{'cpu':21,'device':17},'omitted_matrix_bound_files':{'cpu':22,'device':14},
        'development_structural_parents':2,
        'canonical_exact_by_cell':{c:summary['cpu']['development']['by_cell'][c]['canonical_primary']['exact_plus_eos'] for c in CELLS},
        'device_canonical_exact_by_backend':{b:summary['device']['development'][b]['canonical_primary']['exact_plus_eos'] for b in BACKENDS},
        'device_raw_equal_development':sum(v['identical_raw_outputs'] for v in summary['device']['raw_backend_pairs']['development'].values()),
        'recorded_probe_checks':47,'timing_comparison':json.loads(before['device/matrix.json'])['timing_comparison'],
        'original_raw_custody_verified':False,'actual_execution_attested':False,'hardware_attested':False,'timing_attested':False,
        'inference_replayed':False,'optimizer_replayed':False,'probe_tensor_comparison_replayed':False,'fresh_confirmation':False,
        'learning_adequate':False,'release_ready':False}


def main():
    try:
        if len(sys.argv)!=1:raise ValueError('this read-only checker takes no arguments')
        print(json.dumps(verify(),sort_keys=True))
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(str(error),file=sys.stderr);return 1
    return 0
