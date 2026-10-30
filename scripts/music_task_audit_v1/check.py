"""Immutable-source, retained-output task audit. No model execution or hardware diagnosis."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
ROOT = Path(__file__).resolve().parents[2]
LIMIT = 4 * 1024 * 1024
MODULES = {'frozen_music_eval_result_v1': 'scripts/frozen_music_eval_result_v1/__init__.py',
 'frozen_music_eval_v1': 'scripts/frozen_music_eval_v1/__init__.py',
 'music_contrast_v1': 'scripts/music_contrast_v1/__init__.py',
 'music_device_trial_v1': 'scripts/music_device_trial_v1/__init__.py',
 'music_exposure_result_v1': 'scripts/music_exposure_result_v1/__init__.py',
 'music_exposure_v1': 'scripts/music_exposure_v1/__init__.py',
 'music_learning_v1': 'scripts/music_learning_v1/__init__.py',
 'music_pitch_intervention_v1': 'scripts/music_pitch_intervention_v1/__init__.py',
 'music_pitch_result_v1': 'scripts/music_pitch_result_v1/__init__.py',
 'music_placement_result_v1': 'scripts/music_placement_result_v1/__init__.py',
 'music_placement_v1': 'scripts/music_placement_v1/__init__.py',
 'music_robustness_result_v1': 'scripts/music_robustness_result_v1/__init__.py',
 'music_task_audit_v1': 'scripts/music_task_audit_v1/__init__.py',
 'symbolic_music_v1': 'scripts/symbolic_music_v1/__init__.py',
 'trinite': 'src/trinite/__init__.py',
 'frozen_music_eval_result_v1.check': 'scripts/frozen_music_eval_result_v1/check.py',
 'frozen_music_eval_v1.artifacts': 'scripts/frozen_music_eval_v1/artifacts.py',
 'frozen_music_eval_v1.protocol': 'scripts/frozen_music_eval_v1/protocol.py',
 'frozen_music_eval_v1.receipt': 'scripts/frozen_music_eval_v1/receipt.py',
 'frozen_music_eval_v1.workflow': 'scripts/frozen_music_eval_v1/workflow.py',
 'music_contrast_v1.reporting': 'scripts/music_contrast_v1/reporting.py',
 'music_contrast_v1.spec': 'scripts/music_contrast_v1/spec.py',
 'music_device_trial_v1.spec': 'scripts/music_device_trial_v1/spec.py',
 'music_exposure_result_v1.check': 'scripts/music_exposure_result_v1/check.py',
 'music_exposure_v1.reporting': 'scripts/music_exposure_v1/reporting.py',
 'music_exposure_v1.spec': 'scripts/music_exposure_v1/spec.py',
 'music_learning_v1.codec': 'scripts/music_learning_v1/codec.py',
 'music_learning_v1.numerical': 'scripts/music_learning_v1/numerical.py',
 'music_pitch_intervention_v1.bootstrap': 'scripts/music_pitch_intervention_v1/bootstrap.py',
 'music_pitch_intervention_v1.reporting': 'scripts/music_pitch_intervention_v1/reporting.py',
 'music_pitch_intervention_v1.spec': 'scripts/music_pitch_intervention_v1/spec.py',
 'music_pitch_result_v1.check': 'scripts/music_pitch_result_v1/check.py',
 'music_placement_result_v1.check': 'scripts/music_placement_result_v1/check.py',
 'music_placement_v1.reporting': 'scripts/music_placement_v1/reporting.py',
 'music_placement_v1.spec': 'scripts/music_placement_v1/spec.py',
 'music_robustness_result_v1.check': 'scripts/music_robustness_result_v1/check.py',
 'music_task_audit_v1.analysis': 'scripts/music_task_audit_v1/analysis.py',
 'symbolic_music_v1.core': 'scripts/symbolic_music_v1/core.py',
 'symbolic_music_v1.oracle': 'scripts/symbolic_music_v1/oracle.py',
 'trinite.contracts': 'src/trinite/contracts.py',
 'trinite.foundations_plan': 'src/trinite/foundations_plan.py',
 'trinite.tokenizer': 'src/trinite/tokenizer.py'}
SOURCES = {'src/trinite/__init__.py': 'sha256:029d147c8a741f0cecffc90611d77c6a4047a019863d86ca40992c630a3524b6',
 'src/trinite/contracts.py': 'sha256:36562611c6d60b4426a28e0974cda77b81a305186768f69d85bbbaf12a355d24',
 'src/trinite/tokenizer.py': 'sha256:fa87f79f667da8cabeeae0e999add9f420102efbfb746a0c0b275b8703d845bb',
 'scripts/symbolic_music_v1/__init__.py': 'sha256:aea01a9d1bd6ae591665b9fcc3658189091a162565851d4c92579731f82405db',
 'scripts/symbolic_music_v1/core.py': 'sha256:9c7446c8d275dbc02fe195eac7a660130cf70a6943fa31428c0d4c7f49bba6ff',
 'scripts/symbolic_music_v1/oracle.py': 'sha256:3b6c3e00a2caca2a4d2e12d6ab93fda912197d638cec220175a8073b4ee09af0',
 'scripts/music_learning_v1/__init__.py': 'sha256:352fce75f453ef87602ff6b451800c6eb765fa2edd2c79d5428013b6f05b5c9a',
 'scripts/music_learning_v1/codec.py': 'sha256:5236fab68f2ff11ff4248d1187ddd0db5797138784665eeca4323ff9c192fa0b',
 'scripts/music_contrast_v1/__init__.py': 'sha256:3d67f788b234aba0de63cfe5368e9a9dac7d8d8220ecfc5120b5ce708694428d',
 'scripts/music_contrast_v1/spec.py': 'sha256:34f051fe47feb48baff3ca636730d9348682812e2ffb4bd3e2a4c07b78cb3317',
 'scripts/music_contrast_v1/reporting.py': 'sha256:5f15f9231e3b57e32410a8cd2eb7d66f49565fa13661a680c8327d03fba2b846',
 'src/trinite/foundations_plan.py': 'sha256:f4a4cee055b97e6aa1217340b1a103cd561caeaca01e6b358e71eaedbab8a8d7',
 'scripts/music_learning_v1/numerical.py': 'sha256:214fa9dfd4490cedb7a55489331db1ad550b8b5dfdf5021294a117d470f20813',
 'scripts/music_exposure_v1/__init__.py': 'sha256:36ac2634a27e9076d885362c988335f154cd071ae167b6e7c8d1a481df9f5c6d',
 'scripts/music_exposure_v1/spec.py': 'sha256:c80b16e68ab85ebdfe06a40e406941b3e19ae5596ca7945babac9df8e35448d9',
 'scripts/music_exposure_v1/reporting.py': 'sha256:09c0783716ee95c8b0e7cdfbbcfa748e428ae76980ac23f7bd75df8f25763e74',
 'scripts/music_exposure_result_v1/__init__.py': 'sha256:31b00166faa84fc158cdaf679b2056a7ecbff8b7fc118921cd34a713b8bf46da',
 'scripts/music_exposure_result_v1/check.py': 'sha256:d024c7a0c3559cec069bbe5fbe992da5f3a167467e513bec890dc14a7a56e90f',
 'scripts/frozen_music_eval_v1/__init__.py': 'sha256:24e16706fad053362bd8270c99db1d6220c11560cb61813290956cd23efb9f20',
 'scripts/frozen_music_eval_v1/protocol.py': 'sha256:919f5b6a083b437a340b3263d83c98692f070e633a29408bc1ab88e695e321cd',
 'scripts/frozen_music_eval_v1/artifacts.py': 'sha256:9d8da0e97e48d94a5372daf250c3ba2ecd46dc81ad783e3c3d847c39446ed462',
 'scripts/frozen_music_eval_v1/receipt.py': 'sha256:97f37922eedebd6f1323f03c0123187734b2379b447dac783e155574847cb29b',
 'scripts/frozen_music_eval_v1/workflow.py': 'sha256:c7b05d44b0c6815dc586059bd8004da6273887527d4d592849f068b291233f5d',
 'scripts/frozen_music_eval_result_v1/__init__.py': 'sha256:2c5ea225755009531b69fe12abbb6ebf697f712e14e0d7e88146d842586b27d8',
 'scripts/frozen_music_eval_result_v1/check.py': 'sha256:045c5ca52aa654d91b879eb6b014dd090b09f331e670f000dd5fc3c95a4d324e',
 'scripts/music_pitch_intervention_v1/__init__.py': 'sha256:83debd77b5034b913e86fa01026b14109ea58c90bf2ad9153ffae2ebe711a98d',
 'scripts/music_pitch_intervention_v1/spec.py': 'sha256:c6228050f2f24f47404b7bae747b6578f81d5c368a9db419accaabfd69529855',
 'scripts/music_pitch_intervention_v1/bootstrap.py': 'sha256:e515872f9d841feb9218043f561ff1a8216172b6d26f1fdb615b8506eb2d9c4e',
 'scripts/music_pitch_intervention_v1/reporting.py': 'sha256:a2c7b95a5b5cfbaac7da0a912af6d83ceb1a714eeb0c466c7a9bb0dda23c4bf9',
 'scripts/music_pitch_result_v1/__init__.py': 'sha256:89f6c82e37886da454a6b25300afeac7ce7818f66cca75a0ac758fa1f6a4a622',
 'scripts/music_pitch_result_v1/check.py': 'sha256:f0dbffb32e03e60f52a941c306b2e3d40b5ee0bab9513903a10905a8427c72f0',
 'scripts/music_placement_v1/__init__.py': 'sha256:03a38e3bfe0523473faf6de5fba8f80afb3ce561cfadf12631300b3d5a1b562e',
 'scripts/music_placement_v1/spec.py': 'sha256:60660075dac0062ff334bac0db577cb539108f136b77a0ad927a00af8d3eef33',
 'scripts/music_placement_v1/reporting.py': 'sha256:4bcfd7cddfe9f92cc2133117ec853588834a5895e0b62697c77f20e5655fa3ab',
 'scripts/music_device_trial_v1/__init__.py': 'sha256:d8441a81360272a82f039ca80b6e7f44acd752497272ee7e242103e8d1db25ce',
 'scripts/music_device_trial_v1/spec.py': 'sha256:4c71390dc5df4b99bd347ab04a251b5fe5a39b5e3d5240f84c640ebd601417c6',
 'scripts/music_device_trial_v1/controller.py': 'sha256:aa3de700f821e0aed221b1e1a0c5d6bac7005f41e7f70f038e79b600bf661c23',
 'scripts/music_placement_result_v1/__init__.py': 'sha256:df7e3e281f2ff4bea9478e69a20c50eaae266b86c15aef33cc75c64b34a4aa10',
 'scripts/music_placement_result_v1/check.py': 'sha256:c3847d2364bfcb11a4412e35c467cf4499bf085a156b224e34c90d2d353929cf',
 'scripts/music_placement_v2/controller.py': 'sha256:653778295d54fa210411b1e16676e503da6f80d3542cd55ba153a244c11513e1',
 'scripts/music_placement_v2/workflow.py': 'sha256:82cf0b9512b23703aa4d581ed0b29cc301478aec903491dc35347a4cbbc56b20',
 'scripts/music_placement_v2/reporting.py': 'sha256:14fcccc74c5c6af5c42c0e5a9e46c1212b9fd31ddef58722e205d420f6e78787',
 'scripts/music_placement_v2/spec.py': 'sha256:fe9af81a1bc511f6f7fdb093c650acb221b7d229c3b69bfcb802a5e55293f30e',
 'scripts/music_device_trial_v2/controller.py': 'sha256:4c879da4b99320a3b899ec28ba2e36457a569f662fcfa5579153b45ca56f930b',
 'scripts/music_device_trial_v2/runtime.py': 'sha256:07cf257954b7c0ff99728264e9b635c45815ad3941b73d1a3621123715884ae8',
 'scripts/music_robustness_result_v1/__init__.py': 'sha256:9f4e6cce8bec8386336d5869c2e6e67ab13f161bc597e3c410fc09498e059424',
 'scripts/music_robustness_result_v1/check.py': 'sha256:3ea9f82dd775bb5778ea3a2fda5b4b57aa908cb87bf0849a6914476bfefa461b',
 'scripts/music_task_audit_v1/__init__.py': 'sha256:a446367b088afc819f3936f0721abc3dbc250e2605c8253598bd80f9d4d3cbf6',
 'scripts/music_task_audit_v1/analysis.py': 'sha256:9f81df4dadd97b1e55302f7afd2926ccce069895570d1547b676b36f14a52b4a'}
INPUT_HASHES = {'fixtures/music-contrast-result-20261028-v1/train-17--step-000-journal.jsonl': 'sha256:354497c03b6e5a6e2f498b0e5cf0b1a5c101908cb4b65b7fc2fdd8908ccbbd8d',
 'fixtures/music-contrast-result-20261028-v1/train-17--step-000-predictions.json': 'sha256:1d90af54fe7d157ce3d329dac2dd131075c1e0c99064de8072979f73e711cd7a',
 'fixtures/music-contrast-result-20261028-v1/train-17--step-064-journal.jsonl': 'sha256:ab2b1f2444fcb17bb45fa6eff04db1728cb911af76e02de00665fca765b82587',
 'fixtures/music-contrast-result-20261028-v1/train-17--step-064-predictions.json': 'sha256:62fa9ed6f1bf277fbc4c02d2130168b45250070b4d80629531d3d49fb21fa7b1',
 'fixtures/music-contrast-result-20261028-v1/train-29--step-000-journal.jsonl': 'sha256:e3a0527b4b3fe6ef8f89118af06ad61b97cc88e435c14fca181bcf4eb2a6b00a',
 'fixtures/music-contrast-result-20261028-v1/train-29--step-000-predictions.json': 'sha256:e07352a75364b6070660e0e8841459d7a1be61c456e1d8bf95a9f5154b82a08a',
 'fixtures/music-contrast-result-20261028-v1/train-29--step-064-journal.jsonl': 'sha256:ab2b1f2444fcb17bb45fa6eff04db1728cb911af76e02de00665fca765b82587',
 'fixtures/music-contrast-result-20261028-v1/train-29--step-064-predictions.json': 'sha256:62fa9ed6f1bf277fbc4c02d2130168b45250070b4d80629531d3d49fb21fa7b1',
 'fixtures/music-exposure-result-20261029-v1/README.md': 'sha256:671d04f0e70e823cf56feb3e5f8adb2a503fad56a263375663b10f46ac0fae5f',
 'fixtures/music-exposure-result-20261029-v1/provenance.json': 'sha256:53558b164ffa7a8e77ae2d297890b3743dd238beefc0bbadc41e992ccd5c9e5b',
 'fixtures/music-exposure-result-20261029-v1/summary.json': 'sha256:18cfff95b2680ed1b14601aa7984c3f192e75aae5762dac582cf8d49c0ce78c2',
 'fixtures/music-exposure-result-20261029-v1/train-17--step-000-journal.jsonl': 'sha256:354497c03b6e5a6e2f498b0e5cf0b1a5c101908cb4b65b7fc2fdd8908ccbbd8d',
 'fixtures/music-exposure-result-20261029-v1/train-17--step-000-predictions.json': 'sha256:1d90af54fe7d157ce3d329dac2dd131075c1e0c99064de8072979f73e711cd7a',
 'fixtures/music-exposure-result-20261029-v1/train-17--step-064-journal.jsonl': 'sha256:ab2b1f2444fcb17bb45fa6eff04db1728cb911af76e02de00665fca765b82587',
 'fixtures/music-exposure-result-20261029-v1/train-17--step-064-predictions.json': 'sha256:62fa9ed6f1bf277fbc4c02d2130168b45250070b4d80629531d3d49fb21fa7b1',
 'fixtures/music-exposure-result-20261029-v1/train-17--step-640-journal.jsonl': 'sha256:4a1a05c03980040af21cc4f4c23459aedac0b282a0f17c9bc6bd25d62c31d5df',
 'fixtures/music-exposure-result-20261029-v1/train-17--step-640-predictions.json': 'sha256:503c0f780351647c1a127c7cd4516d6f7f7309853cfa69b9c2a8d7c57f9a35c1',
 'fixtures/music-exposure-result-20261029-v1/train-29--step-000-journal.jsonl': 'sha256:e3a0527b4b3fe6ef8f89118af06ad61b97cc88e435c14fca181bcf4eb2a6b00a',
 'fixtures/music-exposure-result-20261029-v1/train-29--step-000-predictions.json': 'sha256:e07352a75364b6070660e0e8841459d7a1be61c456e1d8bf95a9f5154b82a08a',
 'fixtures/music-exposure-result-20261029-v1/train-29--step-064-journal.jsonl': 'sha256:ab2b1f2444fcb17bb45fa6eff04db1728cb911af76e02de00665fca765b82587',
 'fixtures/music-exposure-result-20261029-v1/train-29--step-064-predictions.json': 'sha256:62fa9ed6f1bf277fbc4c02d2130168b45250070b4d80629531d3d49fb21fa7b1',
 'fixtures/music-exposure-result-20261029-v1/train-29--step-640-journal.jsonl': 'sha256:4a1a05c03980040af21cc4f4c23459aedac0b282a0f17c9bc6bd25d62c31d5df',
 'fixtures/music-exposure-result-20261029-v1/train-29--step-640-predictions.json': 'sha256:503c0f780351647c1a127c7cd4516d6f7f7309853cfa69b9c2a8d7c57f9a35c1',
 'fixtures/frozen-music-eval-result-v1/original-attempt/attempt.json': 'sha256:1ae593877597395cf18315256b547ece687b44a7c7da2f344d4de55214e314ed',
 'fixtures/frozen-music-eval-result-v1/original-attempt/predictions.jsonl': 'sha256:b0939e4508697d4909e08acefea50ac568ae008092f1952068145c814355a56f',
 'fixtures/frozen-music-eval-result-v1/original-attempt/replay-gate.json': 'sha256:82f06bab3e8edb6bb3893cc58deb5eeac51096f9cbae6a9022622a84b0ba8198',
 'fixtures/frozen-music-eval-result-v1/original-attempt/report.json': 'sha256:634847e20f796ce045cc3f81287a9506223125cfa9a4a445d8f044a0019848a3',
 'fixtures/frozen-music-eval-result-v1/original-attempt/worker-completion.json': 'sha256:1c785da4b5013cbbbd00cfac7a5da5a9d92b21f4226200f094d7d32328100666',
 'fixtures/frozen-music-eval-result-v1/audit-receipt.json': 'sha256:9131095f48a76d0240f9442c6cdd83df073250e6b07f17a3ddca89b119976aa2',
 'fixtures/frozen-music-eval-result-v1/original-SHA256SUMS.txt': 'sha256:4b7c91f242e4a7f966703816abee8aa4896e65ff71e95e7d6976815f191e9e0d',
 'fixtures/frozen-music-eval-result-v1/README.md': 'sha256:42aefa202458d528dcb12dd045037a1a129e07feb3a5140d5025367239ba7f93',
 'fixtures/frozen-music-eval-result-v1/provenance-DERIVED.json': 'sha256:aa805d745a204df5941f5f0b563b2e4bbee66268c04bb24c4ff3977958947063',
 'fixtures/music-pitch-result-v1/README.md': 'sha256:98d7557c5a98ad77408638f69e779836c90eea6b534665b94e158542e0a046ae',
 'fixtures/music-pitch-result-v1/original-run/audit.stdout': 'sha256:b37d33ed9aa9d8d5144af5cf9ad9cd07927f40db1016cf51fee3fe647641f4ef',
 'fixtures/music-pitch-result-v1/original-run/control-17/attempt.json': 'sha256:ae8458ebd95eb8ef692620accb87200a692fe099ebc96597f0e29ca131094565',
 'fixtures/music-pitch-result-v1/original-run/control-17/completion.json': 'sha256:7cd61f2876554b450c4a2fc46b6752d6f12b2af13a87236fab559d6c63bb8590',
 'fixtures/music-pitch-result-v1/original-run/control-17/diagnostic-predictions.jsonl': 'sha256:a5e73cd8c5ed4998576ecb6a4ff520dea3aa2c26f48dec282fefd0cfc08ec6c2',
 'fixtures/music-pitch-result-v1/original-run/control-17/report.json': 'sha256:7f877e1d9a2bafe11c231fcbbfba46bfe0fd10c58e6877c28976c7a2481d3c3e',
 'fixtures/music-pitch-result-v1/original-run/control-29/attempt.json': 'sha256:96c8ae1c856415276dd3112a9396fed3a4bce95dd8237e0420c22ba95e74504e',
 'fixtures/music-pitch-result-v1/original-run/control-29/completion.json': 'sha256:f823320643d0be291bd67b41386424f8b06753c2441c675c4041e2fd4f705966',
 'fixtures/music-pitch-result-v1/original-run/control-29/diagnostic-predictions.jsonl': 'sha256:e00d7ae5673385bbb332ee60a2ca59d64d37c1a40e5b886e0e0335fa4150afc6',
 'fixtures/music-pitch-result-v1/original-run/control-29/report.json': 'sha256:810b22359b9486bdd17bb011828fe9875d4971f2940b00f347cdadfb15aa9db0',
 'fixtures/music-pitch-result-v1/original-run/development/attempt.json': 'sha256:2c99badb918654b6f184edd11e31021dcbea26d0ed49daf7851641ca7c505ac4',
 'fixtures/music-pitch-result-v1/original-run/development/completion.json': 'sha256:4cb8877e6ba05d5959c56f6f1ca8cd253114ef1b68a732ec424ccbde6b0569e1',
 'fixtures/music-pitch-result-v1/original-run/development/predictions.jsonl': 'sha256:9ad0182bed9593e0c92f464aa84f4d8f929e0402b0f2a2c22328e90ccb65b76d',
 'fixtures/music-pitch-result-v1/original-run/development/report.json': 'sha256:bc2c09fc9552db509b8695c7a3745ed46192a3023d1645041646a470d70a6571',
 'fixtures/music-pitch-result-v1/original-run/development-gate.json': 'sha256:3bd77afe8c845b4fd282f6dfbdebabe859d0bbd714cabf22c07e7579d85cc889',
 'fixtures/music-pitch-result-v1/original-run/matrix.json': 'sha256:59c92c9217a35b2b5a5594fa9ba839a903807398b30788a1c286095f4b69b6db',
 'fixtures/music-pitch-result-v1/original-run/pitch-17/attempt.json': 'sha256:474c8546d4a63011fb60390bd213f2ba955dea2d70148c73c572da08d697f1f1',
 'fixtures/music-pitch-result-v1/original-run/pitch-17/completion.json': 'sha256:b820a9feaeb3dd049d9a6b00d6f4077ef3373dc2675c2561b5f650a5c4358313',
 'fixtures/music-pitch-result-v1/original-run/pitch-17/diagnostic-predictions.jsonl': 'sha256:9e636fdfcc6cbdd8551537ec8647db9bf5c4d8d2980d3ab630cc04164e97977c',
 'fixtures/music-pitch-result-v1/original-run/pitch-17/report.json': 'sha256:5817673adf8b7531e55a621c730a2700d094de27ef391139a0b0f8437c6648b3',
 'fixtures/music-pitch-result-v1/original-run/pitch-29/attempt.json': 'sha256:b6acfb66cc99f46c516f394a0a302f93c02d52012defe942f442cac63a338e95',
 'fixtures/music-pitch-result-v1/original-run/pitch-29/completion.json': 'sha256:7fffc336b348014bb7e07c52480f1d44ef36a4b5fb607c8b83a32c3fefeb4434',
 'fixtures/music-pitch-result-v1/original-run/pitch-29/diagnostic-predictions.jsonl': 'sha256:f542052ddfcf0982b19d1083ff547e2d054ce2c38115798e31068ece0bd31f34',
 'fixtures/music-pitch-result-v1/original-run/pitch-29/report.json': 'sha256:b70d62895a235b6cce1167b1cdb3f835a33703217af9b419c36e7fa2eabb2007',
 'fixtures/music-pitch-result-v1/provenance-DERIVED.json': 'sha256:ca93a6802e384224a887b475078badc1dd2ba704b3cbd28822439694281daab7',
 'fixtures/music-pitch-result-v1/source-SHA256SUMS': 'sha256:452afdfdeb7b5ab83304dc03d509315064672191d59d7fddea7b7f05f8ec0366',
 'fixtures/music-placement-result-v1/README.md': 'sha256:a127c3b0d932dd8faa019c37ba3b6907df18ee678a3ae4a11716a1690b0ff213',
 'fixtures/music-placement-result-v1/cpu/audit.stdout': 'sha256:a8a8492cc01bbca8593efe97a43e2fabb0f208b6a4617d3eb50a103c2f6b0b7f',
 'fixtures/music-placement-result-v1/cpu/control-17/attempt.json': 'sha256:6dcdd1cf8f4bdbd8434a079cdd5494cdbe57c365a13bdc7e4d1830222c1dbef2',
 'fixtures/music-placement-result-v1/cpu/control-17/completion.json': 'sha256:991728d785210d11aa1b9c37c086ad19068a5b93984febd211e6714e587bbd74',
 'fixtures/music-placement-result-v1/cpu/control-17/diagnostic-predictions.jsonl': 'sha256:83589d201a2ed03121569d955723a4c5122b666d997811eda061651259c8f4b0',
 'fixtures/music-placement-result-v1/cpu/control-17/report.json': 'sha256:b8fb39837a5ef5c716a6e67ae6979a29918e1218a5468cbf60159576c9cc052a',
 'fixtures/music-placement-result-v1/cpu/control-29/attempt.json': 'sha256:5ac9ad2b5cb3eb94a481bc6411e41be93f5b52a20896ff7eb7049d4a6e4a0aa7',
 'fixtures/music-placement-result-v1/cpu/control-29/completion.json': 'sha256:3759ea8127bd620817fc940abed1eb3f13c9788bc7f2a6cde03efd1dea3525a6',
 'fixtures/music-placement-result-v1/cpu/control-29/diagnostic-predictions.jsonl': 'sha256:40c42c1bf4af61393591348b45ab5b78e74a1d916bc6960192b5e4516904cb68',
 'fixtures/music-placement-result-v1/cpu/control-29/report.json': 'sha256:293aeb8b589433abdc041dd358a8e9d6f95d5ff053c4542b31f8899afbb62771',
 'fixtures/music-placement-result-v1/cpu/development/attempt.json': 'sha256:89ef1632937fc38c0498fdd1154c42b9a32a32e15ff4236103c1775bda17515d',
 'fixtures/music-placement-result-v1/cpu/development/completion.json': 'sha256:76fa72c9839384e146947a46c1aa1a9b96a84f264a2fc90c9bd81cd66ffae4a0',
 'fixtures/music-placement-result-v1/cpu/development/predictions.jsonl': 'sha256:6f5b4d236d1f8535e34cae27d4ea60066dde529a6cd2fed3b37d97ce4325ea88',
 'fixtures/music-placement-result-v1/cpu/development/report.json': 'sha256:4ec924ac3338f212752a36bee7814d623a5753d36de004126b38487f60e18d5a',
 'fixtures/music-placement-result-v1/cpu/development-gate.json': 'sha256:5471942d8aea2a01327cf78ace3f024d11b9d06096be0b7bdf9965a6657adb56',
 'fixtures/music-placement-result-v1/cpu/matrix.json': 'sha256:a25c8b1a2a0b66bb536c5fb02c7ed2fa359bdb62845b36fd445eb4b1bf21896e',
 'fixtures/music-placement-result-v1/cpu/swap-17/attempt.json': 'sha256:292c01a61189fe480459d1618b44a1c219cd88cd5383740236220c5f8655ea42',
 'fixtures/music-placement-result-v1/cpu/swap-17/completion.json': 'sha256:aeb1bbbd99a43d4d55e3443659084a64b8dea97eeb79bda9b2c0e9a5feb3181c',
 'fixtures/music-placement-result-v1/cpu/swap-17/diagnostic-predictions.jsonl': 'sha256:652b51f2b8e7b68858dfc7c4082301d278b3bfb9854c6fcdcf21ee6d5eafd692',
 'fixtures/music-placement-result-v1/cpu/swap-17/report.json': 'sha256:cb0f0af5d2b93728d3a88aca06dbb972a264de80c710897f1bbcd6402e96a4fb',
 'fixtures/music-placement-result-v1/cpu/swap-29/attempt.json': 'sha256:9ccb14f0e85aed4b0fbfeb7eb583d02534f29d31f92662a25a008bbb27bd554f',
 'fixtures/music-placement-result-v1/cpu/swap-29/completion.json': 'sha256:9cc79d31d7a9d69210f69c2a7ad214e5ef1f7fe74f1fb606fe2413ced4d22e1d',
 'fixtures/music-placement-result-v1/cpu/swap-29/diagnostic-predictions.jsonl': 'sha256:dc106535e2c2b5493bf8ffb684ad82282ca760e437064484cd48ccbab9ffd283',
 'fixtures/music-placement-result-v1/cpu/swap-29/report.json': 'sha256:00f7679561308532a34f41b62365c3d1e2b8f1a1876001797a96fa6fd2940101',
 'fixtures/music-placement-result-v1/device/audit.stdout': 'sha256:a2712ca2eb45f1fc7aea67e54e37263df6be5ad8068bb7f0c3cbf3ef5eec8798',
 'fixtures/music-placement-result-v1/device/cpu/attempt.json': 'sha256:b332c08ac7dfe13ca256fc8c2555452110b17e850893f869d01e189d27831ede',
 'fixtures/music-placement-result-v1/device/cpu/completion.json': 'sha256:3a90046fd19a6615542d5194515122c6e5e73df9f0774fb0eb575a30ec690144',
 'fixtures/music-placement-result-v1/device/cpu/diagnostic-predictions.jsonl': 'sha256:41eb9b8d0dd70c7e852bb45a55c27656df2a42d5dd4f78afa6974a73fe248c58',
 'fixtures/music-placement-result-v1/device/cpu/report.json': 'sha256:d46773c26352984fb2b4f164a858a57af5388f42bfb145498a145d65507db037',
 'fixtures/music-placement-result-v1/device/cuda/attempt.json': 'sha256:6c223cc64bac1bb64a345e0a16d43014ae440c45a3b2021afe07750fe64f1746',
 'fixtures/music-placement-result-v1/device/cuda/completion.json': 'sha256:2839ce6019d4c35687abb82fa68718b059f8989fe95d7169fa80bd60a787c31a',
 'fixtures/music-placement-result-v1/device/cuda/diagnostic-predictions.jsonl': 'sha256:ffa403e17ed1eb6826e5d00f508655e243145974c6b7b48e0e1535bc6781f182',
 'fixtures/music-placement-result-v1/device/cuda/report.json': 'sha256:473abcd161d879e1c65af29a04892fe43b91cac4c5490a84a5400b80e7764e6a',
 'fixtures/music-placement-result-v1/device/development/attempt.json': 'sha256:11e7a0e765522d0dd0542fbba16fd2adaff4234151e175090418d5b9ff63317d',
 'fixtures/music-placement-result-v1/device/development/completion.json': 'sha256:e4c5834daa3262c9a1bc72d05440964fbebe7ebc8efb03ff4678290828dbaba8',
 'fixtures/music-placement-result-v1/device/development/predictions.jsonl': 'sha256:7f33603c154d8c22c6d7075ec01329bb75775cd37642fe67d3a0bf24ca948e55',
 'fixtures/music-placement-result-v1/device/development/report.json': 'sha256:021b03aee77e0356fae4f393ae63f58194a2e151d69f497dfca272c0f01a6f4a',
 'fixtures/music-placement-result-v1/device/development-gate.json': 'sha256:24bb11366ff5df48dd0af43e0ac01ee641abc763a13b042812e53f2387f49910',
 'fixtures/music-placement-result-v1/device/matrix.json': 'sha256:e6641be61e71d9b1db8919d0e224d95c66bb2b1ec46038d2c8eec340b5c37bb5',
 'fixtures/music-placement-result-v1/device/probe/attempt.json': 'sha256:6a41f1845bf97ad7e486939d870e90f956b119ac328088ca0f972fba774a4982',
 'fixtures/music-placement-result-v1/device/probe/comparison.json': 'sha256:4da425bb13cef1d5bf758bd68f319b57dd93d656fc891350ced1ccb2bf585c23',
 'fixtures/music-placement-result-v1/device/probe/completion.json': 'sha256:bd377f6164dd1ac7cef44a3b8a38ea47362770ad2a5907a1b2024e05c4412aa3',
 'fixtures/music-placement-result-v1/device/probe/report.json': 'sha256:5483eebb2086daf76c3780024a9ceaf6f95022b1b9d5574bf7f046d24698734b',
 'fixtures/music-placement-result-v1/provenance-DERIVED.json': 'sha256:aa243e5fc0131c589cdbb8b6e8cf868015adb995cec81648554322243218398f',
 'fixtures/music-placement-result-v1/source-SHA256SUMS': 'sha256:9178b5006ac3a0ea008fdc0b070f599790df45dcbd983b8c093627de074e0f6f',
 'fixtures/music-robustness-result-v1/README.md': 'sha256:b473ff5e6aeb59cb9a76b9936a3dafe1b50d581e1afce3a909a0e86cf7091c56',
 'fixtures/music-robustness-result-v1/cpu/audit.stdout': 'sha256:28e853ff7f24c3e209de406969e19e09fa924acb4f992aa51ed0d3683cf2f825',
 'fixtures/music-robustness-result-v1/cpu/control-17/attempt.json': 'sha256:b2256480e0fe4a26f1346d369adb2554ad5dc09f31ba70439d95af242405dc58',
 'fixtures/music-robustness-result-v1/cpu/control-17/completion.json': 'sha256:ffa7a6a81511da13f20e9ef8fe7024002b1ee4e816d308826756c79aa1e8f8a3',
 'fixtures/music-robustness-result-v1/cpu/control-17/diagnostic-predictions.jsonl': 'sha256:83589d201a2ed03121569d955723a4c5122b666d997811eda061651259c8f4b0',
 'fixtures/music-robustness-result-v1/cpu/control-17/report.json': 'sha256:84b1b5af3f970dddc26800290e8ac092d76214dc08dda55b49377102aefd15a3',
 'fixtures/music-robustness-result-v1/cpu/control-29/attempt.json': 'sha256:ed63ee0d2270f340f51e488ffb39f7b64bb11aecb434f4244d6de714b2215dfa',
 'fixtures/music-robustness-result-v1/cpu/control-29/completion.json': 'sha256:4c26914930ab2a570693e246c59cc258fedbc837c847e57a6efb71022c38f1da',
 'fixtures/music-robustness-result-v1/cpu/control-29/diagnostic-predictions.jsonl': 'sha256:40c42c1bf4af61393591348b45ab5b78e74a1d916bc6960192b5e4516904cb68',
 'fixtures/music-robustness-result-v1/cpu/control-29/report.json': 'sha256:125e240b67ffd9a082923719fdf3afb05ca58965d9f5eaee6625ff452b75b825',
 'fixtures/music-robustness-result-v1/cpu/development/attempt.json': 'sha256:b206151cf45190fdc11b93b01d0f1c488a2638b71e6a3c886429e4cb51bd05b9',
 'fixtures/music-robustness-result-v1/cpu/development/completion.json': 'sha256:47e690f476f17e6bbe4913e15383e59ef41d2a410bdd22125bc15e514df70210',
 'fixtures/music-robustness-result-v1/cpu/development/predictions.jsonl': 'sha256:6f5b4d236d1f8535e34cae27d4ea60066dde529a6cd2fed3b37d97ce4325ea88',
 'fixtures/music-robustness-result-v1/cpu/development/report.json': 'sha256:c98552a43aa6af52f7fb312bd60f65bdd81e419139866744f991a31b166cb312',
 'fixtures/music-robustness-result-v1/cpu/development-gate.json': 'sha256:860ad49f9bddbe2581404e3daa9881e683527968de93ddc082070df3e6ee1daa',
 'fixtures/music-robustness-result-v1/cpu/matrix.json': 'sha256:3aefd8026194a557cef3dcbee380e61dbb81f50ffb42dfc264900bf0ea0921e2',
 'fixtures/music-robustness-result-v1/cpu/swap-17/attempt.json': 'sha256:ebc385b0d7fcd4feb637d4f684cdc8d9fa82888a14080d8214b411ca317b7e29',
 'fixtures/music-robustness-result-v1/cpu/swap-17/completion.json': 'sha256:94bd4d53cf1aa0edb938c66c760f92bc0d39664c3e4fc2d469e2065acacacc3c',
 'fixtures/music-robustness-result-v1/cpu/swap-17/diagnostic-predictions.jsonl': 'sha256:652b51f2b8e7b68858dfc7c4082301d278b3bfb9854c6fcdcf21ee6d5eafd692',
 'fixtures/music-robustness-result-v1/cpu/swap-17/report.json': 'sha256:9889b6b94fbfd29485f724f552e554d35d0a470ef992a4c49ef302713cff8e4f',
 'fixtures/music-robustness-result-v1/cpu/swap-29/attempt.json': 'sha256:63a03ac3dac30fc69a0068eceb56e8eaa20a162836566473e7cbdd8ba4a1a0e9',
 'fixtures/music-robustness-result-v1/cpu/swap-29/completion.json': 'sha256:f67e3957775dca6f4fee1ee63224b6078901d9f3e2f762811b183fdf68509ce5',
 'fixtures/music-robustness-result-v1/cpu/swap-29/diagnostic-predictions.jsonl': 'sha256:dc106535e2c2b5493bf8ffb684ad82282ca760e437064484cd48ccbab9ffd283',
 'fixtures/music-robustness-result-v1/cpu/swap-29/report.json': 'sha256:01c4024cfe84d86c40f55c05fff006ecf266af4d3ad212dc6f6b42d0dc7ca467',
 'fixtures/music-robustness-result-v1/device/freeze.stderr': 'sha256:86c9c26aeeb44f30172caa707e36a0e998152c69689113127d236fb679d27ce6',
 'fixtures/music-robustness-result-v1/provenance-DERIVED.json': 'sha256:2563ca5d7a966b7ef2bb13e8e8c7882291483597bd716d43c8ae884a8cd2885b',
 'fixtures/music-robustness-result-v1/source-SHA256SUMS': 'sha256:e8b93d7eba92d75f0f5969195c5d4cd610fc7de4b035d857c56513bd51f782c2'}
VERIFIERS = ['music_exposure_result_v1.check',
 'frozen_music_eval_result_v1.check',
 'music_pitch_result_v1.check',
 'music_placement_result_v1.check',
 'music_robustness_result_v1.check']
ANALYSIS_HASHES = {'scripts/music_task_audit_v1/__init__.py': 'sha256:a446367b088afc819f3936f0721abc3dbc250e2605c8253598bd80f9d4d3cbf6',
 'scripts/music_task_audit_v1/analysis.py': 'sha256:9f81df4dadd97b1e55302f7afd2926ccce069895570d1547b676b36f14a52b4a'}

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
    for relative, expected in SOURCES.items():
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



def snapshot():
    result = {relative: read(ROOT/relative) for relative in INPUT_HASHES}
    if any(digest(raw) != INPUT_HASHES[name] for name,raw in result.items()):
        raise ValueError('immutable evidence bytes differ')
    return result


def verify():
    sources()
    for relative, expected in ANALYSIS_HASHES.items():
        if digest(read(ROOT/relative)) != expected:
            raise ValueError('audit analysis source differs')
    before = snapshot()
    receipts = {name: importlib.import_module(name).verify() for name in VERIFIERS}
    sources()
    from music_task_audit_v1.analysis import build_report, INPUT_PATHS
    if not set(INPUT_PATHS) <= set(before):
        raise ValueError('analysis requested unbound evidence')
    report = build_report({name: before[name] for name in INPUT_PATHS})
    if snapshot() != before:
        raise ValueError('evidence changed during audit')
    sources()
    for relative, expected in ANALYSIS_HASHES.items():
        if digest(read(ROOT/relative)) != expected:
            raise ValueError('audit analysis source changed during verification')
    return {'schema':'trinite.music-task-audit.v1', 'evidence_class':'DERIVED',
        'source_hashes':dict(SOURCES), 'audit_implementation_hashes':{p:digest(read(ROOT/p)) for p in ('scripts/check_music_task_audit.py','scripts/music_task_audit_v1/__init__.py','scripts/music_task_audit_v1/check.py','scripts/music_task_audit_v1/analysis.py')}, 'input_hashes':dict(INPUT_HASHES),
        'preservation_receipts':receipts, 'analysis':report,
        'scope':'finite source coverage and retained raw-output associations, not a causal explanation',
        'model_imported':False, 'inference_replayed':False, 'training_performed':False,
        'causal_mechanism_established':False, 'fresh_confirmation':False,
        'full_original_custody_verified':False, 'hardware_attested':False,
        'learning_adequate':False, 'release_ready':False}


def main():
    try:
        if len(sys.argv) != 1:
            raise ValueError('this read-only audit takes no arguments')
        raw = json.dumps(verify(),sort_keys=True,separators=(',',':'),allow_nan=False)
        if len(raw.encode('utf-8')) > LIMIT:
            raise ValueError('derived report exceeds fixed output bound')
        print(raw)
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(str(error),file=sys.stderr)
        return 1
    return 0
