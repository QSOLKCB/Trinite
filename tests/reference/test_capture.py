"""Required optional backend conformance; tiny random models are simulations."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from trinite.contracts import ModelConfig
from trinite.model import Decoder, CaptureSpec
from trinite.reference_capture import hidden, native_prompt, reference_prompt, tensor_manifest, generate, capture
from trinite.reference_geometry import decode_vector


class ReferenceCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

    def test_native_selected_content_and_generation_are_immutable(self):
        model = Decoder(ModelConfig(),lane='dense',seed=0).eval()
        encoding = native_prompt(model,'A:1,2=')
        ids = torch.tensor([encoding['input_ids']]); p = encoding['selected_position']
        before = tensor_manifest(model); rng = torch.get_rng_state().clone()
        capture = hidden(model,encoding,True)
        direct = model(ids,torch.ones_like(ids,dtype=torch.bool),capture=CaptureSpec(('block.0','final'),(p,)))
        self.assertEqual(decode_vector(capture['final']),direct.captures['final'][0,0].tolist())
        self.assertEqual(generate(model,None,encoding,True),generate(model,None,encoding,True))
        self.assertEqual(before,tensor_manifest(model)); self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_qwen_hidden_layers_match_direct_forward_and_do_not_mutate(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
                max_position_embeddings=128,attn_implementation='eager')).eval()
        e = {'input_ids':[1,2,3,4],'selected_position':2}
        before = tensor_manifest(model); rng = torch.get_rng_state().clone()
        capture = hidden(model,e,False)
        ids = torch.tensor([e['input_ids']])
        with torch.no_grad(): direct = model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_hidden_states=True)
        self.assertEqual(decode_vector(capture['block.0']),direct.hidden_states[1][0,2].tolist())
        self.assertEqual(decode_vector(capture['final']),direct.hidden_states[-1][0,2].tolist())
        self.assertEqual(before,tensor_manifest(model)); self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_chat_alignment_keeps_system_and_user_boundaries(self):
        inner = Tokenizer(WordLevel({'[UNK]':0},unk_token='[UNK]')); inner.pre_tokenizer=Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner,unk_token='[UNK]')
        tokenizer.add_special_tokens({'additional_special_tokens':['<|im_start|>','<|im_end|>']})
        tokenizer.chat_template = "{% for m in messages %}{{ '<|im_start|>'+m['role']+'\\n'+m['content']+'<|im_end|>\\n' }}{% endfor %}{{ '<|im_start|>assistant\\n' }}"
        stock = reference_prompt(tokenizer,'A:1,2=',None)
        profile = reference_prompt(tokenizer,'A:1,2=','Only an answer.')
        self.assertLess(stock['user_span'][0],profile['user_span'][0])
        for e in (stock,profile):
            self.assertEqual(e['offsets'][e['selected_position']][1],e['user_span'][1])

    def test_capture_preserves_entry_rng_on_success_and_failure(self):
        model=Decoder(ModelConfig(),lane='dense',seed=0)
        request={'probes':[{'prompt':'A:1,2=','answer':'3','example_identity':'simulation'}],
                 'protocol':{'system':'Only an answer.'}}
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);(directory/'request.json').write_bytes(b'simulation-only\n')
            entry=torch.get_rng_state().clone()
            with patch('trinite.reference_capture.validate',return_value=(request,{})), \
                 patch('trinite.reference_capture.native_model',return_value=model):
                capture(directory,'native')
            self.assertTrue(torch.equal(entry,torch.get_rng_state()))
            with patch('trinite.reference_capture.validate',side_effect=ValueError('rejected input')):
                with self.assertRaises(ValueError):capture(directory,'native')
            self.assertTrue(torch.equal(entry,torch.get_rng_state()))


if __name__ == '__main__': unittest.main()
