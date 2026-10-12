"""Data alignment, completion-only supervision, and SFT -> RL handoff."""
import copy
from pathlib import Path
import tempfile
import unittest

from train_tapo_sft import prepare_sft_rows, split_sft_rows, tokenize_sft_row
from train_tapo_trl import last_boxed, translation_span, load_initial_adapter


class SFTTests(unittest.TestCase):
    def row(self, **changes):
        row = dict(question="தமிழ் கேள்வி", english_question="What is two plus two?",
                   answer="4", language="ta", source_id="gsm8k_train_0")
        return {**row, **changes}

    def source(self):
        return [dict(question="What is two plus two?", answer="Two plus two is <<2+2=4>>4.\n#### 4")]

    def test_official_solution_target_has_translation_and_gold_answer(self):
        row = prepare_sft_rows([self.row()], self.source())[0]
        self.assertEqual(translation_span(row['completion']).group(1).strip(), self.row()['english_question'])
        self.assertEqual(last_boxed(row['completion']), '4')
        self.assertIn('Two plus two is 4.', row['completion'])
        self.assertNotIn('<<', row['completion'])
        self.assertNotIn('####', row['completion'])
        self.assertIn('\n', row['completion'])

    def test_rejects_test_rows_and_misaligned_train_questions(self):
        for row in (self.row(source_id='gsm8k_test_0'), self.row(english_question='Different question'),
                    self.row(answer='5'), self.row(source_id='gsm8k_train_10')):
            with self.assertRaises(ValueError):
                prepare_sft_rows([row], self.source())
        with self.assertRaises(ValueError):
            prepare_sft_rows([self.row(), self.row()], self.source())

    def test_train_eval_split_is_disjoint_and_deterministic(self):
        rows = [{'source_id':f'gsm8k_train_{i}'} for i in range(20)]
        train, val = split_sft_rows(rows, 12, 4, 3407)
        self.assertFalse({r['source_id'] for r in train} & {r['source_id'] for r in val})
        self.assertEqual((train,val),split_sft_rows(rows,12,4,3407))

    def test_validation_excludes_previously_trained_questions(self):
        rows = [{'source_id': f'gsm8k_train_{i}'} for i in range(20)]
        excluded = {r['source_id'] for r in rows[:8]}
        train, val = split_sft_rows(rows, 12, 4, 3407, excluded)
        self.assertFalse(excluded & {r['source_id'] for r in val})
        self.assertFalse({r['source_id'] for r in train} & {r['source_id'] for r in val})
        with self.assertRaisesRegex(ValueError, 'unseen'):
            split_sft_rows(rows, 12, 4, 3407, {r['source_id'] for r in rows})

    def test_supervision_masks_prompt_and_never_crops_gold_answer(self):
        class Tokenizer:
            eos_token='<EOS>'
            eos_token_id=0
            def apply_chat_template(self, messages, **kwargs):
                self.messages=messages
                return {'input_ids':[1,2,3,4]}
            def encode(self, text, **kwargs):
                return [ord(c)+10 for c in text[:-len(self.eos_token)]]+[0]
        t=Tokenizer()
        row=prepare_sft_rows([self.row()],self.source())[0]
        encoded=tokenize_sft_row(row,t,1000)
        self.assertEqual(encoded['labels'][:4],[-100]*4)
        self.assertEqual(encoded['labels'][4:],encoded['input_ids'][4:])
        self.assertEqual(encoded['labels'][-1],0)
        self.assertIsNone(tokenize_sft_row(row,t,len(encoded['input_ids'])-1))
        self.assertEqual(t.messages[-1]['content'],row['question'])
        self.assertNotIn(row['completion'],str(t.messages))

    def test_rl_loads_trained_sft_weights_as_trainable_adapter(self):
        import torch
        from transformers import GPT2Config,GPT2LMHeadModel
        from peft import LoraConfig,get_peft_model
        config=GPT2Config(n_embd=16,n_layer=1,n_head=2,vocab_size=32,n_positions=64)
        config._name_or_path='toy-base'
        pristine=GPT2LMHeadModel(config)
        sft=get_peft_model(copy.deepcopy(pristine),LoraConfig(r=2,lora_alpha=4,
                          target_modules=['c_attn'],task_type='CAUSAL_LM'))
        with torch.no_grad():
            for name,p in sft.named_parameters():
                if '.lora_B.' in name:p.fill_(0.125)
        with tempfile.TemporaryDirectory() as directory:
            sft.save_pretrained(directory)
            rl=load_initial_adapter(copy.deepcopy(pristine),Path(directory))
            expected={n:p for n,p in sft.named_parameters() if '.lora_' in n}
            actual=dict(rl.named_parameters())
            for name,p in expected.items():
                torch.testing.assert_close(actual[name],p)
                self.assertTrue(actual[name].requires_grad)
            wrong=copy.deepcopy(pristine);wrong.name_or_path='other-base'
            with self.assertRaisesRegex(ValueError,'does not match'):
                load_initial_adapter(wrong,Path(directory))


if __name__=='__main__':
    unittest.main()
