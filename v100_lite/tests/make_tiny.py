"""Make a small deterministic FP16 Llama checkpoint without a network download."""

from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

root = Path(__file__).resolve().parents[2] / "artifacts/tiny-llama"
root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
config = LlamaConfig(
    vocab_size=128,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)
model = LlamaForCausalLM(config).half().eval()
model.save_pretrained(root)
vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
vocab.update({f"token{i}": i for i in range(4, 128)})
t = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
t.pre_tokenizer = Whitespace()
PreTrainedTokenizerFast(
    tokenizer_object=t,
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
).save_pretrained(root)
ids = torch.tensor([[4, 5, 6, 7]])
with torch.inference_mode():
    expected = model.generate(ids, max_new_tokens=16, do_sample=False)
(root / "expected.json").write_text(__import__("json").dumps(expected.tolist()))
print(root)
