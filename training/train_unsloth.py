"""LoRA fine-tune of Gemma 4 E4B on the synthetic turn-taking decisions, with Unsloth, on one T4.

    python train_unsloth.py --train train_lean_v5.jsonl --test test_lean_v5.jsonl --out cue-v1 --gguf merged

Each example is the decider's own prompt (system + user, rendered by cue.messages) and a one-letter answer; the
loss is on the answer only, so the model learns which letter to put its probability on, which is exactly
what the live decider reads from the first token. At the end it scores the held-out conversations the same
way and writes a merged GGUF for llama.cpp (the decider's server).
"""
from __future__ import annotations

import argparse
import json
import math


def letters_accuracy(model, tokenizer, rows, limit=600):
    import torch
    ok = binary = n_bin = 0
    letter_ids = {L: tokenizer.encode(L, add_special_tokens=False)[0] for L in "ABCDEFGH"}
    for r in rows[:limit]:
        prompt = tokenizer.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1]
        probs = {L: math.exp(logits[i].item()) for L, i in letter_ids.items()}
        gold = r["messages"][-1]["content"]
        pred = max(probs, key=probs.get)
        ok += pred == gold
        if r["meta"]["decision"] != "barge_in":
            done = (probs["A"] + probs["B"]) / sum(probs[L] for L in "ABCD")
            binary += (done >= 0.5) == (gold in "AB")
            n_bin += 1
    return ok / min(len(rows), limit), binary / max(n_bin, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", default="unsloth/gemma-4-E4B-it")
    ap.add_argument("--out", default="cue-lora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--gguf", default="q8_0", help="quantisation for the exported GGUF; '' to skip; 'merged' writes "
                    "16-bit merged weights only, for llama.cpp's convert_hf_to_gguf.py (Unsloth's GGUF export OOMs a 15 GB box)")
    a = ap.parse_args()

    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, processor = FastModel.from_pretrained(a.model, max_seq_length=a.max_len, load_in_4bit=True,
                                                 full_finetuning=False)
    tokenizer = getattr(processor, "tokenizer", processor)    # Gemma 4 comes back as a multimodal processor
    model = FastModel.get_peft_model(model, r=a.rank, lora_alpha=a.rank, lora_dropout=0.0,
                                     finetune_language_layers=True, finetune_attention_modules=True,
                                     finetune_mlp_modules=True, finetune_vision_layers=False, random_state=7)
    train = [json.loads(l) for l in open(a.train)]
    test = [json.loads(l) for l in open(a.test)]
    print("before:", letters_accuracy(model, tokenizer, test))

    def fmt(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]}
    ds = Dataset.from_list([{"messages": r["messages"]} for r in train]).map(fmt, batched=True, remove_columns=["messages"])
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds,
                         args=SFTConfig(dataset_text_field="text", per_device_train_batch_size=4,
                                        gradient_accumulation_steps=4, num_train_epochs=a.epochs,
                                        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
                                        logging_steps=20, optim="adamw_8bit", weight_decay=0.01, seed=7,
                                        output_dir=a.out, report_to="none", max_seq_length=a.max_len))
    # the loss on the answer only: the prompt is the decider's, the answer one letter
    trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n",
                                      response_part="<|turn>model\n")
    trainer.train()
    print("after:", letters_accuracy(model, tokenizer, test))
    model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    if a.gguf == "merged":
        model.save_pretrained_merged(f"{a.out}-merged", processor, save_method="merged_16bit")
    elif a.gguf:
        model.save_pretrained_gguf(f"{a.out}-gguf", processor, quantization_method=a.gguf)


if __name__ == "__main__":
    main()
