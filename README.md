# Cue v1

**Turn detection for voice agents by a fine-tuned decision model.** Cue v1 is Gemma 4 E4B, fine-tuned to answer one
typed question at every pause in a conversation, using the context of the speaker:

| Answer | What it means | What the agent does |
|---|---|---|
| **DONE** | A complete question, answer, request or statement | Speak now |
| **LIKELY_DONE** | Probably finished, but it trails off | A short grace, then speak |
| **MID_THOUGHT** | The sentence is left hanging: "my number is nine eight four five..." | Keep listening |
| **HOLDING** | Telling a story or explaining something: the pause is thinking | Keep listening, longer |

A second decision runs while the agent is speaking: when the caller makes a sound, **STOP** ("wait, Tuesday doesn't
work") or **CONTINUE** (a backchannel: "hmm", "okay").

The answer is read as probabilities from a single token, so nothing is generated: the agent knows not just whether
to wait, but how long, and why. Combined with an audio turn model (Smart Turn v3.2), Cue v1 is **4th on LiveKit's
end-of-turn benchmark in English** and **1st in Portuguese**, with one model and one English prompt across 14 languages.

- **Weights:** [huggingface.co/Pype/cue-v1](https://huggingface.co/Pype/cue-v1): `cue-v1-q8_0.gguf` for llama.cpp, and the LoRA adapter
- **This repo:** a small client, the prompts, the benchmark code and results, and the training recipe
- **Part of:** Shortwire, Pype's open source voice AI stack, where decision models run the call and the LLM only talks

![Cue v1 on eot-bench, English](assets/cue-v1-english.png)

## Quick start

```bash
pip install -e .
huggingface-cli download Pype/cue-v1 cue-v1-q8_0.gguf --local-dir .
llama-server -m cue-v1-q8_0.gguf -c 8192 -np 4 -ngl 999 --jinja --port 8090
python examples/quickstart.py
```

```
"Sure, it's nine eight four five,"                   -> MID_THOUGHT  p(done) 0.00, wait 1.75 s
"It's nine eight four five six seven one two three four." -> DONE         p(done) 1.00, wait 0.51 s
'So last week I went to the clinic and the doctor told me that' -> MID_THOUGHT  p(done) 0.00, wait 1.75 s
agent talking, caller says "wait, Tuesday doesn't work"   -> STOP (0.998)
agent talking, caller says 'hmm'                          -> CONTINUE (1.000)
```

In your own code:

```python
from cue import Cue, ending, load, p_end, wait, with_voice

cue = Cue("http://localhost:8090/v1")
eot = load("prompts/end_of_turn_phone.json")

caller = "Sure, it's nine eight four five,"
probs = cue.ask(eot, {"agent_question": "What's the best number to reach you on?", "caller_earlier": "",
                      "caller_text": caller, "transcript_end": ending(caller)})
# {'DONE': 0.0, 'LIKELY_DONE': 0.0, 'MID_THOUGHT': 1.0, 'HOLDING': 0.0}

p_end(probs)               # p(the caller is done): DONE + LIKELY_DONE
wait(probs)                # seconds of silence to wait, blended by the answer
with_voice(probs, 0.8)     # the eot-bench recipe: p_end x sqrt(Smart Turn's score)
```

`cue.aask(...)` is the async version. Any OpenAI-compatible server that returns `logprobs` works (llama.cpp, vLLM).

## Prompts

The prompts are part of the model. Each is a JSON decision: an instruction, the options and what they mean, the
inputs it reads, and optional worked examples. `cue.messages(decision, inputs)` renders it exactly as Cue v1 was
trained and benchmarked (`tests/` check this byte for byte against Shortwire's renderer).

| File | Use |
|---|---|
| `prompts/end_of_turn_phone.json` | End of turn, a caller and an agent (the eot-bench prompt, with 11 English examples) |
| `prompts/end_of_turn_dialogue.json` | End of turn, two people talking (the TurnBench prompt) |
| `prompts/barge_in.json` | STOP or CONTINUE while the agent is speaking |

## Results: LiveKit's eot-bench

All 400 English caller turns and every turn in 13 more languages, scored with the benchmark's own harness. Cue v1 =
the fine-tuned model's p(done) × √Smart Turn v3.2, with Smart Turn run by us on the same moments.

| English | Latency @ 5% cut-offs | @ 10% | Cut-offs @ 300 ms | @ 600 ms |
|---|---|---|---|---|
| LiveKit turn-detector v1 (hosted) | 543 ms | 295 ms | 9.9% | 4.5% |
| Baton v1 | 577 ms | 350 ms | 12.3% | 4.8% |
| Soniox | 647 ms | 512 ms | – | 5.5% |
| **Cue v1** | **719 ms** | **532 ms** | **26.8%** | **7.5%** |
| ultraVAD | 899 ms | 663 ms | 27.7% | 11.9% |
| Smart Turn v3.2 | 1,051 ms | 739 ms | 35.2% | 14.8% |
| LiveKit turn-detector v1-mini (local) | 1,070 ms | 698 ms | 27.8% | 12.1% |
| OpenAI GPT Realtime 2 | 1,143 ms | 824 ms | – | – |
| Deepgram Flux | 1,151 ms | 548 ms | 12.9% | 9.9% |

Across the 14 languages, with the same English prompt: **1st in Portuguese** (681 ms against 702 ms for LiveKit v1
and Soniox), **2nd in Spanish, Korean, Chinese and Indonesian**, top 4 in 13 of 14. Faster than Smart Turn v3.2 in 26
of 28 language × budget comparisons and ultraVAD in 25 of 28. Still behind LiveKit's hosted model in all but one.

![Latency at a 5% false cut-off rate, 14 languages](assets/cue-v1-languages-latency.png)

<details>
<summary>The other views: cut-offs at 300 ms and 600 ms, latency at 10%</summary>

![False cut-offs at a 300 ms latency budget](assets/cue-v1-languages-cutoffs.png)

![False cut-offs at a 600 ms latency budget](assets/cue-v1-languages-cutoffs-600ms.png)

![Latency at a 10% false cut-off rate](assets/cue-v1-languages-latency-10pct.png)

</details>

Every number is in `results/eot_bench/` (`summary.json`, and `report.md` with all four views per language). The
images are drawn from `summary.json` by `results/eot_bench/make_infographics.py` and `make_linkedin.py`.

### Reproduce

```bash
git clone https://github.com/livekit/eot-bench && pip install -e eot-bench -e ".[eval]"
# Cue v1's words (the model served as above), every turn of every language
python eval/eot_bench/run.py --harness eot-bench --rows 0 --lang en ar de es fr hi id it ja ko nl pt tr zh
# Smart Turn v3.2 on the same moments
python eval/eot_bench/run.py --harness eot-bench --rows 0 --lang en ar de es fr hi id it ja ko nl pt tr zh \
    --reference eot_harness.smart_turn_adapter:SmartTurnAudioAdapter
# combine, rank against every published model, draw the heatmaps
python eval/eot_bench/leaderboard.py --harness eot-bench
```

Each run prints its model calls and how many failed; a failed call scores 0, so check that it says `(0 failed)`.
llama.cpp is not bit-exact when it serves several requests at once: a rerun matches the published scores to within
about 0.003 on average, which does not move the ranks.

## Training

Cue v1 never saw a real recording or a benchmark transcript. It was trained on **310 conversations written by an
LLM** to a written specification (`training/SPEC.md`): 150 phone calls between a caller and an agent and 160
two-person conversations, each on a different topic, with about 10,600 labelled pauses and about 1,000 listener sounds
(backchannels and real interruptions). The mix was shaped by a few aggregate statistics of TurnBench's public dev
split (how often a pause is a hold, how often a hold follows a complete sentence, how often a turn ends without
punctuation); no benchmark audio, transcript or label was used as an example.

```bash
python training/validate.py conversations/*.jsonl              # check a batch against the spec
python training/build.py --lean --v5 conversations/*.jsonl     # -> train_lean_v5.jsonl, test_lean_v5.jsonl
python training/train_unsloth.py --train train_lean_v5.jsonl --test test_lean_v5.jsonl --out cue-v1 --gguf merged
python training/eval_letters.py test_lean_v5.jsonl              # held-out accuracy against a served model
```

- LoRA rank 16 on Gemma 4 E4B (4-bit) with Unsloth, the loss on the answer letter only, one epoch, one NVIDIA T4
  (float32, about 2.5 hours). `--gguf merged` writes 16-bit merged weights for llama.cpp's `convert_hf_to_gguf.py`.
- Held out (10% of conversations, never trained on): **83.2%** exact over the options and **86.9%** done vs not,
  against 38.3% and 58.2% for the base model with the same prompts. The base model almost never answers HOLDING.
- We also tried Unsloth's decision-model head (`FastDecisionModel`) on the same data: 72.2% exact after one epoch,
  behind the letter fine-tune and not servable by llama.cpp, so Cue v1 is the letter fine-tune.
- The 310 conversations themselves are not published; the spec and scripts are.

## Limits

- **Reads words, not prosody.** Pair it with an audio turn model; on eot-bench the combination is what ranks 4th.
- **Needs an ASR and a GPU.** About 100 to 300 ms per decision on a T4 with llama.cpp; not an on-device model.
- **Trained on English.** The other 13 languages run on the English prompt; Arabic is the weakest (6th).
- **Trained on simulated conversations.** Measure it on your own calls before relying on its probabilities.

## Licence

Apache 2.0 (`LICENSE`), the same as Gemma 4 E4B. See `NOTICE`.

```bibtex
@misc{pype2026cue,
  title  = {Cue v1: turn detection by a fine-tuned decision model},
  author = {Pype},
  year   = {2026},
  url    = {https://huggingface.co/Pype/cue-v1}
}
```
