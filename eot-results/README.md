# Cue v1: results and analysis

What we measured, how, what helped, what didn't, and where Cue v1 still loses. Every eot-bench number here can be
checked from the saved predictions in [`runs/`](runs/) with LiveKit's own harness. See
[Check our numbers](#check-our-numbers).

| In this folder | What it is |
|---|---|
| `runs/<language>/` | Every run behind the numbers, in the harness's own format: predictions, manifest, metrics |
| `summary.json`, `report.md` | Cue v1 ranked against every published model, all 14 languages and four views |
| `verify.py` | Recomputes all of it with LiveKit's harness. No GPU or model needed |
| `reproduce/` | Runs the benchmark from scratch against a served Cue v1 |
| `charts/` | The charts below |
| `prompts/` | The production prompt that the v5 prompt is compared with under "What helped" |

**Cue v1 = the fine-tuned Gemma 4 E4B's logits (the words and context) × √Smart Turn v3.2 (the voice).** "Gemma part
only" below is the fine-tune on its own; "Smart Turn" is Smart Turn v3.2 on its own.

## Charts

![Cue v1 on eot-bench, English](charts/cue-v1-english.png)

![Latency at a 5% false cut-off rate, 14 languages](charts/cue-v1-languages-latency.png)

<details>
<summary>The other views: cut-offs at 300 ms and 600 ms, latency at 10%</summary>

![False cut-offs at a 300 ms latency budget](charts/cue-v1-languages-cutoffs.png)

![False cut-offs at a 600 ms latency budget](charts/cue-v1-languages-cutoffs-600ms.png)

![Latency at a 10% false cut-off rate](charts/cue-v1-languages-latency-10pct.png)

</details>

## Check our numbers

`runs/<language>/` holds every run behind the numbers, in the harness's own format (predictions, manifest, metrics):

| Folder | What it is |
|---|---|
| `cue_v1/` | Cue v1: the published predictions |
| `cue_v1_gemma_part/` | The fine-tune's words-only scores, before the voice is added |
| `smart_turn_v3.2/` | Smart Turn v3.2 run by us on the same moments: the voice half |
| `cue_v1_on_base_gemma/` | The same recipe on the base Gemma, for the fine-tune comparison |

No GPU or model needed. From the repo root, one script does three things with LiveKit's harness:

- recomputes every run's metrics;
- rebuilds Cue v1 from its two halves;
- ranks it against the published models.

```bash
git clone https://github.com/livekit/eot-bench && pip install -e eot-bench
python eot-results/verify.py --harness eot-bench
# ... Cue v1: 4 of 13 in English; 1 of 11 in Portuguese
# ALL CHECKS PASS
```

## Reproduce from scratch

Serve Cue v1 first (`cue serve`, see the [main README](../README.md)), then, from the repo root:

```bash
git clone https://github.com/livekit/eot-bench
pip install -e eot-bench -e ".[voice]" pandas pyarrow torch torchaudio transformers   # the last three: Smart Turn's step
# the Gemma part, every turn of every language
python eot-results/reproduce/run.py --harness eot-bench --rows 0 --lang en ar de es fr hi id it ja ko nl pt tr zh
# Smart Turn v3.2 on the same moments
python eot-results/reproduce/run.py --harness eot-bench --rows 0 --lang en ar de es fr hi id it ja ko nl pt tr zh \
    --reference eot_harness.smart_turn_adapter:SmartTurnAudioAdapter
# combine (Gemma part x sqrt Smart Turn), rank against every published model, draw the heatmaps
python eot-results/reproduce/leaderboard.py --harness eot-bench
```

To run another prompt, such as the production one above, add `--decision eot-results/prompts/end_of_turn_production.json`.
Each run prints its model calls and how many failed. A failed call scores 0, so check that it says `(0 failed)`.
llama.cpp is not bit-exact when it serves several requests at once: a rerun matches the published scores to within
about 0.003 on average, which does not move the ranks.

## The benchmark

[LiveKit's eot-bench](https://livekit.com/benchmarks/eot-bench): a caller pauses; did their turn end? Each model gives
a score at every scored moment of every pause, and the harness sweeps thresholds to find two trade-offs:

- **Latency at a 5% / 10% false cut-off rate:** tune the model so at most 5 (or 10) callers in 100 get cut off; how long
  after the caller actually finished does it decide? Lower is better.
- **Cut-offs at a 300 / 600 ms latency budget:** make the model decide within 300 (or 600) ms; what share of callers
  get cut off? Lower is better.

These latencies measure *when* the model decides, not how fast the hardware is (see [Speed and cost](#speed-and-cost)).
Words are assumed to arrive 0.5 s after they are spoken, for the ASR and the decision. We score every caller turn:
400 in English, and every turn of 13 more languages.

## How we kept ourselves honest

- **A held-out English set, used once.** English turns were shuffled with seed 7. Every choice (the prompt, the way
  words and voice are combined, the square root) was made on the first **150 turns**. The other **250 were held back**
  and scored once, at the end. Held back: **689 / 494 ms** at 5% / 10% cut-offs; development set: 702 / 468 ms (these
  two are the recipe with the base Gemma, before the fine-tune existed). The recipe did not overfit the turns it was
  tuned on.
- **Other languages were never tuned.** All 13 run on the same English prompt, English examples included. The one
  language-aware step is automatic: where most caller turns end without punctuation in the transcript (Arabic, Hindi,
  Indonesian, Japanese, Korean, Chinese), the "how the transcript ends" input is set to neutral.
- **The fine-tune never saw the benchmark.** It was trained only on simulated conversations (see
  [Training](#training)).
- **Failed model calls are counted.** A failed call scores 0 and the run says how many failed; every published run
  says `(0 failed)`. One early run lost 62% of its calls to a dropped network tunnel; it was discarded and rerun.

## English

| | Latency @5% | Latency @10% | Cut-offs @300 ms | Cut-offs @600 ms | AUC |
|---|---|---|---|---|---|
| LiveKit turn-detector v1 (hosted) | 543 ms | 295 ms | 9.9% | 4.5% | |
| Baton v1 | 577 ms | 350 ms | 12.3% | 4.8% | |
| Soniox stt-rt-preview | 647 ms | 512 ms | – | 5.5% | |
| **Cue v1** | **719 ms** | **532 ms** | **26.8%** | **7.5%** | **0.961** |
| Cue v1, Gemma part only | 750 ms | 552 ms | 34.0% | 7.8% | 0.920 |
| ultraVAD | 899 ms | 663 ms | 27.7% | 11.9% | |
| Smart Turn v3.2 | 1,051 ms | 739 ms | 35.2% | 14.8% | |
| LiveKit turn-detector v1-mini (local) | 1,070 ms | 698 ms | 27.8% | 12.1% | |
| Deepgram Flux | 1,151 ms | 548 ms | 12.9% | 9.9% | |

Cue v1 is **4th of 13 on all four views**.

**Where Cue v1 matches LiveKit's hosted model.** Below about 600 ms LiveKit v1 is clearly better. From about 800 ms
the two are close, and at one second they are equal:

| If the agent decides within… | LiveKit v1 cuts off | Cue v1 cuts off |
|---|---|---|
| 300 ms | 9.9% | 26.8% |
| 600 ms | 4.5% | 7.5% |
| 800 ms | 3.0% | 3.7% |
| **1,000 ms** | **1.8%** | **1.8%** |
| 1,200 ms | 1.4% | 1.3% |

## 14 languages

Each cell is **Gemma part only → Cue v1**, with Cue v1's rank among the models the benchmark reports for that language.

| Language | Latency @5% (ms) | Latency @10% (ms) | Cut-offs @300 ms | Cut-offs @600 ms |
|---|---|---|---|---|
| English | 750 → 719 (4th of 13) | 552 → 532 (4th) | 34.0% → 26.8% (4th) | 7.8% → 7.5% (4th) |
| Portuguese | 800 → **681** (1st of 11) | 583 → 498 (3rd) | 27.4% → 27.7% (3rd) | 9.4% → **6.5%** (2nd) |
| Spanish | 884 → **792** (2nd of 11) | 690 → 603 (3rd) | 40.2% → 29.9% (4th) | 13.5% → 10.1% (3rd) |
| French | 738 → 647 (3rd of 11) | 524 → 449 (3rd) | 23.7% → 24.4% (4th) | 8.1% → **5.2%** (2nd) |
| German | 802 → 786 (3rd of 11) | 548 → 559 (4th) | 22.7% → 22.7% (3rd) | 8.5% → 9.2% (4th) |
| Italian | 924 → 912 (3rd of 11) | 620 → 595 (3rd) | 34.6% → 31.6% (3rd) | 10.4% → 9.9% (3rd) |
| Dutch | 730 → 770 (3rd of 10) | 590 → 554 (4th) | 23.7% → 24.4% (3rd) | 9.6% → 9.4% (4th) |
| Japanese | 567 → 557 (3rd of 10) | 374 → 406 (3rd) | 13.1% → 15.1% (3rd) | 3.9% → 4.6% (3rd) |
| Korean | 593 → **608** (2nd of 9) | 462 → 406 (3rd) | 19.6% → 21.2% (3rd) | 4.7% → **5.5%** (2nd) |
| Chinese | 862 → **838** (2nd of 9) | 645 → 637 (3rd) | 40.1% → 35.5% (3rd) | 11.8% → 11.7% (3rd) |
| Indonesian | 881 → **871** (2nd of 9) | 673 → **598** (2nd) | 41.1% → **34.4%** (2nd) | 13.4% → **9.9%** (2nd) |
| Hindi | 628 → 607 (4th of 10) | 513 → 494 (3rd) | 37.5% → 30.8% (4th) | 6.5% → 5.3% (4th) |
| Turkish | 1015 → 1012 (3rd of 9) | 743 → 692 (3rd) | 41.3% → **33.9%** (2nd) | 14.9% → 11.6% (3rd) |
| Arabic | 829 → 815 (6th of 9) | 685 → 640 (5th) | 51.5% → 39.9% (4th) | 14.8% → 12.4% (4th) |

- **1st in Portuguese** at a 5% cut-off rate (681 ms, ahead of LiveKit v1 and Soniox at 702 ms). Cue v1 beats LiveKit's
  hosted model in three views: that one, and cut-offs at 600 ms in French (5.2% against 5.4%) and Portuguese (6.5%
  against 7.3%).
- Top 4 in 13 of 14 languages at the 5% budget. Arabic is the weak spot (6th).
- Across 14 languages × the 5% and 10% budgets, Cue v1 is faster than OpenAI GPT Realtime 2, VAP and Gradium in 28 of
  28, Smart Turn v3.2 in 26 of 28, ultraVAD and LiveKit v1-mini in 25 of 28, Deepgram Flux in 12 of 18, Soniox in 12 of
  28, and LiveKit v1 in 1 of 28.

## What helped

**1. A careful prompt.** The production end-of-turn prompt ([`prompts/end_of_turn_production.json`](prompts/end_of_turn_production.json):
four options, no examples) against the "v5" prompt, which
spells out what *done* means on a phone call, adds an input for how the transcript ends, and carries 11 worked
examples. English, Gemma words only:

| Prompt | Base Gemma | Fine-tune |
|---|---|---|
| Production prompt, no examples | 1,270 / 849 ms | 948 / 694 ms |
| v5 without its examples | | 910 / 662 ms |
| **v5 with its 11 examples** | 757 / 576 ms | **750 / 552 ms** |

The examples alone are worth about 160 ms. Cue v1 uses the v5 prompt (`cue/prompts/end_of_turn_phone.json`).

**2. Adding the voice, with the square root.** The words decide most; the voice breaks ties. On the 150 development
turns we compared ways to combine the words' p(done) with Smart Turn's score: the product, the mean, the minimum,
voice squared, the voice early and the words late, and the words × √voice. The square root (the voice counts half
as much as the words) was best, and was then fixed for every turn and language. Cue v1 against its Gemma part alone:
faster in 12 of 14 languages at both budgets; fewer cut-offs at 600 ms in 11 of 14. In Dutch and Korean the Gemma part
alone is slightly faster at 5%.

**3. The fine-tune.** On held-out simulated conversations (10% of the conversations, never trained on):

| | Base Gemma 4 E4B | Cue v1's Gemma part |
|---|---|---|
| Exact answer, every held-out question | 38.3% | **83.2%** |
| End of turn: done vs not done | 58.2% | **86.9%** |
| Interruptions: STOP vs CONTINUE (90 questions) | 93.3% | **98.9%** |

The base model almost never answers HOLDING; the fine-tune learned it. On eot-bench, with the v5 prompt and Smart Turn
both ways, the fine-tune is faster than the base Gemma in 11 of 14 languages at both budgets: Portuguese from 4th to
1st, Hindi from 7th to 4th, French 698 → 647 ms. At 5% it is slower in Dutch (688 → 770 ms) and Arabic (794 →
815 ms), and level in Chinese; at 10% it is slower in English (502 → 532 ms) and Chinese (606 → 637 ms). In English alone, the fine-tune and a well-prompted base Gemma tie
(719 / 532 against 723 / 502 ms). The base-Gemma runs are in `runs/*/cue_v1_on_base_gemma/`.

## What didn't help

- **Letting Gemma hear the audio.** Gemma 4 E4B can take audio; giving it the last seconds of the caller's voice with
  the words raised AUC by less than 0.01, not enough to justify the extra audio in every question.
- **Hand-built prosody.** Pitch and energy at the end of the pause (Praat) as an extra input made the Gemma decision
  worse on the development set (890 / 597 against 778 / 570 ms).
- **A different voice model.** VAP's published scores in place of Smart Turn gave a higher AUC but no latency gain
  (733 / 545 against 723 / 502 ms on all 400 English turns). A higher AUC is not lower latency on this benchmark.
- **Unsloth's decision head.** Training Unsloth's `FastDecisionModel` (a scoring head instead of the letter) on the
  same data: 72.2% exact after one epoch against 83.2%, and not servable by llama.cpp.

## Speed and cost

The decision is one forward pass and one token: the option letters' probabilities are read from the logits, and the
long prompt is shared and cached across calls. One NVIDIA T4, llama.cpp, q8_0, the phone prompt (~750 cached tokens,
~70 new tokens per question):

| Questions in flight | Decisions per second | Time per decision (median / 95th percentile) |
|---|---|---|
| 1 | 12.5 | 79 / 96 ms |
| 2 | 13.6 | 145 / 173 ms |
| 4 | 10.5 | 364 / 419 ms |
| 8 | 12.2 | 649 / 718 ms |
| 16 | 13.0 | 1,220 / 1,332 ms |

About 13 decisions a second per T4; beyond two at once, questions queue. On eot-bench Cue asked about six questions
per caller turn (2,357 for 400 English turns). At an assumed six caller turns a minute, that is roughly 20 concurrent
calls per T4 with every decision under 200 ms, about $0.0005 per call-minute at around $0.58 an hour for a T4. This is
an estimate: measure your own call pattern. Smart Turn v3.2 is a small ONNX model that runs on the CPU.

## Interruptions

The second decision, STOP or CONTINUE when the caller speaks over the agent: 98.9% on 90 held-out questions (base
Gemma: 93.3%). Clear interruptions are near-certain ("wait, Tuesday doesn't work": STOP 0.999; "hmm": CONTINUE 1.0).
**A known miss:** "hmm, actually no" comes back CONTINUE (0.94); short objections can pass as backchannels until more
words arrive.

## Training

- **310 conversations written by an LLM** to a written specification: 150 phone calls between a caller and an agent
  and 160 two-person conversations, each on a different topic, with about 10,600 labelled pauses and about 1,000
  listener sounds. The conversations, the specification and the training scripts are not published.
- The mix was shaped by a few aggregate statistics of TurnBench's public dev split (how often a pause is a hold, how
  often a hold follows a complete sentence, how often a turn ends without punctuation). No benchmark audio, transcript
  or label was used as an example.
- LoRA rank 16 on Gemma 4 E4B (4-bit) with Unsloth, the loss on the answer letter only, one epoch, one T4 (about 2.5
  hours), merged and quantised to q8_0 GGUF.

## Where Cue v1 still loses

- **LiveKit's hosted model is ahead** in every language except Portuguese at a 5% cut-off rate (and French and
  Portuguese at 600 ms), by about 180 ms at 5% in English.
- **Fast budgets.** Under ~600 ms, Cue v1 cuts off more callers than the leaders (26.8% at 300 ms in English).
- **Arabic** (6th at 5%), where the base Gemma with the same recipe was faster (794 against 815 ms).
- **The Gemma part reads words**, so it needs an ASR; a GPU keeps each decision in time.
- **Simulated training data.** Real calls are messier; measure on your own calls before relying on the probabilities.
