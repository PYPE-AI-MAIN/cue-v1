# Cue v1

**Turn detection for voice agents.** Cue v1 knows when a caller has finished speaking, and when they are really
interrupting:

| | Answer | What it means | What the agent does |
|---|---|---|---|
| **End of turn** (the caller pauses) | **DONE** | A complete question, answer, request or statement | Reply now |
| | **LIKELY_DONE** | Probably finished, but it trails off | A short grace, then reply |
| | **MID_THOUGHT** | The sentence is left hanging: "my number is nine eight four five..." | Keep listening |
| | **HOLDING** | Telling a story or explaining something: the pause is thinking | Keep listening, longer |
| **Interruptions** (the caller speaks while the agent talks) | **STOP** | A real interruption: a question, a correction, "wait", "hello?" | Stop talking and listen |
| | **CONTINUE** | Only listening: "hmm", "okay", laughter, noise, a fragment | Keep talking |

How it works:

- **The model.** Cue v1 is Gemma 4 E4B, fine-tuned on these questions.
- **One token, no text.** Each answer is read from a single token as probabilities over the states above, so nothing
  is generated. The agent learns not only *whether* to act but *how long to wait*.
- **Words and voice.** The model reads the words and the context. Combined with
  [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3), which hears the voice, as p(done) × √Smart Turn,
  it is **4th on LiveKit's eot-bench in English and 1st in Portuguese**, with one English prompt across 14 languages
  ([results](eot-results/README.md)).

The model is on Hugging Face: [Pype/cue-v1](https://huggingface.co/Pype/cue-v1). This repo gets it running and lets
you use it: from a terminal, from your own code, or in [LiveKit](livekit-integration/README.md).

## Quick start

```bash
pip install "cue-turn @ git+https://github.com/PYPE-AI-MAIN/cue-v1"
brew install llama.cpp          # Cue runs on llama.cpp's llama-server; Linux, NVIDIA and Docker: see "Serve it yourself"
cue demo                        # downloads Cue v1 from Hugging Face once (about 8 GB), starts it, asks five questions
```

What Cue v1 answers in the demo:

| The agent said | The caller said | Cue v1 |
|---|---|---|
| "What's the best number to reach you on?" | "Sure, it's nine eight four five," | MID_THOUGHT: p(done) 0.00, wait 1.75 s |
| "What's the best number to reach you on?" | "It's nine eight four five six seven one two three four." | DONE: p(done) 1.00, wait 0.51 s |
| "How can I help you today?" | "So last week I went to the clinic and the doctor told me that" | MID_THOUGHT: p(done) 0.00, wait 1.75 s |
| "Your appointment is on Tuesday at ten" (still talking) | "wait, Tuesday doesn't work" | STOP (0.998) |
| "Your appointment is on Tuesday at ten" (still talking) | "hmm" | CONTINUE (1.000) |

To keep it running and ask your own questions, serve it in one terminal and ask from another:

```bash
cue serve                       # http://localhost:8090/v1, until Ctrl-C
cue ask "Sure, it's nine eight four five," --agent-said "What's the best number to reach you on?"
cue ask "wait, Tuesday doesn't work" --barge-in --agent-said "Your appointment is on Tuesday at ten"
```

## Use it in your code

```python
from cue import Cue, p_end, serve, wait

server = serve()                  # or skip this and use a running server: Cue("http://localhost:8090/v1")
cue = Cue(server.url)

# the caller paused: are they done?
probs = cue.end_of_turn("Sure, it's nine eight four five,",
                        agent_said="Thanks. What's the best number to reach you on?")
# {'DONE': 0.0, 'LIKELY_DONE': 0.0, 'MID_THOUGHT': 1.0, 'HOLDING': 0.0}
p_end(probs)                      # p(the caller is done) = DONE + LIKELY_DONE
wait(probs)                       # how long to keep listening, blended by the answer: 1.75 s here

# the caller spoke over the agent: stop?
cue.barge_in("wait, Tuesday doesn't work", agent_said="Your appointment is on Tuesday at ten")
# {'STOP': 0.998, 'CONTINUE': 0.002}

# two people talking, not a caller and an agent
cue.dialogue_end_of_turn("I was thinking we could", other_said="What should we do this weekend?")

server.stop()
```

Every method has an async twin (`aend_of_turn`, `abarge_in`, `adialogue_end_of_turn`). `serve()` also works as a
`with` block.

### At each pause, in any voice stack

Ask Cue when your speech recogniser gives you the caller's words and they pause, then act on the answer:

- **Caller and agent:** reply when `p_end(probs)` crosses a threshold you tune on your own calls, for the cut-off rate
  you can accept. For the benchmarked recipe, add Smart Turn v3.2 on the caller's audio (`pip install
  "cue-turn[voice]"`, it runs on the CPU):

  ```python
  from cue import with_voice
  from cue.smart_turn import SmartTurn

  smart_turn = SmartTurn().fresh()       # downloads the model once; one fresh() per call
  smart_turn.push(pcm_16k)               # the caller's audio as it arrives (16 kHz mono)
  p = with_voice(cue.end_of_turn(words, agent_said=last_agent_message), smart_turn.predict())
  ```

- **Two people talking:** wait `wait(probs)` seconds of silence, and use the words alone. In our two-person tests,
  adding Smart Turn made it slightly worse.
- **Interruptions:** ask `cue.barge_in(words, agent_said=...)`, passing what the agent has said so far. Stop when
  p(STOP) ≥ 0.5, and ask again as more words arrive.

## Use it in LiveKit

Cue v1 drops into a [LiveKit Agents](https://github.com/livekit/agents) voice agent as its turn detector:

```python
from cue_livekit import CueTurnDetector

session = AgentSession(..., turn_handling={"turn_detection": CueTurnDetector("http://localhost:8090/v1"),
                                           "endpointing": {"min_delay": 0.3, "max_delay": 2.0}})
```

It also adds Smart Turn on the caller's audio, a fallback for when Cue doesn't answer, and Cue-decided barge-in. The
guide, with code for each: **[livekit-integration/](livekit-integration/README.md)**.

## Serve it yourself

`cue serve` and `serve()` download `cue-v1-q8_0.gguf` from Hugging Face once and start llama.cpp's `llama-server`
with the flags Cue v1 was benchmarked with. Options: `--port`, `--host 0.0.0.0` (to serve other machines),
`--gpu-layers 0` (CPU only: it works, but slower), `--model` (a local GGUF).

**llama-server** comes from llama.cpp:
- **macOS:** `brew install llama.cpp`.
- **Linux and Windows:** the [prebuilt releases](https://github.com/ggml-org/llama.cpp/releases), or a build with
  CUDA for an NVIDIA GPU.
- **Elsewhere:** set `CUE_LLAMA_SERVER` to its path.

To run it without this package, llama-server can fetch the model itself:

```bash
llama-server --hf-repo Pype/cue-v1 --hf-file cue-v1-q8_0.gguf -c 8192 -np 4 -ngl 999 --jinja --port 8090
# or on an NVIDIA GPU, with Docker:
docker run --gpus all -p 8090:8090 ghcr.io/ggml-org/llama.cpp:server-cuda \
    --hf-repo Pype/cue-v1 --hf-file cue-v1-q8_0.gguf -c 8192 -np 4 -ngl 999 --jinja --host 0.0.0.0 --port 8090
```

Any OpenAI-compatible server that returns `logprobs` works too, such as vLLM: point `Cue` at its URL.

**Speed.** One NVIDIA T4 answers about 13 decisions a second: 79 ms each one at a time, under 200 ms two at once.
That is roughly 20 concurrent calls per T4, an estimate ([Speed and cost](eot-results/README.md#speed-and-cost)).

## Prompts

The prompts are part of the model, and they ship inside the package (`cue/prompts/`):

| Prompt | Used by | Question |
|---|---|---|
| `end_of_turn_phone` | `end_of_turn` | End of turn, a caller and an agent (the eot-bench prompt, with 11 English examples) |
| `end_of_turn_dialogue` | `dialogue_end_of_turn` | End of turn, two people talking |
| `barge_in` | `barge_in` | Interruptions: STOP or CONTINUE while the agent is speaking |

Each is a JSON decision: an instruction, the options and what they mean, the inputs it reads, and worked examples.
`cue.messages(decision, inputs)` renders it exactly as Cue v1 was trained and benchmarked; `tests/` check this byte
for byte. To ask with your own variant, use `cue.ask(cue.load("my_prompt.json"), inputs)`.

## Results

On LiveKit's eot-bench, all 400 English caller turns and every turn in 13 more languages, scored with the benchmark's
own harness:

| English | Latency @5% | Latency @10% | Cut-offs @300 ms | Cut-offs @600 ms |
|---|---|---|---|---|
| LiveKit turn-detector v1 (hosted) | 543 ms | 295 ms | 9.9% | 4.5% |
| **Cue v1** | **719 ms** | **532 ms** | **26.8%** | **7.5%** |
| Smart Turn v3.2 | 1,051 ms | 739 ms | 35.2% | 14.8% |

Cue v1 is **4th of 13** in English, **1st in Portuguese**, and top 4 in 13 of the 14 languages. LiveKit's hosted model
still leads everywhere but Portuguese. [`eot-results/`](eot-results/README.md) has the full table:
- every language and every view;
- what helped and what didn't;
- every saved prediction, with one script that re-checks all the numbers using LiveKit's own harness, with no GPU.

## Limits

- **The Gemma part reads words, not prosody.** That is what Smart Turn adds.
- **Short objections can pass as backchannels.** "Hmm, actually no" said over the agent comes back CONTINUE (0.94).
- **Fast budgets.** Under about 600 ms, Cue v1 cuts off more callers than the leaders (26.8% at 300 ms in English).
- **Needs a speech recogniser and a GPU.** It is not an on-device model.
- **Trained on English.** The other 13 languages run on the English prompt; Arabic is the weakest (6th).
- **Trained on simulated conversations.** Measure it on your own calls before relying on its probabilities.

## Part of Shortwire

Cue is the turn engine in **Shortwire**, Pype's open source voice AI stack. There, background decision models
handle how the call moves (when the caller is done, when to hold, when to stop talking), and the LLM focuses on
what the agent says. **Cue v1 works best with Shortwire.** We'll share more details soon.

## Licence

Apache 2.0 (`LICENSE`), the same as Gemma 4 E4B. See `NOTICE`. To cite Cue v1, see `CITATION.cff`.
