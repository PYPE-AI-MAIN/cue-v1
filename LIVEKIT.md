# Cue v1 in LiveKit Agents

Cue v1 plugs into a [LiveKit Agents](https://github.com/livekit/agents) voice agent as its **turn detector**. Each
time the caller pauses, LiveKit asks how likely it is that they have finished. Cue answers from the caller's words and
the agent's last question:

- A half-said phone number gets a longer wait.
- A finished request gets a short one.

There are two optional additions:

| You add | What it does | Code |
|---|---|---|
| [Words only](#1-words-only-a-drop-in-turn-detector) | Cue decides how long LiveKit waits before replying | 2 lines |
| [Smart Turn v3.2](#2-add-smart-turn-v32-the-benchmarked-recipe) | Adds how the caller *sounds*: p(done) × √Smart Turn, the recipe Cue v1 is benchmarked with | + an `stt_node` |
| [Barge-in](#3-let-cue-decide-barge-in) | When the caller speaks over the agent, Cue tells "wait, which doctor?" (stop) from "hmm, okay" (keep talking) | + a `tts_node` |

Everything here is in `cue/livekit.py` and `cue/smart_turn.py`. A complete agent with all three switches is in
[`examples/livekit_agent.py`](examples/livekit_agent.py).

Built and tested on **livekit-agents 1.8.1**. Below, "she" is the agent and "the caller" is the person on the line.

## 0. Serve Cue and install

Cue runs on a GPU behind any OpenAI-compatible server that returns `logprobs`. With llama.cpp:

```bash
huggingface-cli download Pype/cue-v1 cue-v1-q8_0.gguf --local-dir .
llama-server -m cue-v1-q8_0.gguf -c 8192 -np 4 -ngl 999 --jinja --port 8090
```

`-np` is how many questions llama.cpp answers at once. A call has at most two out at a time: an end-of-turn question
and, with barge-in, a barge-in question. Each takes well under a second, so a few slots serve many calls. The README
estimates about 20 concurrent calls per T4 ([Speed and cost](RESULTS.md#speed-and-cost)).

Install the client with the LiveKit extra (`livekit-agents[silero]`, `numpy`, `onnxruntime`, `huggingface_hub`), plus
the plugins your agent uses (the examples here use OpenAI's):

```bash
pip install "cue-turn[livekit] @ git+https://github.com/PYPE-AI-MAIN/cue-v1"    # or, from a clone: pip install -e ".[livekit]"
pip install "livekit-agents[openai]~=1.8"
```

Check the server answers. This should print a p(done) close to 1 for a full phone number:

```bash
python -c "
from cue import Cue, ending, load, p_end
from cue.livekit import prompt
said = 'It is nine eight four five six seven one two three four.'
probs = Cue('http://localhost:8090/v1').ask(load(prompt('end_of_turn_phone')), {'agent_question': 'What is your number?',
    'caller_earlier': '', 'caller_text': said, 'transcript_end': ending(said)})
print(p_end(probs))"
```

From a clone, `python examples/quickstart.py` runs a few more.

## 1. Words only: a drop-in turn detector

```python
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, silero

from cue.livekit import CueTurnDetector


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    detector = CueTurnDetector("http://localhost:8090/v1")      # one per call
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(), llm=openai.LLM(model="gpt-4o-mini"), tts=openai.TTS(),   # any plugins
        turn_handling={
            "turn_detection": detector,
            "endpointing": {"min_delay": 0.3, "max_delay": 2.0},
        },
    )
    ctx.add_shutdown_callback(detector.aclose)
    await session.start(agent=Agent(instructions="You are a friendly clinic receptionist."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

A detector holds one call's state. Create it inside the entrypoint, once per call, and never share it between calls.

### How it works

Once Silero has decided the caller stopped speaking and there is a transcript, LiveKit hands the detector the
conversation, with the caller's new words as the last message. The detector asks Cue its end-of-turn question with
these inputs:

| Input | Taken from |
|---|---|
| `agent_question` | Her messages since the caller last spoke: the last sentence of her latest one if it reads as a question, else its last 200 characters. Empty if the caller has spoken twice in a row |
| `caller_earlier` | The caller's previous message (its last 300 characters) |
| `caller_text` | The caller's words this turn |
| `transcript_end` | How the STT ended them: `a question mark`, `a full stop` or `no sentence end` |

This is the mapping Cue v1 was benchmarked with on eot-bench, for an STT that punctuates. If your STT seldom
punctuates the caller's language, pass `punctuated=False`: the input then says so instead of "no sentence end", as
eot-bench did for Arabic, Hindi, Indonesian, Japanese, Korean and Chinese.

Cue answers DONE, LIKELY_DONE, MID_THOUGHT or HOLDING, read as probabilities from one token. The detector returns
**p(done) = DONE + LIKELY_DONE**:

- below `threshold` (default 0.5), LiveKit waits `max_delay` for more words;
- otherwise it waits `min_delay` and replies.

**Where the time goes.** LiveKit counts both delays from the end of the caller's speech, but it only asks the detector
after Silero's end of speech (`min_silence_duration`, 0.55 s by default). A `min_delay` of 0.3 s has passed by then,
so on the "likely done" path Cue's decision time adds to the reply: 79 ms one at a time on a T4, under 200 ms two at
once, plus the network. On the "not done" path it
comes out of `max_delay`. If Cue has not answered within `timeout` (default 1 s), the turn is treated as done, and
that turn's reply comes up to `timeout` later than without a detector.

**Logs.** Every decision is logged on the `cue.livekit` logger at INFO level:

```
cue: end_of_turn MID_THOUGHT p(done)=0.02 -> wait (158 ms) 'My number is 98450.'
cue: end_of_turn DONE p(done)=0.98 -> reply (154 ms) 'My number is 98450. 1 2 3 4 5.'
```

The last answered decision is kept in `detector.last_end_of_turn` (the option, the four probabilities, Smart Turn's
score, p and the time). The last barge-in decision is in `detector.last_barge_in`.

### Options

```python
from cue.livekit import CueTurnDetector, prompt

CueTurnDetector(
    "http://localhost:8090/v1",
    model="cue", api_key="x",            # whatever your server expects
    end_of_turn=prompt("end_of_turn_dialogue"),  # two people talking; default end_of_turn_phone (caller and agent)
    threshold=0.5,                       # below it LiveKit waits max_delay
    timeout=1.0,                         # seconds; no answer in time = done
    punctuated=True,                     # False: your STT seldom punctuates the caller's language
    languages=["en", "hi"],              # only these, in any form ("hi-IN", "english"); default all
    graded_waits=False,                  # True: Cue's own waits when the caller is not done (see below)
)
```

- **The dialogue prompt** reads her last message and the caller's words, as it was trained. For two people talking,
  use it with `graded_waits=True` and without Smart Turn: in our two-person tests, adding Smart Turn made it slightly
  worse.
- **Your own prompt JSON** (in the format of `prompts/`) must read one of the two shipped input sets. These are the
  phone prompt's `agent_question`, `caller_earlier`, `caller_text` and `transcript_end`, or the dialogue prompt's three.
  The detector raises `ValueError` for anything else.
- **Languages.** Cue v1 ran 14 languages on its English prompt on eot-bench. For a language outside `languages`, the
  turn is treated as done (`min_delay`). With barge-in on, barge-in still runs for those languages.

## 2. Add Smart Turn v3.2 (the benchmarked recipe)

Cue reads words, not prosody. Its eot-bench ranking is for Cue's p(done) × √(Smart Turn v3.2's score on the caller's
audio). To score turns the same way in LiveKit:

1. Load the Smart Turn model once per process.
2. Give each call's detector its own `fresh()` copy.
3. Route the caller's audio through the detector in your agent's `stt_node`.

```python
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import openai, silero

from cue.livekit import CueTurnDetector
from cue.smart_turn import SmartTurn


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["smart_turn"] = SmartTurn()      # the model, once per process


class Assistant(Agent):
    def __init__(self, detector: CueTurnDetector) -> None:
        super().__init__(instructions="You are a friendly clinic receptionist.")
        self.cue = detector

    async def stt_node(self, audio, model_settings):
        # the caller's audio passes through the detector on its way to the STT, unchanged
        async for event in Agent.default.stt_node(self, self.cue.listen_audio(audio), model_settings):
            yield event


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    detector = CueTurnDetector("http://localhost:8090/v1", threshold=0.3,
                               smart_turn=ctx.proc.userdata["smart_turn"].fresh())   # this call's own audio buffer
    session = AgentSession(
        vad=ctx.proc.userdata["vad"], stt=openai.STT(), llm=openai.LLM(model="gpt-4o-mini"), tts=openai.TTS(),
        turn_handling={"turn_detection": detector, "endpointing": {"min_delay": 0.3, "max_delay": 2.0}},
    )
    detector.attach(session)                       # tells Smart Turn when a new turn of the caller's audio starts
    ctx.add_shutdown_callback(detector.aclose)
    await session.start(agent=Assistant(detector), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
```

**The model.** `SmartTurn()` downloads `smart-turn-v3.2-cpu.onnx` (int8) from `pipecat-ai/smart-turn-v3` on Hugging
Face into its cache. Each later load still checks the Hub; set `HF_HUB_OFFLINE=1` once the file is cached, or pass a
path: `SmartTurn("model.onnx")`. Our eot-bench scores used the full-precision export, `smart-turn-v3.2-gpu.onnx`; pass
its path for the closest match. We have not compared the two exports' scores. The full-precision export takes about
120 ms per score on one thread of an Apple-silicon laptop. The int8 file was not timed here.

**The audio window.** Smart Turn scores the last 8 s of the caller's audio, resampled to 16 kHz mono, at each pause.
It runs on the CPU beside the Cue question, not after it, and the preprocessing follows the model's reference
inference. The window starts again:

- when she starts a new speech, unless the caller is talking at that moment;
- after words the caller says over her that barge-in lets pass (they are not part of the caller's next turn).

**Threshold.** Multiplying by √Smart Turn lowers the score, so start the threshold lower than 0.5. **0.3 is a starting
point, not a measured optimum**: tune it on your own calls.

To show how the product works, here is a wiring test. Cue was a stand-in server with fixed answers: DONE, or
MID_THOUGHT for one to nine digits, both at 0.98. Smart Turn was the full-precision export.

- The request "I want to book an appointment. I have had knee pain since last week." scored 0.98 × √0.97 = 0.97.
- "My number is 98450." scored 0.02 × √0.99 = 0.02.

Before a fix to the audio window, the same request clip scored 0.37 from Smart Turn (p = 0.60). These numbers show the
arithmetic, not how well Cue v1 does.

## 3. Let Cue decide barge-in

What LiveKit 1.8.1 does on its own when the caller speaks over her:

- **Self-hosted:** it stops her after 0.5 s of the caller's speech, so a "hmm" or "okay" that long stops her too. A
  sound with no words (a cough) pauses her, and she resumes after 2 s.
- **LiveKit Cloud, and dev or console mode:** its adaptive interruption tries to ignore backchannels.

With `barge_in=True`, Cue decides instead, from the caller's words and what she has said, the same way wherever it
runs:

- **STOP:** a question, a correction, "wait", "hello?", a full answer.
- **CONTINUE:** listening sounds, laughter, noise, a fragment.

It needs four things. It also needs every speech to be uninterruptible in LiveKit's eyes. Don't pass
`allow_interruptions=True` to `say()` or `generate_reply()`, and don't turn interruptions on for an `Agent`. For such
a speech, LiveKit's own interruption stays on: it stops her after 0.5 s of sound and answers the words Cue let pass.
The detector logs a warning the first time it sees one.

```python
detector = CueTurnDetector("http://localhost:8090/v1", barge_in=True)

session = AgentSession(
    vad=silero.VAD.load(), stt=..., llm=..., tts=...,
    turn_handling={
        "turn_detection": detector,
        "endpointing": {"min_delay": 0.3, "max_delay": 2.0},
        # 1. Cue decides interruptions, so LiveKit's are off; and the STT keeps hearing the caller while she speaks
        #    (by default LiveKit feeds it silence during a speech it may not interrupt)
        "interruption": {"enabled": False, "discard_audio_if_uninterruptible": False},
    },
)
detector.attach(session)        # 2. follows which of her speeches is playing, and the caller's interim transcripts


class Assistant(Agent):
    def __init__(self, detector: CueTurnDetector) -> None:
        super().__init__(instructions="You are a friendly clinic receptionist.")
        self.cue = detector

    async def tts_node(self, text, model_settings):
        # 3. her words pass through the detector on their way to the TTS: barge-in reads what she has said so far
        async for frame in Agent.default.tts_node(self, self.cue.listen_text(text), model_settings):
            yield frame

# 4. await session.start(agent=Assistant(detector), ...) and ctx.add_shutdown_callback(detector.aclose), as above
```

To use Smart Turn as well, give the agent both `stt_node` and `tts_node`, as `examples/livekit_agent.py` does.

### What happens

| When | Cue is asked | Then |
|---|---|---|
| The caller speaks over her, and the STT sends interim transcripts | barge-in, on the latest words, one question at a time | **STOP**: she is stopped, her queued speeches are dropped, and LiveKit takes the words as an ordinary turn. **CONTINUE**: she keeps talking |
| LiveKit asks about the caller's words while she is still talking | barge-in, on the final transcript | **STOP**: as above, then the end-of-turn question. **CONTINUE**: the detector reports the turn as done at once, so LiveKit commits it while she is still speaking and drops it: no reply, not in the chat history |
| LiveKit asks before her next speech has made a sound (the LLM still writing, the TTS still synthesising, a greeting, a tool's reply, a speech queued behind one that just ended) | end-of-turn only | that speech and any queued behind it are dropped, and LiveKit answers everything the caller said |
| The caller speaks while a tool runs after she has spoken | end-of-turn only | as without barge-in: LiveKit answers, and the tool's result is still spoken |
| The caller speaks while she is silent | end-of-turn only | as in section 1 |

LiveKit asks about the caller's words only once Silero has heard about 0.55 s of silence and the final transcript is
in. So words that end shortly before she finishes are asked about after she has finished. They get only the
end-of-turn question and become a turn of their own: a "hmm" in her last half second or so gets a reply. In our
sessions the final came about 0.5 s after the words with Sarvam, and about 1.2 s with ElevenLabs.

A tool's speech still working in the background is never stopped, so its result is still spoken after LiveKit's
answer.

**What she has said so far.** The barge-in question includes her words, but only the part she has actually spoken.
LiveKit sends the TTS the whole reply as fast as the LLM writes it. So the detector estimates the spoken part from
how long the speech has been playing × `chars_per_s` (default 14 characters a second; set it to your voice's rate).
Otherwise a "yes" over "Dr. Rao is free on Tuesday..." would read as the answer to "Shall I book that?", which she
hasn't reached yet. A tool's reply counts as a new speech, though LiveKit reuses the speech handle for it. A speech
sent to the TTS in segments (an `llm_node` that yields `FlushSentinel`) is read in order. A speech whose text doesn't
pass through the `tts_node` (`session.say(text, audio=...)`) reads as nothing said.

**If Cue doesn't answer in time**, three or more words stop her and fewer don't (`stop_without_answer=3`). Words are
counted by spaces, so "ठीक है" is two. A language written without spaces (Chinese, Japanese) counts as one word.

**How soon she stops.** Cue needs the words, so she stops later than she would with LiveKit's sound-based
interruption. How much later depends on your STT. In our test sessions, the caller said "Wait, which doctor did you say?"
(1.8 s of audio) over her:

| STT | Interim transcripts | She stopped after the caller began |
|---|---|---|
| ElevenLabs Scribe v2 realtime (`vad_silence_threshold_secs=0.3`) | yes, about one a second | 1.3 to 2.2 s in three runs: on the interim "Wait." in one, on "Wait, which doctor did" in the others |
| Sarvam saaras:v4 (`mode="transcribe"`, `negative_frames_count=8`) | none, only the final transcript | 2.4 s in every run |

Sarvam's default server VAD sends the final about 0.3 s later than this setting, and she stops correspondingly later.
Cue itself was a stand-in server here, answering STOP on "wait". With Cue v1, how early she stops also depends on
which of the first words Cue already reads as STOP.

## Thresholds and delays

| Setting | Start with | Meaning |
|---|---|---|
| `threshold` | 0.5 words only; 0.3 with Smart Turn | Below it, LiveKit waits `max_delay` |
| `endpointing.min_delay` | 0.3 s | Silence before replying when Cue says the caller is likely done (already passed when Cue is asked, see above) |
| `endpointing.max_delay` | 2.0 s | Silence before replying when not. Cue's own waits are 1.75 s for MID_THOUGHT and 2.25 s for HOLDING (`cue.WAITS`) |
| `timeout` | 1.0 s | A later answer counts as none |
| `chars_per_s` | 14 | Her speaking rate, for barge-in |

LiveKit's turn-detector interface has two waits, so Cue's four answers collapse to one probability against one
threshold: MID_THOUGHT and HOLDING both get `max_delay`.

For Cue's own graded waits, pass `graded_waits=True` and call `detector.attach(session)`. When Cue says the caller is
not done, the detector sets LiveKit's `max_delay` to `cue.wait(probs)` before answering: 1.75 s for MID_THOUGHT and
2.25 s for HOLDING, blended by the probabilities. It works because LiveKit 1.8.1 reads `max_delay` after the detector
answers. It needs the default fixed endpointing mode, and it was checked only on 1.8.1.

## What has been tested, and what hasn't

**Tested:**

- **Unit tests.** `tests/test_livekit.py` (29 tests) runs against a fake Cue server, LiveKit's own `ChatContext`, and
  a fake session that follows LiveKit 1.8.1's speech handles. It covers:
  - the inputs and request body for both prompts;
  - p(done) against the threshold;
  - the timeout and failure fallbacks;
  - Smart Turn's √ weighting, its parallel run and its failure fallback;
  - the language filter;
  - barge-in on final and interim transcripts;
  - reading only what she has said;
  - each of the barge-in situations in the table above, a tool's reply, queued speeches, segments and graded waits;
  - resampling 48 kHz stereo audio for Smart Turn.

  Run them with `pip install -e ".[livekit,test]" && pytest`.
- **Real LiveKit 1.8.1 sessions**, with a scripted LLM and a real-time audio output, one per tricky situation:
  - a tool running after she has spoken;
  - a tool's reply, being written and playing;
  - a reply queued behind a greeting that is still playing;
  - a greeting still synthesising, with a reply queued behind it;
  - a reply still being written;
  - a STOP while a tool works in the background;
  - her speech ending, with another queued, while Cue is answering;
  - a reply sent to the TTS in segments;
  - a `say()` with recorded audio;
  - a backchannel in the last moments of her speech.
- **Scripted live calls.** These used Silero VAD, an STT (Sarvam saaras:v4 or ElevenLabs Scribe v2 realtime, set as
  in the stop-time table above), Gemma 4 E4B for her replies and ElevenLabs TTS. The caller was played from audio clips in real time, with `aec_warmup_duration=None` because a
  simulated line has no echo. To test the wiring, Cue was a stand-in server with fixed answers:
  - "wait", or three words or more: STOP;
  - two words or fewer: CONTINUE;
  - one to nine digits: MID_THOUGHT;
  - anything else: DONE.

  The checks:
  - a "हाँ हाँ" over her greeting was dropped, and the greeting went on;
  - the request was answered;
  - "Wait, which doctor did you say?" over her reply stopped her, and she answered it;
  - "My number is 98450." waited, and "1 2 3 4 5." joined it in one turn.

  | Setup | Checks passed (last run, final code) | Notes |
  |---|---|---|
  | Sarvam STT | 8 of 9 | Every behaviour check passed. The miss is the harness's own "exactly one STOP" check: the request arrived as two finals, she began answering the first, and the second rightly got a STOP of its own. Earlier runs passed all 9 |
  | Sarvam STT with Smart Turn | 9 of 9 | An earlier run missed one: the STT dropped "had knee pain" from the request, which was still answered |
  | Sarvam STT, base Gemma 4 E4B as Cue | 8 of 8 | Cue v1's base model, not the fine-tune, served by llama.cpp on a T4 in Mumbai with the same prompts. Decisions took 121 to 560 ms across three runs, including the network |
  | ElevenLabs STT | 9 of 9 | Earlier runs passed 7 of 9: ElevenLabs connection resets on our account lost words or delayed a final by seconds, plus the same "one STOP" check |

**Not measured:**

- **Cue v1 itself inside LiveKit.** Its published numbers are offline eot-bench scores of p(done) × √Smart Turn on
  recorded turns.
- **The words-only setup** on any benchmark.
- **The thresholds above.**
- **The barge-in decision** on any public set.

Measure on your own calls before relying on any of them.

## Limits

- **LiveKit's turn-detector interface is internal.** It is the `_TurnDetector` protocol in `livekit.agents.voice.turn`.
  Barge-in also reads which speech a `tts_node` belongs to from LiveKit internals. So the extra pins
  `livekit-agents<1.9` until a newer version is checked.
- **With barge-in, every speech can be stopped by Cue**, including ones you start with `allow_interruptions=False`.
  A speech that allows interruptions is the opposite problem: LiveKit stops it on sound alone (see section 3).
- **A small window stays open.** After the detector answers, LiveKit may wait a little longer before it commits the
  turn. A queued speech of hers that becomes current in that window is not seen, and LiveKit skips the caller's turn.
- **The first 3 s she speaks are not heard.** LiveKit feeds the STT silence during them (`aec_warmup_duration`, for
  echo cancellation), so nothing the caller says then reaches Cue. On lines without echo, pass
  `aec_warmup_duration=None` to `AgentSession`, as LiveKit already does for outbound SIP calls.
- **Not for realtime (speech-to-speech) models with server-side turn detection.** They don't use a turn detector.
- **Cue needs a GPU server.** One T4 with llama.cpp (q8_0) takes 79 ms per decision one at a time and under 200 ms two
  at once, plus the network.
