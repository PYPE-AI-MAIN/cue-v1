"""Cue v1 in a LiveKit Agents voice agent (built and tested on livekit-agents 1.8.1).

LiveKit takes a turn detector in ``turn_handling["turn_detection"]``: an object it asks, once the caller has paused
and there are words, for p(the turn is over). Below its threshold, LiveKit waits ``endpointing.max_delay`` before
replying; otherwise ``min_delay``. CueTurnDetector answers that from Cue v1's end-of-turn decision:

    detector = CueTurnDetector("http://localhost:8090/v1")
    session = AgentSession(..., turn_handling={"turn_detection": detector,
                                              "endpointing": {"min_delay": 0.3, "max_delay": 2.0}})

Two optional parts, each a few lines in your Agent (see LIVEKIT.md):

- **Smart Turn v3.2 on the caller's audio** (``smart_turn=SmartTurn()``): p(done) x sqrt(Smart Turn), the recipe Cue v1
  is benchmarked with. The detector hears the caller's audio through ``listen_audio`` in your ``stt_node``.
- **Barge-in** (``barge_in=True``): when the caller speaks over her, Cue's STOP / CONTINUE decision says whether she
  stops or the words were only "hmm, okay". It is asked on each interim transcript as the words come in (if your STT
  sends them) and on the final one. It needs LiveKit's own interruptions off, the STT listening while she speaks,
  ``detector.attach(session)``, and what she says through ``listen_text`` in your ``tts_node``. Words that end
  before her next speech has made a sound (the LLM still writing, the TTS still synthesising) replace that speech,
  so LiveKit answers them. Every speech must be uninterruptible in LiveKit's eyes (no ``allow_interruptions=True``).

A detector holds one call's state: make one per call (per AgentSession), never share it between calls.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any, AsyncIterable, AsyncIterator

from .client import Cue, p_end, wait, with_voice
from .decision import UNPUNCTUATED, Decision, ending, last_question, load

log = logging.getLogger("cue.livekit")

_PROMPT_DIRS = (Path(__file__).resolve().parent / "prompts", Path(__file__).resolve().parents[1] / "prompts")

# the end-of-turn inputs this adapter fills: the phone prompt's four and the dialogue prompt's three
PHONE_INPUTS = ("agent_question", "caller_earlier", "caller_text", "transcript_end")
DIALOGUE_INPUTS = ("Other person last said", "Speaker said", "Speaker's words end with")


def prompt(name: str) -> Path:
    """A prompt shipped with Cue v1 (``end_of_turn_phone``, ``end_of_turn_dialogue``, ``barge_in``)."""
    for d in _PROMPT_DIRS:
        p = d / f"{name}.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}.json: pass its path (the prompts are in the repo and on the Hugging Face page)")


def _messages(chat_ctx: Any) -> list[dict[str, str]]:
    """The user and assistant messages of a LiveKit ChatContext (or a list of {"role", "content"} dicts)."""
    items = getattr(chat_ctx, "items", chat_ctx)
    out = []
    for it in items:
        if isinstance(it, dict):
            role, text = it.get("role"), it.get("content")
        else:
            if getattr(it, "type", "message") != "message":
                continue
            role, text = getattr(it, "role", None), getattr(it, "text_content", None)
        if role in ("user", "assistant") and text:
            out.append({"role": role, "content": str(text)})
    return out


def end_of_turn_inputs(chat_ctx: Any, *, punctuated: bool = True) -> dict[str, str] | None:
    """Cue's end-of-turn inputs from what LiveKit hands a turn detector: the conversation, with the caller's new
    words as the last user message. None: no words.

    The phone prompt's inputs are mapped as on eot-bench: ``agent_question`` from her messages after the caller's
    previous message (empty when the caller has spoken twice in a row), ``caller_earlier`` the caller's previous
    message. ``punctuated=False`` (an STT that seldom punctuates the language) says so instead of "no sentence end",
    as eot-bench did for ar, hi, id, ja, ko and zh. The dialogue prompt's inputs are mapped as in its training: her
    last message and the caller's words."""
    msgs = _messages(chat_ctx)
    if not msgs or msgs[-1]["role"] != "user":
        return None
    caller = msgs[-1]["content"].strip()
    if not caller:
        return None
    question, earlier = "", ""
    for m in reversed(msgs[:-1]):
        text = m["content"].strip()
        if m["role"] == "assistant" and not question:
            question = last_question(text) or text[-200:]
        elif m["role"] == "user":
            earlier = text[-300:]
            break
    other = next((m["content"].strip() for m in reversed(msgs[:-1]) if m["role"] == "assistant"), "")
    end = ending(caller)
    return {"agent_question": question, "caller_earlier": earlier, "caller_text": caller,
            "transcript_end": UNPUNCTUATED if not punctuated and end == "no sentence end" else end,
            "Other person last said": other[-240:], "Speaker said": caller[-400:], "Speaker's words end with": end}


def _language(code: Any) -> str:
    """The base language of a code in any common form: "en-IN", "english", "cmn-Hans-CN" -> "en", "en", "zh"."""
    try:
        from livekit.agents.language import LanguageCode
        return LanguageCode(str(code)).language
    except Exception:
        return str(code).split("-")[0].split("_")[0].lower()


def _speech_handle() -> Any:
    """The LiveKit speech a tts_node is synthesising (None if this LiveKit version does not say)."""
    try:
        from livekit.agents.voice.agent_activity import _SpeechHandleContextVar
        return _SpeechHandleContextVar.get(None)
    except Exception:
        return None


class CueTurnDetector:
    """Cue v1 as LiveKit's turn detector, optionally with Smart Turn v3.2 and Cue's barge-in decision.

    ``threshold``: below it LiveKit waits ``max_delay``. 0.5 for words alone; with Smart Turn the score is p(done) x
    sqrt(Smart Turn), so start lower (0.3) and tune on your calls. ``timeout``: an answer later than this counts as
    no answer, and a turn with no answer is treated as done. ``languages``: the only languages Cue decides for (any
    form: "en", "hi-IN", "english"); for others the turn is treated as done (and barge-in still runs).
    ``graded_waits``: when Cue says the caller is not done, wait ``cue.wait(probs)`` (MID_THOUGHT 1.75 s, HOLDING
    2.25 s, blended) instead of LiveKit's one ``max_delay``; needs ``attach(session)`` and LiveKit's fixed endpointing
    mode (the default), and relies on LiveKit 1.8.1 reading ``max_delay`` after the detector answers."""

    def __init__(self, base_url: str = "http://localhost:8090/v1", *, model: str = "cue", api_key: str = "x",
                 end_of_turn: str | Path | None = None, barge_in: bool | str | Path = False,
                 threshold: float = 0.5, smart_turn: Any = None, timeout: float = 1.0, languages: Any = None,
                 punctuated: bool = True, graded_waits: bool = False, chars_per_s: float = 14.0,
                 stop_without_answer: int = 3) -> None:
        self.cue = Cue(base_url, model, api_key, timeout=max(timeout, 1.0) + 1.0)
        self.eot: Decision = load(end_of_turn or prompt("end_of_turn_phone"))
        unknown = [n for n in self.eot.inputs if n not in PHONE_INPUTS + DIALOGUE_INPUTS]
        if unknown:
            raise ValueError(f"{self.eot.name} reads inputs this detector cannot fill: {unknown}. "
                             f"Use the phone prompt's {list(PHONE_INPUTS)} or the dialogue prompt's "
                             f"{list(DIALOGUE_INPUTS)}.")
        self.barge: Decision | None = None
        if barge_in:
            self.barge = load(prompt("barge_in") if barge_in is True else barge_in)
        self.threshold, self.timeout, self.punctuated, self.graded_waits = threshold, timeout, punctuated, graded_waits
        self.smart_turn = smart_turn
        self.languages = {_language(x) for x in languages} if languages else None
        self.chars_per_s, self.stop_without_answer = chars_per_s, stop_without_answer
        self.session: Any = None
        self._language: Any = None               # the language LiveKit last asked about
        # what of hers is playing: a speech handle and its step (a tool's reply is the same handle's next step),
        # since when, and the text sent to the TTS for each (handle id, step)
        self._audible: Any = None
        self._audible_step = 0
        self._audible_since = 0.0
        self._texts: dict[Any, str] = {}
        self._warned_interruptible = False
        # barge-in on interim transcripts: the caller's latest words over her, and the task asking about them
        self._heard = ""
        self._barge_task: asyncio.Task | None = None
        self.last_end_of_turn: dict[str, Any] = {}    # the last answered end-of-turn decision, for your logs
        self.last_barge_in: dict[str, Any] = {}       # the last barge-in decision

    # -- LiveKit's turn detector interface -----------------------------------------------------------
    @property
    def model(self) -> str:
        return "cue-v1"

    @property
    def provider(self) -> str:
        return "pype"

    async def unlikely_threshold(self, language: Any = None) -> float:
        return self.threshold

    async def supports_language(self, language: Any = None) -> bool:
        """Cue v1 was trained on English; on eot-bench it ran on 14 languages with the English prompt. With barge-in
        on, always True: LiveKit only calls predict_end_of_turn for a supported language, and barge-in lives there."""
        self._language = language
        return self.barge is not None or self._decides(language)

    def _decides(self, language: Any) -> bool:
        return self.languages is None or language is None or _language(language) in self.languages

    async def predict_end_of_turn(self, chat_ctx: Any, *, timeout: float | None = None) -> float:
        inputs = end_of_turn_inputs(chat_ctx, punctuated=self.punctuated)
        if inputs is None:
            return 1.0
        if self.barge is not None:
            if self._barge_task is not None and not self._barge_task.done():
                await asyncio.wait({self._barge_task}, timeout=self.timeout + 1.5)    # an interim decision, in flight
            if self._speech()[0] == "audible":
                if await self._barge_in(inputs["caller_text"]) == "CONTINUE" and self._speech()[0] == "audible":
                    # LiveKit drops words that end over her only if she is still speaking when it commits the turn:
                    # say "done" so it commits now, and forget their audio
                    if self.smart_turn is not None:
                        self.smart_turn.reset()
                    return 1.0
            # her next speech has made no sound yet (the LLM writing, the TTS synthesising, or queued behind one that
            # just ended) and the caller went on: the turn was not over. Drop it, or LiveKit would skip these words
            await self._drop_unheard()
        if not self._decides(self._language):
            return 1.0
        t0 = time.perf_counter()
        # Smart Turn (CPU, ~100 ms) runs beside the Cue question, not after it
        heard = asyncio.ensure_future(asyncio.to_thread(self.smart_turn.predict)) if self.smart_turn is not None else None
        try:
            probs = await asyncio.wait_for(self.cue.aask(self.eot, inputs), timeout or self.timeout)
        except Exception as e:                    # late, down, refused: the turn goes on as done
            log.warning("cue: no end-of-turn answer (%s): treating the turn as done", type(e).__name__)
            if heard is not None:
                heard.cancel()
            return 1.0
        ms = (time.perf_counter() - t0) * 1000
        voice = None
        if heard is not None:
            try:
                voice = await heard
            except Exception as e:                # Smart Turn failed: the words alone decide
                log.warning("cue: smart turn failed (%s): words only", e)
        p = with_voice(probs, voice) if voice is not None else p_end(probs)
        if self.barge is not None:
            await self._drop_unheard()            # a queued speech may have become hers while Cue was answering
        if self.graded_waits and p < self.threshold and self.session is not None:
            try:                                  # LiveKit 1.8.1 reads max_delay after the detector has answered
                self.session.update_options(endpointing_opts={"max_delay": wait(probs)})
            except Exception as e:
                log.warning("cue: graded wait not set: %r", e)
        best = max(probs, key=probs.get)
        self.last_end_of_turn = {"option": best, "probs": probs, "smart_turn": voice, "p": p, "ms": ms,
                                 "caller_text": inputs["caller_text"]}
        log.info("cue: end_of_turn %s p(done)=%.2f%s -> %s (%.0f ms) %r", best, p_end(probs),
                 f" smart_turn={voice:.2f} p={p:.2f}" if voice is not None else "",
                 "reply" if p >= self.threshold else "wait", ms, inputs["caller_text"][-60:])
        return p

    # -- following the session ---------------------------------------------------------------------------
    def attach(self, session: Any) -> CueTurnDetector:
        """Follow the session: which of her speeches is playing (barge-in reads what she has said of it; Smart Turn
        starts a new window of the caller's audio), and the caller's interim transcripts (barge-in). Call it once,
        after creating the AgentSession."""
        self.session = session
        if self.smart_turn is not None:
            self.smart_turn.reset()

        @session.on("agent_state_changed")
        def _state(ev) -> None:
            if ev.new_state == "speaking":
                self._speech()                    # notes a new speech as audible
            elif ev.old_state == "speaking":
                self._heard = ""

        @session.on("user_input_transcribed")
        def _transcribed(ev) -> None:
            # STTs with interim transcripts (Deepgram, ElevenLabs, ...): decide barge-in as the words come in, so a
            # real interruption stops her within its first words, not after the whole sentence is transcribed
            if self.barge is None or getattr(ev, "is_final", True) or self._speech()[0] != "audible":
                return
            text = (getattr(ev, "transcript", "") or "").strip()
            if not text or text == self._heard:
                return
            self._heard = text
            if self._barge_task is None or self._barge_task.done():
                self._barge_task = asyncio.get_running_loop().create_task(self._barge_on_interims())
        return self

    def _speech(self) -> tuple[str, Any]:
        """Her current speech: ("audible", handle) once it has made a sound, ("unheard", handle) before (the LLM
        writing, the TTS synthesising), ("none", None) if there is none (listening, or a tool running after she
        has spoken: LiveKit answers the caller itself then)."""
        cs = getattr(self.session, "current_speech", None)
        if cs is None or cs.done() or getattr(cs, "interrupted", False):
            return "none", None
        if self.barge is not None and getattr(cs, "allow_interruptions", False) and not self._warned_interruptible:
            self._warned_interruptible = True
            log.warning("cue: a speech allows interruptions: LiveKit stops her on sound and answers backchannels "
                        "itself (barge-in needs every speech uninterruptible)")
        step = getattr(cs, "num_steps", 1)
        if cs is self._audible and step == self._audible_step:
            return "audible", cs
        if getattr(self.session, "agent_state", None) == "speaking":
            # playing: a new speech, or a tool's reply (the same handle's next step). LiveKit says "speaking" from
            # the first audio frame; "thinking" also shows while a reply is queued behind one still playing, which
            # stays the audible one
            self._audible, self._audible_step, self._audible_since, self._heard = cs, step, time.monotonic(), ""
            if self.smart_turn is not None and getattr(self.session, "user_state", None) != "speaking":
                self.smart_turn.reset()           # what the caller says next is a new turn
            return "audible", cs
        return "unheard", cs

    def spoken(self) -> str:
        """What she has said of the speech that is playing: its playout time x her speaking rate, cut to a word.
        LiveKit sends the TTS the whole reply as fast as the LLM writes it, so the text alone would include a
        question she has not reached yet, and a "yes" over her would read as its answer."""
        key = (getattr(self._audible, "id", None), self._audible_step)
        text = self._texts.get(key, "") if self._audible is not None else ""
        n = int((time.monotonic() - self._audible_since) * self.chars_per_s)
        if n >= len(text):
            return text
        cut = text[:n]
        return cut[: cut.rfind(" ")].rstrip() if " " in cut else ""

    # -- barge-in --------------------------------------------------------------------------------------
    async def _barge_on_interims(self) -> None:
        """One question at a time, always about the latest words; stops at the first STOP."""
        asked = ""
        while self._heard and self._heard != asked and self._speech()[0] == "audible":
            asked = self._heard
            if await self._barge_in(asked) == "STOP":
                return

    async def _barge_in(self, caller: str) -> str:
        said = self.spoken()
        inputs = {"agent_said": said[-300:], "agent_question": last_question(said), "caller_text": caller}
        t0 = time.perf_counter()
        try:
            probs = await asyncio.wait_for(self.cue.aask(self.barge, inputs), self.timeout)
            option, how = max(probs, key=probs.get), f"{max(probs.values()):.2f}"
        except Exception as e:                    # no answer: three or more words stop her, a "hmm" does not
            words = sum(1 for w in caller.split() if re.search(r"\w", w))
            option = "STOP" if words >= self.stop_without_answer else "CONTINUE"
            how = f"no answer ({type(e).__name__}), by word count"
        log.info("cue: barge_in %s %s (%.0f ms) %r", option, how, (time.perf_counter() - t0) * 1000, caller[-60:])
        self.last_barge_in = {"option": option, "how": how, "caller_text": caller, "agent_said": inputs["agent_said"]}
        if option == "STOP" and self._speech()[0] != "none":
            await self._drop()                    # she is speaking, or her next speech is about to
        return option

    async def _drop_unheard(self) -> None:
        if self._speech()[0] == "unheard":
            log.info("cue: the caller went on before her speech was heard: replacing it")
            await self._drop()

    async def _drop(self) -> None:
        """Stop her current speech and drop her queued ones (forced: LiveKit's own interruptions are off; a queued
        speech left behind would become current, and LiveKit skips a turn that ends during a speech it may not
        interrupt). A tool's speech still working in the background is left alone, so its result is still spoken.
        Then wait until LiveKit has let go of the current speech, so it takes the caller's words as an ordinary turn."""
        cs = getattr(self.session, "current_speech", None)
        activity = getattr(self.session, "_activity", None)
        queued = [item[-1] for item in sorted(getattr(activity, "_speech_q", None) or [], key=lambda i: i[:2])]
        background = getattr(activity, "_background_speeches", None) or set()
        for speech in ([cs] if cs is not None else []) + queued:
            if speech in background or speech.done():
                continue
            try:
                speech.interrupt(force=True)
            except Exception as e:
                log.warning("cue: interrupt failed: %r", e)
        if cs is not None:
            try:
                await asyncio.wait_for(asyncio.shield(asyncio.ensure_future(cs.wait_for_playout())), 1.5)
            except Exception as e:
                log.warning("cue: her speech did not stop in time: %r", e)

    # -- taps for your Agent's nodes -----------------------------------------------------------------
    async def listen_text(self, text: AsyncIterable[str]) -> AsyncIterator[str]:
        """In your tts_node: what she is about to say, per speech (barge-in reads the part that has played)."""
        handle = _speech_handle()
        key = (getattr(handle, "id", None), getattr(handle, "num_steps", 1) if handle is not None else 0)
        if key not in self._texts:                # a new speech or a tool's reply
            self._texts[key] = ""
            while len(self._texts) > 8:          # the last few are enough
                del self._texts[next(iter(self._texts))]
        elif self._texts[key] and not self._texts[key].endswith(" "):
            self._texts[key] += " "               # the same speech's next segment (an llm_node FlushSentinel)
        async for chunk in text:
            self._texts[key] = self._texts.get(key, "") + chunk
            yield chunk

    async def listen_audio(self, audio: AsyncIterable[Any]) -> AsyncIterator[Any]:
        """In your stt_node: the caller's audio, for Smart Turn (resampled to 16 kHz mono)."""
        from livekit import rtc
        resampler: rtc.AudioResampler | None = None
        rate = None
        async for frame in audio:
            if self.smart_turn is not None and isinstance(frame, rtc.AudioFrame):
                if frame.sample_rate != 16000 or frame.num_channels != 1:
                    if resampler is None or rate != (frame.sample_rate, frame.num_channels):
                        rate = (frame.sample_rate, frame.num_channels)
                        resampler = rtc.AudioResampler(frame.sample_rate, 16000, num_channels=frame.num_channels)
                    out = resampler.push(frame)
                else:
                    out = [frame]
                for f in out:
                    import numpy as np
                    pcm = np.frombuffer(f.data, dtype=np.int16)
                    if f.num_channels > 1:
                        pcm = pcm.reshape(-1, f.num_channels).mean(axis=1).astype(np.int16)
                    self.smart_turn.push(pcm)
            yield frame

    async def aclose(self) -> None:
        if self._barge_task is not None:
            self._barge_task.cancel()
        await self.cue.aclose()
