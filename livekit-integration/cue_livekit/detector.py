"""Cue v1 in a LiveKit Agents voice agent (tested on livekit-agents 1.7.1, 1.8.1 and 1.8.7).

LiveKit takes a turn detector in ``turn_handling["turn_detection"]``: an object it asks, once the caller has paused
and there are words, for p(the turn is over). Below its threshold, LiveKit waits ``endpointing.max_delay`` before
replying; otherwise ``min_delay``. CueTurnDetector answers that from Cue v1's end-of-turn decision:

    detector = CueTurnDetector("http://localhost:8090/v1")
    session = AgentSession(..., turn_handling={"turn_detection": detector,
                                              "endpointing": {"min_delay": 0.3, "max_delay": 2.0}})

Two optional parts, each a few lines in your Agent (see README.md in this folder):

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

import httpx

from cue.client import WAITS, Cue, p_end, wait, with_voice
from cue.decision import UNPUNCTUATED, Decision, ending, last_question, load, prompt

log = logging.getLogger("cue.livekit")

# graded waits when Cue's words say the caller is not done: blended over the not-done answers only (1.75 to 2.25 s)
NOT_DONE_WAITS = {o: WAITS[o] for o in ("MID_THOUGHT", "HOLDING")}

# the end-of-turn inputs this adapter fills: the phone prompt's four and the dialogue prompt's three
PHONE_INPUTS = ("agent_question", "caller_earlier", "caller_text", "transcript_end")
DIALOGUE_INPUTS = ("Other person last said", "Speaker said", "Speaker's words end with")


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


def _str(language: Any) -> str | None:
    return str(language) if language else None


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
    ``fallback``: a text turn detector with ``predict_end_of_turn`` (``MultilingualModel()`` or ``EnglishModel()`` from
    livekit-plugins-turn-detector; LiveKit's streaming ``inference.TurnDetector`` cannot be one) that decides when Cue
    gives no answer in time and for languages outside ``languages``; without one those turns are treated as done.
    ``graded_waits``: when Cue's words say the caller is not done, wait between 1.75 s (MID_THOUGHT) and 2.25 s
    (HOLDING), blended by their probabilities, instead of LiveKit's one ``max_delay``, which is put back on every other
    turn; needs ``attach(session)`` and LiveKit's fixed endpointing mode (the default), and relies on LiveKit reading
    ``max_delay`` after the detector answers (1.7.1 to 1.8.7 do). ``keepalive_s``: how long the client keeps the
    connection to the Cue server between decisions (the server or a proxy in front of it must keep it too)."""

    def __init__(self, base_url: str = "http://localhost:8090/v1", *, model: str = "cue", api_key: str = "x",
                 end_of_turn: str | Path | None = None, barge_in: bool | str | Path = False,
                 threshold: float = 0.5, smart_turn: Any = None, timeout: float = 1.0, languages: Any = None,
                 punctuated: bool = True, graded_waits: bool = False, fallback: Any = None,
                 chars_per_s: float = 14.0, stop_without_answer: int = 3, keepalive_s: float = 60.0) -> None:
        self.cue = Cue(base_url, model, api_key, timeout=max(timeout, 1.0) + 1.0)
        # keep the connection between decisions: httpx drops an idle one after 5 s, and a caller's turns are further
        # apart than that (measured over TLS: 141 ms a decision on a new connection, 67 ms on a kept one)
        self.cue._async = httpx.AsyncClient(timeout=self.cue.timeout, limits=httpx.Limits(keepalive_expiry=keepalive_s))
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
        if fallback is not None and not callable(getattr(fallback, "predict_end_of_turn", None)):
            raise TypeError("fallback must be a text turn detector with predict_end_of_turn, such as MultilingualModel() "
                            "or EnglishModel(); LiveKit's streaming inference.TurnDetector cannot be one")
        self.fallback = fallback
        self._fell_back = False                  # this turn was decided by the fallback (its threshold applies)
        # graded waits: the endpointing they changed, its own max_delay (put back on every other turn), and whether
        # it currently holds a graded value
        self._graded_ep: Any = None
        self._base_max_delay: float | None = None
        self._graded = False
        self._warm_task: asyncio.Task | None = None
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
        # every end-of-turn call, for your logs: status "cue" (Cue answered), "fallback" (the fallback decided),
        # "no_answer" (no answer, treated as done) or "skipped" (no words, a backchannel over her, another language)
        self.last_end_of_turn: dict[str, Any] = {}
        self.last_barge_in: dict[str, Any] = {}       # the last barge-in decision

    # -- LiveKit's turn detector interface -----------------------------------------------------------
    @property
    def model(self) -> str:
        return "cue-v1"

    @property
    def provider(self) -> str:
        return "pype"

    async def unlikely_threshold(self, language: Any = None) -> float | None:
        if self._fell_back:                      # LiveKit asks right after predict_end_of_turn: the fallback's scale
            try:
                return await self.fallback.unlikely_threshold(language)
            except Exception:
                return self.threshold
        return self.threshold

    async def supports_language(self, language: Any = None) -> bool:
        """Cue v1 was trained on English; on eot-bench it ran on 14 languages with the English prompt. LiveKit only
        calls predict_end_of_turn for a supported language, so this is also True when barge-in is on (barge-in lives
        there) or when the fallback detector supports the language (it decides those turns)."""
        self._language = language
        if self.barge is not None or self._decides(language):
            return True
        if self.fallback is not None:
            try:
                if await self.fallback.supports_language(language):
                    return True
            except Exception:
                pass
        # LiveKit will not ask about this turn: record it, so last_end_of_turn is not left on the previous one
        self._set_max_delay(None)
        self.last_end_of_turn = {"status": "skipped", "language": _str(language), "p": None, "caller_text": "",
                                 "reason": "language"}
        return False

    def _decides(self, language: Any) -> bool:
        return self.languages is None or language is None or _language(language) in self.languages

    async def predict_end_of_turn(self, chat_ctx: Any, *, timeout: float | None = None) -> float:
        self._fell_back = False
        language = self._language
        inputs = end_of_turn_inputs(chat_ctx, punctuated=self.punctuated)
        if inputs is None:
            return self._done("skipped", language, reason="no_words")
        if self.barge is not None:
            if self._barge_task is not None and not self._barge_task.done():
                await asyncio.wait({self._barge_task}, timeout=self.timeout + 1.5)    # an interim decision, in flight
            if self._speech()[0] == "audible":
                if await self._barge_in(inputs["caller_text"]) == "CONTINUE" and self._speech()[0] == "audible":
                    # LiveKit drops words that end over her only if she is still speaking when it commits the turn:
                    # say "done" so it commits now, and forget their audio
                    if self.smart_turn is not None:
                        self.smart_turn.reset()
                    return self._done("skipped", language, inputs, reason="backchannel")
            # her next speech has made no sound yet (the LLM writing, the TTS synthesising, or queued behind one that
            # just ended) and the caller went on: the turn was not over. Drop it, or LiveKit would skip these words
            await self._drop_unheard()
        if not self._decides(language):
            return await self._fall_back(chat_ctx, timeout, language, inputs, reason="language")
        t0 = time.perf_counter()
        # Smart Turn (CPU, ~100 ms) runs beside the Cue question, not after it
        heard = asyncio.ensure_future(asyncio.to_thread(self.smart_turn.predict)) if self.smart_turn is not None else None
        try:
            probs = await asyncio.wait_for(self.cue.aask(self.eot, inputs), timeout or self.timeout)
        except Exception as e:                    # late, down, refused: the fallback decides, or the turn is done
            if heard is not None:
                heard.cancel()
            return await self._fall_back(chat_ctx, timeout, language, inputs, reason="no_answer",
                                         error=type(e).__name__)
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
        # graded: only when Cue's words say not done (a Smart Turn veto keeps the session's own max_delay)
        self._set_max_delay(wait(probs, NOT_DONE_WAITS) if p < self.threshold and p_end(probs) < self.threshold
                            else None)
        best = max(probs, key=probs.get)
        self.last_end_of_turn = {"status": "cue", "language": _str(language), "option": best, "probs": probs,
                                 "smart_turn": voice, "p": p, "ms": ms, "caller_text": inputs["caller_text"]}
        log.info("cue: end_of_turn %s p(done)=%.2f%s -> %s (%.0f ms) %r", best, p_end(probs),
                 f" smart_turn={voice:.2f} p={p:.2f}" if voice is not None else "",
                 "reply" if p >= self.threshold else "wait", ms, inputs["caller_text"][-60:])
        return p

    async def _fall_back(self, chat_ctx: Any, timeout: float | None, language: Any, inputs: dict, *, reason: str,
                         error: str | None = None) -> float:
        """A turn Cue does not decide: the fallback detector does if there is one and it supports the language, else
        the turn is treated as done (min_delay), as LiveKit does without a detector."""
        why = f"no end-of-turn answer ({error})" if error is not None else f"language {language} is not Cue's"
        fallback: dict[str, Any] = {}
        if self.fallback is not None:
            t0 = time.perf_counter()
            try:
                if await self.fallback.supports_language(language):
                    # its own budget: LiveKit passes no timeout, and the fallback's default (3 s) is longer than ours
                    p = float(await self.fallback.predict_end_of_turn(
                        chat_ctx, timeout=timeout if timeout is not None else self.timeout))
                    self._fell_back = True
                    log.info("cue: %s: the fallback decided p=%.2f", why, p)
                    return self._done("fallback", language, inputs, reason=reason, error=error, p=p,
                                      ms=(time.perf_counter() - t0) * 1000)
                fallback = {"fallback": "declined"}   # e.g. MultilingualModel with no language from the STT
            except Exception as e:
                fallback = {"fallback": "failed", "fallback_error": type(e).__name__}
        if error is not None:
            log.warning("cue: %s: %s", why, "treating the turn as done" if not fallback
                        else f"the fallback {fallback['fallback']}: treating the turn as done")
        return self._done("no_answer" if reason == "no_answer" else "skipped", language, inputs, reason=reason,
                          error=error, **fallback)

    def _done(self, status: str, language: Any, inputs: dict | None = None, *, p: float = 1.0, **extra: Any) -> float:
        """Record a turn Cue did not answer, and return what LiveKit gets (1.0: done, unless the fallback said)."""
        self._set_max_delay(None)
        self.last_end_of_turn = {"status": status, "language": _str(language), "p": p,
                                 "caller_text": (inputs or {}).get("caller_text", ""),
                                 **{k: v for k, v in extra.items() if v is not None}}
        return p

    def _endpointing(self) -> Any:
        """LiveKit's live endpointing object (the session's, merged with the current Agent's)."""
        recognition = getattr(getattr(self.session, "_activity", None), "_audio_recognition", None)
        return getattr(recognition, "_endpointing", None)

    def _set_max_delay(self, seconds: float | None) -> None:
        """Graded waits: Cue's wait for this turn, or (None) the endpointing's own max_delay back. Changes only
        max_delay on LiveKit's live endpointing object, so min_delay and an Agent's own endpointing are kept."""
        if not self.graded_waits or self.session is None:
            return
        ep = self._endpointing()
        if ep is None or type(ep).__name__ != "BaseEndpointing":    # fixed mode only: dynamic learns its own delays
            return
        if ep is not self._graded_ep:            # new: the call starts, an agent handoff, update_options
            self._graded_ep, self._base_max_delay, self._graded = ep, ep.max_delay, False
        try:
            if seconds is not None:
                ep.update_options(max_delay=seconds)
                self._graded = True
            elif self._graded:
                ep.update_options(max_delay=self._base_max_delay)
                self._graded = False
        except Exception as e:
            log.warning("cue: graded wait not set: %r", e)

    async def warm(self) -> None:
        """Open the connection to the Cue server now, so the call's first decision does not pay for it."""
        try:
            base = self.cue.url.rsplit("/chat/completions", 1)[0]
            await self.cue._async.get(f"{base}/models", headers=self.cue.headers, timeout=2.0)
        except Exception as e:
            log.info("cue: warm-up failed (%r): the first decision opens the connection", e)

    # -- following the session ---------------------------------------------------------------------------
    def attach(self, session: Any) -> CueTurnDetector:
        """Follow the session: which of her speeches is playing (barge-in reads what she has said of it; Smart Turn
        starts a new window of the caller's audio), and the caller's interim transcripts (barge-in). Call it once,
        after creating the AgentSession. It also opens the connection to the Cue server."""
        self.session = session
        if self.smart_turn is not None:
            self.smart_turn.reset()
        try:
            self._warm_task = asyncio.get_running_loop().create_task(self.warm())
        except RuntimeError:                      # no event loop yet: the first decision opens the connection
            pass

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
        for task in (self._barge_task, self._warm_task):
            if task is not None:
                task.cancel()
        await self.cue.aclose()
