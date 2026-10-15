"""CueTurnDetector against a fake Cue server (httpx's mock transport) and LiveKit's own classes where they are
installed: the inputs it sends, the probability it returns, its fallbacks, and barge-in against a fake session that
behaves as LiveKit 1.8.1's does (current_speech, agent_state, interrupt)."""
import asyncio
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from cue import load
from cue_livekit import CueTurnDetector, end_of_turn_inputs, prompt

np = pytest.importorskip("numpy")
PROMPTS = ("end_of_turn_phone", "end_of_turn_dialogue", "barge_in")


def server(rules: dict[str, list[tuple[str, str]]], *, delay: float = 0.0, fail: bool = False):
    """A Cue server: for each decision (told apart by its instruction), the first rule whose text is in the
    caller's words answers that option with p 0.97 (else the first option). Records every request."""
    decisions = {n: load(prompt(n)) for n in PROMPTS}
    asked = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        name = next(n for n, d in decisions.items() if system.startswith(d.instruction[:60]))
        caller = re.search(r"(?:Caller|Speaker) said: (.*)", user).group(1)
        asked.append((name, user, body))
        if delay:
            await asyncio.sleep(delay)
        if fail:
            return httpx.Response(503)
        d = decisions[name]
        option = next((o for sub, o in rules.get(name, []) if sub.lower() in caller.lower()), d.ids[0])
        top = [{"token": d.letter(option), "logprob": math.log(0.97)}] + \
              [{"token": d.letter(o), "logprob": math.log(0.03 / (len(d.ids) - 1))} for o in d.ids if o != option]
        return httpx.Response(200, json={"choices": [{"message": {"content": d.letter(option)},
                                                      "logprobs": {"content": [{"top_logprobs": top}]}}]})
    return handler, asked


def detector(rules, **kw) -> tuple[CueTurnDetector, list]:
    handler, asked = server(rules, delay=kw.pop("delay", 0.0), fail=kw.pop("fail", False))
    det = CueTurnDetector("http://cue.test/v1", **kw)
    det.cue._async = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return det, asked


def chat(*turns: tuple[str, str]):
    return [{"role": r, "content": t} for r, t in turns]


def names(asked):
    return [n for n, _, _ in asked]


def run(coro):
    return asyncio.run(coro)


# -- end of turn ---------------------------------------------------------------------------------------------
def test_the_inputs_are_the_ones_cue_was_benchmarked_with():
    msgs = chat(("assistant", "Thanks. What's the best number to reach you on?"), ("user", "Sure, it's nine eight four five,"))
    got = end_of_turn_inputs(msgs)
    assert {k: got[k] for k in ("agent_question", "caller_earlier", "caller_text", "transcript_end")} == {
        "agent_question": "What's the best number to reach you on?", "caller_earlier": "",
        "caller_text": "Sure, it's nine eight four five,", "transcript_end": "no sentence end"}
    msgs = chat(("user", "I need an appointment"), ("assistant", "Sure, for whom?"), ("user", "for my father"))
    assert end_of_turn_inputs(msgs)["caller_earlier"] == "I need an appointment"
    assert end_of_turn_inputs(chat(("assistant", "Hello"))) is None
    # the caller twice in a row (LiveKit's history while her reply to the first is being written): as on eot-bench,
    # agent_question comes only from her messages after the caller's previous one
    two = end_of_turn_inputs(chat(("assistant", "Which doctor?"), ("user", "Dr. Rao"), ("user", "on Monday")))
    assert two["agent_question"] == "" and two["caller_earlier"] == "Dr. Rao"
    assert two["Other person last said"] == "Which doctor?"


def test_an_stt_that_seldom_punctuates_says_so_as_eot_bench_did():
    msgs = chat(("assistant", "Which city?"), ("user", "मैं पुणे में रहता हूँ"))
    assert end_of_turn_inputs(msgs)["transcript_end"] == "no sentence end"
    assert end_of_turn_inputs(msgs, punctuated=False)["transcript_end"].startswith("no punctuation")


def test_livekit_chat_context_is_read_the_same_way():
    llm = pytest.importorskip("livekit.agents.llm")
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content="You are an assistant.")
    ctx.add_message(role="assistant", content="Which city are you in?")
    ctx.add_message(role="user", content="I stay in Pune but")
    got = end_of_turn_inputs(ctx)
    assert (got["agent_question"], got["caller_earlier"], got["caller_text"], got["transcript_end"]) == (
        "Which city are you in?", "", "I stay in Pune but", "no sentence end")


def test_a_half_said_number_is_unlikely_to_be_the_end_and_a_full_one_likely():
    det, asked = detector({"end_of_turn_phone": [("nine eight four five,", "MID_THOUGHT"), ("1234", "DONE")]})

    async def main():
        half = await det.predict_end_of_turn(chat(("assistant", "Your number?"), ("user", "It's nine eight four five,")))
        full = await det.predict_end_of_turn(chat(("assistant", "Your number?"), ("user", "It's 98450 1234.")))
        assert half < await det.unlikely_threshold("en") <= full
        assert half == pytest.approx(0.03 / 3 * 2, abs=1e-6) and full == pytest.approx(0.97 + 0.01, abs=1e-6)
        body = asked[0][2]
        assert body["max_tokens"] == 1 and body["logprobs"] is True and body["reasoning_effort"] == "none"
        assert det.last_end_of_turn["option"] == "DONE" and set(det.last_end_of_turn["probs"]) == set(det.eot.ids)
    run(main())


def test_the_dialogue_prompt_gets_the_callers_words_and_an_unknown_prompt_fails_loudly(tmp_path):
    det, asked = detector({"end_of_turn_dialogue": [("Pune but", "MID_THOUGHT")]}, end_of_turn=prompt("end_of_turn_dialogue"))
    p = run(det.predict_end_of_turn(chat(("assistant", "Which city are you in?"), ("user", "I stay in Pune but"))))
    user = asked[0][1]
    assert "Speaker said: I stay in Pune but" in user and "Other person last said: Which city are you in?" in user
    assert "(nothing)" not in user and p < 0.5
    custom = json.loads(prompt("end_of_turn_phone").read_text())
    custom["inputs"] = ["agent_question", "caller_mood"]
    (tmp_path / "custom.json").write_text(json.dumps(custom))
    with pytest.raises(ValueError, match="caller_mood"):
        CueTurnDetector(end_of_turn=tmp_path / "custom.json")


def test_no_answer_is_treated_as_done():
    det, _ = detector({}, fail=True)
    assert run(det.predict_end_of_turn(chat(("user", "hello")))) == 1.0
    det, _ = detector({}, delay=0.3, timeout=0.05)
    assert run(det.predict_end_of_turn(chat(("user", "hello")))) == 1.0


def test_smart_turn_multiplies_by_the_square_root_of_its_score():
    st = SimpleNamespace(predict=lambda: 0.25, reset=lambda: None)
    det, _ = detector({"end_of_turn_phone": [("done", "DONE")]}, smart_turn=st)
    p = run(det.predict_end_of_turn(chat(("user", "that's all, I'm done."))))
    assert p == pytest.approx((0.97 + 0.01) * 0.5, abs=1e-6)


def test_smart_turn_runs_beside_cue_and_a_failure_falls_back_to_the_words():
    import time as _t

    def slow():
        _t.sleep(0.2)
        return 1.0
    det, _ = detector({"end_of_turn_phone": [("done", "DONE")]}, smart_turn=SimpleNamespace(predict=slow), delay=0.2)
    t0 = _t.perf_counter()
    run(det.predict_end_of_turn(chat(("user", "that's all, I'm done."))))
    assert _t.perf_counter() - t0 < 0.35, "0.2 s each, side by side"

    def broken():
        raise RuntimeError("no model")
    det, _ = detector({"end_of_turn_phone": [("done", "DONE")]}, smart_turn=SimpleNamespace(predict=broken))
    assert run(det.predict_end_of_turn(chat(("user", "that's all, I'm done.")))) == pytest.approx(0.98, abs=1e-6)


def test_languages_any_form_and_barge_in_still_runs_for_the_others():
    det, asked = detector({"end_of_turn_phone": [("", "MID_THOUGHT")]}, languages=["english", "hi"])

    async def main():
        assert await det.supports_language("en-IN") and await det.supports_language("hi-IN")
        assert not await det.supports_language("ta-IN")
        barge, asked2 = detector({"barge_in": [("wait", "STOP")], "end_of_turn_phone": [("", "MID_THOUGHT")]},
                                 languages=["en"], barge_in=True)
        session = FakeSession()
        barge.attach(session)
        assert await barge.supports_language("ta-IN"), "LiveKit must still call it: barge-in lives there"
        her = await speak(barge, session, "Your appointment is on Tuesday.", 0.05)
        p = await barge.predict_end_of_turn(chat(("user", "wait wait")))
        assert names(asked2) == ["barge_in"] and her.forced == [True], "barge-in ran, end of turn did not"
        assert p == 1.0, "a language Cue does not decide for: treated as done"
    run(main())


# -- barge-in --------------------------------------------------------------------------------------------------
class FakeSpeech:
    """A LiveKit SpeechHandle as the detector uses it (a tool's reply is the same handle, num_steps + 1)."""
    n = 0

    def __init__(self):
        FakeSpeech.n += 1
        self.id, self.num_steps, self.allow_interruptions = f"speech_{FakeSpeech.n}", 1, False
        self.interrupted, self._done, self.forced = False, False, []

    def done(self):
        return self._done

    def interrupt(self, *, force=False):
        self.forced.append(force)
        self.interrupted = self._done = True
        return self

    async def wait_for_playout(self):
        return None


class BaseEndpointing:
    """LiveKit's fixed-mode endpointing, as the detector uses it (same name, max_delay, update_options)."""

    def __init__(self, min_delay=0.3, max_delay=2.0):
        self.min_delay, self.max_delay, self.updates = min_delay, max_delay, []

    def update_options(self, *, min_delay=None, max_delay=None):
        self.updates.append(max_delay)
        self.max_delay = max_delay if max_delay is not None else self.max_delay


class FakeSession:
    """What the detector needs of an AgentSession: on(), agent_state, user_state, current_speech, and the activity's
    speech queue, background (tool) speeches and live endpointing."""

    def __init__(self, endpointing=None):
        self.handlers, self.agent_state, self.user_state = {}, "listening", "listening"
        self.current_speech = None
        self._activity = SimpleNamespace(_speech_q=[], _background_speeches=set(),
                                         _audio_recognition=SimpleNamespace(_endpointing=endpointing or BaseEndpointing()))

    @property
    def endpointing(self):
        return self._activity._audio_recognition._endpointing

    def on(self, name):
        def add(fn):
            self.handlers.setdefault(name, []).append(fn)
            return fn
        return add

    def set_agent(self, state):
        old, self.agent_state = self.agent_state, state
        for fn in self.handlers.get("agent_state_changed", []):
            fn(SimpleNamespace(old_state=old, new_state=state))

    def transcribe(self, text, final=False):
        for fn in self.handlers.get("user_input_transcribed", []):
            fn(SimpleNamespace(transcript=text, is_final=final))

    def queue(self, speech):
        self._activity._speech_q.append((0, float(len(self._activity._speech_q)), speech))



async def tts(det, speech, text):
    """Her text through the detector's tts tap, as LiveKit runs a tts_node: inside the speech's context."""
    var = pytest.importorskip("livekit.agents.voice.agent_activity")._SpeechHandleContextVar
    token = var.set(speech)
    try:
        async def gen():
            yield text
        async for _ in det.listen_text(gen()):
            pass
    finally:
        var.reset(token)


async def speak(det, session, text, seconds, *, tts_tap=True, speech=None):
    """A speech of hers: its text through the tts tap, then it plays for `seconds`."""
    speech = speech or FakeSpeech()
    if tts_tap:
        await tts(det, speech, text)
    session.current_speech = speech
    session.set_agent("speaking")
    await asyncio.sleep(seconds)
    return speech


def test_a_real_interruption_stops_her_before_livekit_ends_the_turn():
    det, asked = detector({"barge_in": [("wait", "STOP")], "end_of_turn_phone": [("tuesday", "DONE")]},
                          barge_in=True, chars_per_s=200)
    session = FakeSession()
    det.attach(session)

    async def main():
        her = await speak(det, session, "Your appointment is on Tuesday at ten. Shall I book it?", 0.4)
        p = await det.predict_end_of_turn(chat(("assistant", "Hello"), ("user", "wait, Tuesday doesn't work")))
        assert her.forced == [True], "STOP: interrupted, forced (LiveKit's own interruptions are off)"
        assert p >= 0.5
        assert names(asked) == ["barge_in", "end_of_turn_phone"], "barge-in first, while she is still talking"
    run(main())


def test_a_stop_drops_her_queued_speeches_but_not_a_tools_speech_in_the_background():
    det, _ = detector({"barge_in": [("wait", "STOP")]}, barge_in=True)
    session = FakeSession()
    det.attach(session)

    async def main():
        her = await speak(det, session, "Your appointment is on Tuesday.", 0.05)
        queued, tool = FakeSpeech(), FakeSpeech()
        session.queue(queued)
        session._activity._background_speeches.add(tool)
        await det.predict_end_of_turn(chat(("user", "wait, which day?")))
        assert her.forced == [True] and queued.forced == [True], "or the queued one plays and LiveKit skips the turn"
        assert tool.forced == [], "a tool still working: its result is still spoken"
    run(main())


def test_a_backchannel_over_her_does_not_stop_her_and_is_committed_at_once():
    st = SimpleNamespace(resets=0, predict=lambda: 0.1)
    st.reset = lambda: setattr(st, "resets", st.resets + 1)
    det, asked = detector({"barge_in": [("hmm", "CONTINUE")]}, barge_in=True, smart_turn=st)
    session = FakeSession()
    det.attach(session)

    async def main():
        her = await speak(det, session, "The plan covers hospital bills up to five lakhs and", 0.05)
        resets = st.resets
        p = await det.predict_end_of_turn(chat(("user", "hmm okay")))
        assert names(asked) == ["barge_in"], "no end-of-turn question for words that do not stop her"
        assert p == 1.0, "LiveKit commits now, while she is still speaking, and so drops the words"
        assert her.forced == [] and session.agent_state == "speaking"
        assert st.resets == resets + 1, "their audio is not part of the caller's next turn"
    run(main())


def test_barge_in_reads_what_she_has_said_not_what_she_will_say():
    det, asked = detector({"barge_in": [("yes", "CONTINUE")]}, barge_in=True, chars_per_s=50)
    session = FakeSession()
    det.attach(session)

    async def main():
        await speak(det, session, "Dr. Rao is free on Tuesday at 4 pm. Shall I book that?", 0.25)
        await det.predict_end_of_turn(chat(("user", "yes")))
        name, user, _ = asked[0]
        assert name == "barge_in"
        assert "Agent is saying: Dr. Rao is\n" in user and "Shall I book" not in user, user
        assert "Agent asked: (nothing)" in user
    run(main())


def test_a_speech_in_segments_is_read_in_order():
    det, _ = detector({}, barge_in=True, chars_per_s=500)
    session = FakeSession()
    det.attach(session)

    async def main():
        her = FakeSpeech()
        await tts(det, her, "Dr. Rao has a slot on Tuesday.")
        await tts(det, her, "Shall I book it?")           # the next segment (an llm_node FlushSentinel)
        await speak(det, session, "", 0.2, tts_tap=False, speech=her)
        assert det.spoken() == "Dr. Rao has a slot on Tuesday. Shall I book it?"
    run(main())


def test_a_tools_reply_is_a_new_speech_though_livekit_reuses_the_handle():
    det, asked = detector({"barge_in": [("yes", "CONTINUE")], "end_of_turn_phone": [("wednesday", "DONE")]},
                          barge_in=True, chars_per_s=50)
    session = FakeSession()
    det.attach(session)

    async def main():
        her = await speak(det, session, "One moment, let me check.", 1.0)
        session.current_speech = None                     # the tool runs: LiveKit holds her speech in the background
        session._activity._background_speeches.add(her)
        session.set_agent("thinking")
        await asyncio.sleep(0.3)
        session._activity._background_speeches.discard(her)
        her.num_steps = 2                                 # the tool's reply: the same handle, its next step
        await tts(det, her, "Dr. Rao is free on Tuesday at four. Shall I book it?")
        session.current_speech = her
        await det.predict_end_of_turn(chat(("assistant", "One moment."), ("user", "actually, make it Wednesday")))
        assert her.forced == [True], "not heard yet: replaced, so LiveKit answers the caller"

        det2, asked2 = detector({"barge_in": [("yes", "CONTINUE")]}, barge_in=True, chars_per_s=50)
        session2 = FakeSession()
        det2.attach(session2)
        her2 = await speak(det2, session2, "One moment, let me check.", 1.0)
        session2.current_speech = None
        session2.set_agent("thinking")
        await asyncio.sleep(0.3)
        her2.num_steps = 2
        await tts(det2, her2, "Dr. Rao is free on Tuesday at four. Shall I book it?")
        await speak(det2, session2, "", 0.25, tts_tap=False, speech=her2)    # it plays: spoken() starts again
        await det2.predict_end_of_turn(chat(("user", "yes")))
        user = asked2[0][1]
        assert "Agent is saying: Dr. Rao is\n" in user and "Shall I book" not in user, user
    run(main())


def test_a_speech_with_no_text_through_the_tts_tap_is_not_read_as_the_previous_one():
    det, asked = detector({"barge_in": [("yes", "CONTINUE")]}, barge_in=True, chars_per_s=500)
    session = FakeSession()
    det.attach(session)

    async def main():
        await speak(det, session, "Shall I book Dr. Rao for Tuesday?", 0.1)
        session.current_speech._done = True
        session.set_agent("listening")
        await speak(det, session, "", 0.1, tts_tap=False)   # session.say(text, audio=...): no tts_node
        await det.predict_end_of_turn(chat(("user", "yes")))
        assert "Shall I book" not in asked[0][1] and "Agent is saying: (nothing)" in asked[0][1]
    run(main())


def test_with_no_barge_in_answer_three_words_stop_her_and_a_short_backchannel_does_not():
    det, _ = detector({}, barge_in=True, fail=True)
    session = FakeSession()
    det.attach(session)

    async def main():
        her = await speak(det, session, "The plan covers", 0.05)
        for backchannel in ("hmm", "ठीक है", "हाँ हाँ"):        # Devanagari words are one word each, not three
            await det.predict_end_of_turn(chat(("user", backchannel)))
        assert her.forced == []
        await det.predict_end_of_turn(chat(("user", "no tell me the price")))
        assert her.forced == [True]
    run(main())


def test_interim_words_stop_her_before_the_sentence_is_finished():
    det, asked = detector({"barge_in": [("wait", "STOP"), ("hmm", "CONTINUE")]}, barge_in=True, delay=0.1)
    session = FakeSession()
    det.attach(session)

    async def main():
        her = await speak(det, session, "The plan covers hospital bills up to five lakhs and", 0.05)
        session.transcribe("hmm")
        await asyncio.sleep(0.02)                         # that question is out
        session.transcribe("hmm okay")
        session.transcribe("hmm okay wait")               # newer words while it is out: asked next, the latest only
        await asyncio.sleep(0.05)
        assert her.forced == [], "a hmm does not stop her"
        await asyncio.sleep(0.3)
        assert her.forced == [True]
        callers = [re.search(r"Caller said: (.*)", u).group(1) for n, u, _ in asked if n == "barge_in"]
        assert callers == ["hmm", "hmm okay wait"], callers
        # the final transcript then ends the turn: she has stopped, so no barge-in question on it
        await det.predict_end_of_turn(chat(("user", "hmm okay wait, what about Tuesday")))
        assert names(asked).count("barge_in") == 2 and her.forced == [True]
    run(main())


def test_interims_are_ignored_when_she_is_not_speaking_or_barge_in_is_off():
    det, asked = detector({"barge_in": [("wait", "STOP")]}, barge_in=True)
    session = FakeSession()
    det.attach(session)
    off, asked_off = detector({}, barge_in=False)
    session_off = FakeSession()
    off.attach(session_off)

    async def main():
        session.transcribe("wait")
        her = await speak(off, session_off, "The plan covers", 0.01)
        session_off.transcribe("wait")
        await asyncio.sleep(0.05)
        assert asked == [] and asked_off == [] and her.forced == []
    run(main())


def test_words_before_her_reply_makes_a_sound_replace_it_and_whatever_is_queued():
    det, asked = detector({"end_of_turn_phone": [("monday", "DONE")]}, barge_in=True)
    session = FakeSession()
    det.attach(session)

    async def main():
        reply, behind, tool = FakeSpeech(), FakeSpeech(), FakeSpeech()   # LiveKit is writing her reply to "Dr. Rao"
        session.current_speech = reply
        session.queue(behind)
        session._activity._background_speeches.add(tool)
        session.set_agent("thinking")
        await det.predict_end_of_turn(chat(("assistant", "Which doctor?"), ("user", "Dr. Rao"), ("user", "on Monday")))
        assert reply.forced == [True] and behind.forced == [True] and tool.forced == []
        assert names(asked) == ["end_of_turn_phone"], "no barge-in question: she has said nothing"
    run(main())


def test_words_while_a_say_is_synthesising_replace_it_too():
    det, asked = detector({}, barge_in=True)
    session = FakeSession()
    det.attach(session)

    async def main():
        greeting = FakeSpeech()                           # session.say(): no "thinking", still "listening" until audio
        session.current_speech = greeting
        await det.predict_end_of_turn(chat(("user", "hello, is this the clinic?")))
        assert greeting.forced == [True] and names(asked) == ["end_of_turn_phone"]
    run(main())


def test_when_her_speech_ends_during_the_question_the_next_one_is_replaced():
    det, asked = detector({"barge_in": [("wait", "STOP")]}, barge_in=True, delay=0.1)
    session = FakeSession()
    det.attach(session)

    async def main():
        first = await speak(det, session, "Dr. Rao is available on Tuesday.", 0.05)
        second = FakeSpeech()

        async def her_speech_ends():                      # while Cue is answering, the queued speech takes over
            await asyncio.sleep(0.05)
            first._done, session.current_speech = True, second
            session.set_agent("listening")
        asyncio.get_running_loop().create_task(her_speech_ends())
        await det.predict_end_of_turn(chat(("user", "wait, which doctor?")))
        assert second.forced == [True], "or LiveKit skips the turn while it plays"
    run(main())


def test_words_while_a_tool_runs_are_left_to_livekit():
    det, asked = detector({}, barge_in=True)
    session = FakeSession()
    det.attach(session)

    async def main():
        her = await speak(det, session, "One moment, I am booking that.", 0.01)
        session.current_speech = None                     # her speech ended, the tool runs in the background
        session._activity._background_speeches.add(her)
        session.set_agent("thinking")
        p = await det.predict_end_of_turn(chat(("user", "okay")))
        assert her.forced == [] and names(asked) == ["end_of_turn_phone"] and p == pytest.approx(0.98)
    run(main())


def test_a_reply_queued_behind_a_playing_greeting_does_not_hide_the_greeting():
    det, asked = detector({"barge_in": [("hmm", "CONTINUE")]}, barge_in=True)
    session = FakeSession()
    det.attach(session)

    async def main():
        greeting = await speak(det, session, "Thank you for calling the clinic. Our timings are nine to six.", 0.05)
        session.set_agent("thinking")                     # LiveKit: a reply was queued; the greeting still plays
        await det.predict_end_of_turn(chat(("user", "hmm")))
        assert names(asked) == ["barge_in"] and greeting.forced == []
    run(main())


def test_a_speech_that_allows_interruptions_is_warned_about(caplog):
    det, _ = detector({}, barge_in=True)
    session = FakeSession()
    det.attach(session)

    async def main():
        her = FakeSpeech()
        her.allow_interruptions = True
        await speak(det, session, "Hello", 0.01, speech=her)
    with caplog.at_level("WARNING", logger="cue.livekit"):
        run(main())
    assert "allows interruptions" in caplog.text


def test_graded_waits_set_livekits_max_delay_from_cues_answer():
    det, _ = detector({"end_of_turn_phone": [("pune but", "HOLDING"), ("done", "DONE")]}, graded_waits=True)
    session = FakeSession()
    det.attach(session)

    async def main():
        await det.predict_end_of_turn(chat(("user", "I stay in Pune but")))
        await det.predict_end_of_turn(chat(("user", "that's all, I'm done.")))
    run(main())
    holding = (0.97 * 2.25 + 0.01 * 1.75) / 0.98      # blended over the not-done answers only
    assert session.endpointing.updates == [pytest.approx(holding), 2.0], "then a finished turn: its own max_delay back"


class Fallback:
    """A LiveKit turn detector standing in for MultilingualModel: its own p and its own threshold."""

    def __init__(self, p=0.2, threshold=0.6, languages=("en", "ta")):
        self.p, self.threshold, self.languages, self.asked = p, threshold, languages, 0

    async def supports_language(self, language):
        return language is None or str(language).split("-")[0] in self.languages

    async def unlikely_threshold(self, language):
        return self.threshold

    async def predict_end_of_turn(self, chat_ctx, *, timeout=3):
        self.asked += 1
        self.timeouts = getattr(self, "timeouts", []) + [timeout]
        return self.p


def test_with_no_answer_the_fallback_decides_on_its_own_scale_and_cue_takes_over_again():
    fb = Fallback(p=0.2, threshold=0.6)
    det, _ = detector({}, fail=True, fallback=fb)

    async def main():
        assert await det.supports_language("en")
        p = await det.predict_end_of_turn(chat(("user", "my number is nine eight")))
        assert p == 0.2 and await det.unlikely_threshold("en") == 0.6, "the fallback's own threshold, for its p"
        assert fb.timeouts == [1.0], "LiveKit passes no timeout: the fallback gets ours, not its own 3 s"
        s = det.last_end_of_turn
        assert s["status"] == "fallback" and s["reason"] == "no_answer" and s["error"] and s["language"] == "en"
        good, _ = detector({"end_of_turn_phone": [("done", "DONE")]}, fallback=fb)
        await good.supports_language("en")
        assert await good.predict_end_of_turn(chat(("user", "that's all, I'm done."))) == pytest.approx(0.98)
        assert await good.unlikely_threshold("en") == 0.5 and good.last_end_of_turn["status"] == "cue"
    run(main())


def test_the_fallback_decides_languages_cue_does_not_and_livekit_is_told_so():
    fb = Fallback(p=0.3, languages=("ta",))
    det, asked = detector({}, languages=["en", "hi"], fallback=fb)
    plain, _ = detector({}, languages=["en", "hi"])

    async def main():
        assert await det.supports_language("ta-IN"), "LiveKit must call predict_end_of_turn for the fallback to decide"
        assert not await plain.supports_language("ta-IN") and not await det.supports_language("kn-IN")
        await det.supports_language("ta-IN")
        p = await det.predict_end_of_turn(chat(("user", "naan chennai la irukken")))
        assert p == 0.3 and asked == [] and det.last_end_of_turn["status"] == "fallback"
        assert det.last_end_of_turn["reason"] == "language"
    run(main())


def test_every_turn_leaves_a_status():
    det, _ = detector({"barge_in": [("hmm", "CONTINUE")], "end_of_turn_phone": [("done", "DONE")]}, barge_in=True)
    session = FakeSession()
    det.attach(session)

    async def main():
        await det.predict_end_of_turn(chat(("assistant", "Hello")))
        assert det.last_end_of_turn["status"] == "skipped" and det.last_end_of_turn["reason"] == "no_words"
        await det.predict_end_of_turn(chat(("user", "that's all, I'm done.")))
        assert det.last_end_of_turn["status"] == "cue" and det.last_end_of_turn["option"] == "DONE"
        await speak(det, session, "The plan covers", 0.01)
        await det.predict_end_of_turn(chat(("user", "hmm")))
        assert det.last_end_of_turn["status"] == "skipped" and det.last_end_of_turn["reason"] == "backchannel"
        down, _ = detector({}, fail=True)
        await down.predict_end_of_turn(chat(("user", "hello")))
        assert down.last_end_of_turn["status"] == "no_answer" and down.last_end_of_turn["p"] == 1.0
        assert down.last_end_of_turn["error"], "why there was no answer"
    run(main())


def test_graded_waits_put_the_sessions_own_max_delay_back_when_cue_does_not_decide():
    det, _ = detector({"end_of_turn_phone": [("pune but", "HOLDING")]}, graded_waits=True)
    session = FakeSession()
    det.attach(session)

    async def main():
        await det.predict_end_of_turn(chat(("user", "I stay in Pune but")))
        assert session.endpointing.max_delay == pytest.approx(2.24, abs=0.01)
        det.cue._async = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
        await det.predict_end_of_turn(chat(("user", "I stay in Pune but")))
        assert session.endpointing.max_delay == 2.0, "Cue did not decide: the session's own wait"
        n = len(session.endpointing.updates)
        await det.predict_end_of_turn(chat(("user", "still nothing")))
        assert len(session.endpointing.updates) == n, "nothing to put back twice"
    run(main())


def test_graded_waits_change_only_max_delay_and_keep_an_agents_own_endpointing():
    ep_mod = pytest.importorskip("livekit.agents.voice.endpointing")
    agent_ep = ep_mod.BaseEndpointing(min_delay=1.2, max_delay=3.0)      # set on the Agent, merged by LiveKit
    det, _ = detector({"end_of_turn_phone": [("pune but", "HOLDING")]}, graded_waits=True)
    session = FakeSession(endpointing=agent_ep)
    det.attach(session)

    async def main():
        await det.predict_end_of_turn(chat(("user", "I stay in Pune but")))
        assert agent_ep.min_delay == 1.2 and agent_ep.max_delay == pytest.approx(2.24, abs=0.01)
        await det.predict_end_of_turn(chat(("user", "that's all.")))         # DONE (the server's default answer)
        assert (agent_ep.min_delay, agent_ep.max_delay) == (1.2, 3.0), "the Agent's own, not the session's"
        handoff = ep_mod.BaseEndpointing(min_delay=0.5, max_delay=1.5)      # a new agent: a new endpointing
        session._activity._audio_recognition._endpointing = handoff
        await det.predict_end_of_turn(chat(("user", "I stay in Pune but")))
        await det.predict_end_of_turn(chat(("user", "that's all.")))
        assert handoff.max_delay == 1.5, "its own max_delay back, not the previous agent's"
    run(main())


def test_graded_waits_leave_a_smart_turn_veto_and_dynamic_endpointing_alone():
    st = SimpleNamespace(predict=lambda: 0.05, reset=lambda: None)
    det, _ = detector({"end_of_turn_phone": [("98450", "DONE")]}, graded_waits=True, smart_turn=st, threshold=0.3)
    session = FakeSession()
    det.attach(session)
    p = run(det.predict_end_of_turn(chat(("user", "My number is 98450."))))
    assert p < 0.3 and session.endpointing.updates == [], "Cue's words say done: the session's own max_delay"

    class DynamicEndpointing(BaseEndpointing):
        pass
    det2, _ = detector({"end_of_turn_phone": [("pune but", "HOLDING")]}, graded_waits=True)
    dynamic = FakeSession(endpointing=DynamicEndpointing())
    det2.attach(dynamic)
    run(det2.predict_end_of_turn(chat(("user", "I stay in Pune but"))))
    assert dynamic.endpointing.updates == [], "dynamic mode learns its own delays"


def test_a_streaming_turn_detector_is_refused_as_fallback_and_the_status_says_what_the_fallback_did():
    with pytest.raises(TypeError, match="predict_end_of_turn"):
        CueTurnDetector(fallback=SimpleNamespace(supports_language=None, unlikely_threshold=None))

    class Picky(Fallback):
        async def supports_language(self, language):
            return language is not None                          # MultilingualModel: no language, no decision

    class Broken(Fallback):
        async def predict_end_of_turn(self, chat_ctx, *, timeout=3):
            raise RuntimeError("model not loaded")

    async def main():
        det, _ = detector({}, fail=True, fallback=Picky())
        await det.supports_language(None)
        assert await det.predict_end_of_turn(chat(("user", "hello"))) == 1.0
        s = det.last_end_of_turn
        assert s["status"] == "no_answer" and s["fallback"] == "declined" and s["error"]
        det, _ = detector({}, fail=True, fallback=Broken())
        await det.supports_language("en")
        assert await det.predict_end_of_turn(chat(("user", "hello"))) == 1.0
        s = det.last_end_of_turn
        assert s["fallback"] == "failed" and s["fallback_error"] == "RuntimeError" and s["error"] != "RuntimeError"
    run(main())


def test_a_second_cue_server_can_be_the_fallback():
    backup, asked_backup = detector({"end_of_turn_phone": [("98450", "MID_THOUGHT")]})
    det, _ = detector({}, fail=True, fallback=backup)

    async def main():
        await det.supports_language("en")
        p = await det.predict_end_of_turn(chat(("assistant", "Your number?"), ("user", "My number is 98450")))
        assert p < 0.5 and det.last_end_of_turn["status"] == "fallback" and len(asked_backup) == 1
        assert await det.unlikely_threshold("en") == 0.5
    run(main())


def test_a_language_livekit_will_not_ask_about_still_leaves_a_status():
    det, _ = detector({"end_of_turn_phone": [("done", "DONE")]}, languages=["en"])

    async def main():
        await det.supports_language("en")
        await det.predict_end_of_turn(chat(("user", "that's all, I'm done.")))
        assert det.last_end_of_turn["status"] == "cue"
        assert not await det.supports_language("ta-IN")
        s = det.last_end_of_turn
        assert s["status"] == "skipped" and s["reason"] == "language" and s["language"] == "ta-IN"
    run(main())


def test_the_connection_is_kept_between_decisions_and_opened_on_attach():
    det = CueTurnDetector("http://cue.test/v1", api_key="k")
    assert det.cue._async._transport._pool._keepalive_expiry == 60.0, "httpx's default 5 s is shorter than a turn"
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json={"data": []})
    det.cue._async = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def main():
        det.attach(FakeSession())
        await asyncio.sleep(0.05)
        await det.aclose()
    run(main())
    assert seen == [("GET", "http://cue.test/v1/models", "Bearer k")]


# -- Smart Turn ------------------------------------------------------------------------------------------------
def test_the_caller_audio_reaches_smart_turn_at_16k_mono_and_passes_through_unchanged():
    rtc = pytest.importorskip("livekit.rtc")
    got = []
    det, _ = detector({}, smart_turn=SimpleNamespace(push=lambda pcm: got.append(pcm), reset=lambda: None))
    t = np.arange(4800) / 48000
    tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    frames = [rtc.AudioFrame(np.repeat(tone, 2).tobytes(), 48000, 2, 4800) for _ in range(5)]   # 0.5 s, 48k stereo

    async def main():
        async def gen():
            for f in frames:
                yield f
        return [f async for f in det.listen_audio(gen())]
    out = run(main())
    assert out == frames
    n = sum(len(g) for g in got)
    assert all(g.dtype == np.int16 for g in got) and 0.4 * 16000 <= n <= 0.5 * 16000, n
    assert np.abs(np.concatenate(got)).max() > 4000, "the tone survives the resample and the downmix"


def test_smart_turn_features_have_the_models_shape_and_a_sane_filter_bank():
    from cue.smart_turn import features, mel_filters
    f = features(np.zeros(16000, dtype=np.float32))
    assert f.shape == (1, 80, 800) and f.dtype == np.float32
    m = mel_filters()
    assert m.shape == (80, 201) and (m >= 0).all() and (m.sum(axis=1) > 0).all()


def test_fresh_shares_the_model_and_not_the_buffer():
    pytest.importorskip("onnxruntime")
    cached = sorted(Path.home().glob(".cache/huggingface/hub/models--pipecat-ai--smart-turn-v3/snapshots/*/*.onnx"))
    if not cached:
        pytest.skip("no Smart Turn model cached")
    from cue.smart_turn import SmartTurn
    a = SmartTurn(cached[0])
    b = a.fresh()
    a.push(np.zeros(1600, dtype=np.int16))
    assert b.session is a.session and b.samples == 0 and a.samples == 1600 and b.predict() == 0.5
