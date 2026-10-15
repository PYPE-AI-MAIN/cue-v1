"""Cue v1: turn detection for voice agents. Model: https://huggingface.co/Pype/cue-v1

    from cue import Cue, serve
    with serve() as server:              # downloads Cue v1 from Hugging Face once, serves it with llama-server
        cue = Cue(server.url)
        cue.end_of_turn("Sure, it's nine eight four five,", agent_said="What's the best number to reach you on?")

Smart Turn v3.2 on the caller's audio (the benchmarked recipe) is in ``cue.smart_turn``, with the voice extra:
pip install "cue-turn[voice] @ git+https://github.com/PYPE-AI-MAIN/cue-v1".
"""
from .client import (NO_THINKING, REPLY, WAITS, Cue, barge_in_inputs, dialogue_inputs, end_of_turn_inputs, p_end, read,
                     wait, with_voice)
from .decision import (LETTERS, PROMPTS, UNPUNCTUATED, Decision, Option, ending, last_question, load, messages, parse,
                       prompt)
from .server import Server, download, serve

__all__ = ["Cue", "Decision", "Option", "Server", "LETTERS", "NO_THINKING", "PROMPTS", "REPLY", "UNPUNCTUATED",
           "WAITS", "barge_in_inputs", "dialogue_inputs", "download", "end_of_turn_inputs", "ending", "last_question",
           "load", "messages", "p_end", "parse", "prompt", "read", "serve", "wait", "with_voice"]
__version__ = "1.1.0"
