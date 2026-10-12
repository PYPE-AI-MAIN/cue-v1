# Synthetic turn-taking conversations: the writing spec

Each line of a batch file is ONE conversation as a JSON object. Write natural, spoken conversations as a
speech recogniser would transcribe them, with every pause inside a turn marked and labelled. The text is
the training data for two decisions:

- end_of_turn, asked at every pause: DONE / LIKELY_DONE / MID_THOUGHT / HOLDING
- barge_in, asked when the listener makes a sound while the other person is talking: STOP / CONTINUE

## The JSON shape (exact keys, nothing else)

```json
{"type": "casual", "topic": "planning a friend's surprise birthday",
 "turns": [
   {"speaker": "A",
    "chunks": [
      {"text": "So I was thinking we could do it at Maya's place.", "pause": "HOLDING"},
      {"text": "Because her flat has that big terrace and", "pause": "MID_THOUGHT"},
      {"text": "it fits like thirty people easily.", "pause": "END_LIKELY"}],
    "listener": [{"after_chunk": 0, "text": "mm-hmm", "kind": "BACKCHANNEL"}]},
   {"speaker": "B",
    "chunks": [{"text": "Wait, does she even know about it?", "pause": "END_DONE"}]}
 ]}
```

- `type`: one of `casual`, `argumentative`, `collaborative`, `instructional`, `narrative`, `transactional`, `phone_agent`.
- `turns`: 14 to 28 turns, speakers alternating A, B, A, B... For `phone_agent`, A is always the agent and B the caller.
- `chunks`: the stretches of speech in that turn, split exactly where the speaker pauses. Each chunk's
  `pause` says what that pause is:
  - `MID_THOUGHT`: the sentence is left hanging. It stops on "and", "but", "because", "the", "with", "like",
    "so", "that", a filler ("um", "uh"), a repeated word, or a noun that needs more; or a number / address /
    list is only partly said; or they have only set up what they are about to say ("The thing is,").
  - `HOLDING`: the chunk IS a complete sentence (usually with a full stop), but the speaker is clearly going
    on: in a story, an explanation, a list of reasons, step-by-step instructions, an argument that continues.
    The pause is them thinking. THIS IS THE MOST IMPORTANT CASE: write many of them.
  - `END_DONE`: the turn ends and the floor passes: they asked a question, answered, made their point,
    reacted, agreed or declined.
  - `END_LIKELY`: the turn ends, but it is the kind of statement that could have had one more sentence.
  - Only the LAST chunk of a turn may be `END_DONE` or `END_LIKELY`, and the last chunk must be one of them
    (unless the turn is interrupted, below). Every earlier chunk is `MID_THOUGHT` or `HOLDING`.
- `listener` (optional): sounds the OTHER person makes during this turn, after chunk `after_chunk`:
  - `BACKCHANNEL`: "yeah", "mm-hmm", "okay", "right", "uh huh", "oh wow", "oh no", "i know", "yes", "sure",
    "hmm", "oh okay", "really?", a laugh written as "haha". The speaker keeps going.
  - `INTERRUPT`: the listener genuinely takes the floor: an objection, a correction, an urgent question,
    a "wait, wait", a change of subject. Write it as their opening words only ("wait sorry, not tuesday,
    i said thursday"). After an INTERRUPT the speaker stops: it must be attached to the turn's LAST
    chunk, and that last chunk gets `"pause": "INTERRUPTED"` (it is usually cut mid-sentence). The next
    turn belongs to the interrupter and starts by continuing what they began.

## Proportions to aim for (per batch, roughly)

- About 35% of all pauses are holds (`MID_THOUGHT` + `HOLDING`) and 65% are turn ends.
- Of the holds, about 70% are `HOLDING` (complete sentence, more coming) and 30% `MID_THOUGHT`.
- Of the turn ends: about 17% are questions, about 25% have NO end punctuation at all ("yeah", "i guess",
  "that's it", "so yeah", "anyway", "i don't know", "fair enough"), the rest statements.
- Many turns are a single chunk; longer turns (stories, explanations, instructions) have 2 to 6 chunks.
- About one backchannel every 3 to 4 multi-chunk turns; about one INTERRUPT per conversation.
- Chunk length: median about 7 words, sometimes 1 to 2 words ("um", "so yeah"), sometimes 25+.

## Texture: write it the way a recogniser hears speech

- Fillers and false starts: "um", "uh", "like", "you know", "i mean", "sort of", repeated words ("the the"),
  restarts ("we should, we could just"). Contractions, casual grammar.
- Punctuation as a recogniser adds it: mostly full stops, commas and question marks; sometimes none.
  Never write "..." (recognisers don't).
- Occasionally lowercase sentence starts. Numbers sometimes as words ("twenty two"), sometimes digits.
- Vary speakers: different ages, regions (American, British, Indian English, others), moods (calm, annoyed,
  excited, hesitant), and relationships (friends, coworkers, strangers, parent and child, customer and agent).

## Variety

Every conversation in a batch needs a different topic, and topics must not repeat across batches. For
`phone_agent`, cover: clinic and hospital appointments, lab reports, pharmacy refills, insurance claims, bank
cards, loans, telecom plans, electricity bills, food delivery problems, travel bookings, car service,
returns and refunds, school admissions, government services. The agent asks questions; callers answer in
fragments, read out numbers and dates in pieces (perfect MID_THOUGHT material: "my number is nine eight
four five" + pause + "two two one ..."), hesitate, go off on tangents, and interrupt.

Do NOT copy or paraphrase any real dataset or real conversation. Invent everything, including names.

## Barge-in batches (only when your instructions say "barge-in batch")

All conversations are `phone_agent`. The agent (A) has many multi-chunk turns: explaining a policy, listing
options, reading back details, giving instructions, asking a question at the end of a long turn. The caller
(B) makes 4 to 8 listener sounds per conversation during the agent's turns, labelled the way the production
barge_in decision defines them:

- `INTERRUPT` (the agent should STOP): a question, a request, an objection, a correction, "wait", "hello?",
  "ruko ek minute", "sorry, sorry", or a full answer to the question the agent has just finished asking
  ("Am I speaking with Ashish?" -> caller over her: "yes speaking").
- `BACKCHANNEL` (the agent should CONTINUE): only listening or not meant for her: "hmm", "haan", "ji",
  "okay", "achha", "right", "yes yes", laughter ("haha"), noise ("cough", "[noise]"), a fragment ("the"),
  a word to someone else in the room ("one sec beta"), or a bare "yes" when her question is NOT finished yet
  ("Would this be for you, or for" -> "yes" is CONTINUE).

Make roughly half the listener sounds BACKCHANNEL and half INTERRUPT, and include the hard ones on purpose:
short interrupts ("wait", "no no", "hello?"), long backchannels ("okay okay okay", "haan haan theek hai"),
and answers that arrive too early. The schema rule still holds: an INTERRUPT is attached to the LAST chunk
of the agent's turn, that chunk is "INTERRUPTED", and the caller's next turn continues what they began.
BACKCHANNELs can follow any chunk.
