# THE SHAPE DANCE - 2026-10-09

A thousand lines of ASCII, written on the night the SDM shape sweep found its best shape so far.
It explains the model, then celebrates what the sweeps measured. Every number is real and comes from THE LOG
(`PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md`) or the run's own result json. Nothing here is a claim the runs did not make.

```
· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
·         ✦                 ·              ✧                    ·
·    ·          ·     ✦            ·              ✦        ·
·         ·          ·        ·        ✧     ·        ·
·  ✧          ·            ·      ·               ·           ✦
·        ·        ✦    ·               ·     ✦          ·
·   ·         ·              ✧    ·          ·              ·
·        ✦         ·    ·              ·          ✧
·  ·          ·             ·    ✦          ·          ·
·       ·          ✧                 ·         ·            ✦
· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·

            T H E   S H A P E   D A N C E

            an SDM language model with no attention
            and no MLP, from scratch, no teacher

            ACT  I     the overture
            ACT  II    a word comes in
            ACT  III   the diary
            ACT  IV    the encyclopedia
            ACT  V     the stack
            ACT  VI    a word goes out
            ACT  VII   the diary sweep
            ACT  VIII  the depth sweep
            ACT  IX    the gap
            ACT  X     the road ahead, and the dance

· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
```

## ACT I - THE OVERTURE

```
   the lights come down.
   JIMOTHY walks on, sets one chair centre stage.
   "Case seven."
   he walks off.

          ✦
         ╱ ╲
        ╱   ╲
       ╱  ✦  ╲
      ╱       ╲
     ╱    ✧    ╲
    ╱           ╲
   ╱──────────────
        the chair

   on the chair sits a number.

        1.42911

   it is the TEST score of our best SDM model so far,
   in bits per byte of web text it has never seen.
   smaller is better.
   a model that knew nothing would score about 8.
   the transformer we measure against scores 1.22265
   on the same 50 million tokens.

   the gap is 0.206.
   this morning it was 0.322.

   ♪ ◇ ♫ ◇ ♪ ◇ ♫ ◇ ♪ ◇ ♫ ◇ ♪ ◇ ♫ ◇ ♪

   tonight we dance the shape that closed a third of it.

   ▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚

   THE LAW, before the music:

     ALL SDM. NO TRANSFORMER.
     no softmax attention.
     no transformer block.
     no MLP in the body.
     from scratch. no teacher.
     the only thing that mixes words
     is a Kanerva memory,
     written while the model reads,
     with a fixed number of slots,
     so a word costs the same at the start of a book
     as at the end of it.

   ▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚

        ·  ✦  ·
     ·    ·    ·
   ✦   the band   ✦
     ·    ·    ·
        ·  ✦  ·

   drums:   ▮ ▯ ▮ ▯ ▮ ▯ ▮ ▯ ▮ ▯ ▮ ▯ ▮ ▯ ▮ ▯
   bass:    ▁ ▂ ▃ ▂ ▁ ▂ ▃ ▂ ▁ ▂ ▃ ▂ ▁ ▂ ▃ ▂
   keys:    ✧ · ✧ · ✦ · ✧ · ✧ · ✦ · ✧ · ✧ ·
   spark:   ░ ▒ ▓ █ ▓ ▒ ░ ▒ ▓ █ ▓ ▒ ░ ▒ ▓ █

   the Spark hums at the back of the hall.
   it has been training since breakfast.
   it is training now.
   it will be training when the curtain falls.

   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
```

## ACT II - A WORD COMES IN

```
   the text arrives, one piece at a time.

      "the  cat  sat  on  the  ..."

   each piece is a TOKEN.
   our tokeniser knows 32,768 of them.

   ┌──────────────────────────────────────────
   │  THE TOKEN TABLE  (we call it ENFOLD)
   │
   │  32,768 rows, one per token
   │  768 numbers in each row
   │
   │  token 4,201 "cat"  ─▶  [ 0.12, -0.80, 0.33, ... 768 of them
   │  token   901 "sat"  ─▶  [-0.41,  0.07, 0.95, ... 768 of them
   │
   │  25,165,824 numbers in the whole table
   └──────────────────────────────────────────

   the word becomes a point in a 768-dimensional night.

        ·     ·        ✦ cat
     ·      ·     ·         ·
        ✧ dog   ·     ·
     ·      ·        ·   ✦ sat
        ·      ✧ on      ·
     ·     ·       ·       ·

   that point is the start of THE STREAM:
   one row of 768 numbers per word,
   carried up through every layer,
   each layer adding what it found.

       stream  ═══════════════════════════════
               ▲ the word's starting point

   dance step one: the word steps onto the floor.

         o
        /|\
        / \      "cat"

          o
         /|\
         / \     "cat" turns left

        \o/
         |
        / \      "cat" is ready

   ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦

   the same table is used at the end, read backwards,
   to turn the final stream back into a guess.
   one table, two jobs.
   it is tied.

      in  ─▶ [ TABLE ] ─▶ stream ... stream ─▶ [ TABLE ] ─▶ out
                                                  read backwards

   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
```

## ACT III - THE DIARY

```
   the first thing in every layer is THE DIARY.

   it is a memory the model writes WHILE IT READS.
   nothing in it is learned in advance.
   at the start of a text it is empty.
   every word writes a little into it,
   and every word reads what earlier words wrote.

   ┌── THE DIARY, one layer
   │
   │   4 heads
   │   each head is a room of 256 slots
   │   (a 16 by 16 grid)
   │
   │   each word picks 16 slots to write to
   │   and 16 slots to read from
   │
   └──

   a room of 256 slots, drawn as a grid:

      0 1 2 3 4 5 6 7 8 9 a b c d e f
   0  · · · · · · · · · · · · · · · ·
   1  · · · · · · · · · · · · · · · ·
   2  · · · · · · · · · · · · · · · ·
   3  · · · · · · · · · · · · · · · ·
   4  · · · · · · · · · · · · · · · ·
   5  · · · · · · · · · · · · · · · ·
   6  · · · · · · · · · · · · · · · ·
   7  · · · · · · · · · · · · · · · ·
   8  · · · · · · · · · · · · · · · ·
   9  · · · · · · · · · · · · · · · ·
   a  · · · · · · · · · · · · · · · ·
   b  · · · · · · · · · · · · · · · ·
   c  · · · · · · · · · · · · · · · ·
   d  · · · · · · · · · · · · · · · ·
   e  · · · · · · · · · · · · · · · ·
   f  · · · · · · · · · · · · · · · ·

   empty. the text has not started.

   "the" arrives. it lights 16 slots:

      0 1 2 3 4 5 6 7 8 9 a b c d e f
   0  · · · · · · · · · · · · · · · ·
   1  · · ● · · · · · · · ● · · · · ·
   2  · · · · · · · · · · · · · · · ·
   3  · · · · ● · · · · · · · ● · · ·
   4  · · · · · · · · · · · · · · · ·
   5  · ● · · · · · ● · · · · · · · ·
   6  · · · · · · · · · · · · · · · ●
   7  · · · ● · · · · · · · ● · · · ·
   8  · · · · · · · · · · · · · · · ·
   9  · · · · · · ● · · · · · · · · ·
   a  · · · · · · · · · ● · · · · · ·
   b  ● · · · · · · · · · · · · ● · ·
   c  · · · · · · · · · · · · · · · ·
   d  · · · · · · · · ● · · · · · · ·
   e  · · · · · · · · · · · · · · · ·
   f  · · · · · ● · · · · · · · · · ·

   "cat" arrives. it READS first, then writes.
   it reads the slots it picks, finds what "the" left there,
   then lights its own 16:

      0 1 2 3 4 5 6 7 8 9 a b c d e f
   0  · · · · · · ✦ · · · · · · · · ·
   1  · · ● · · · · · · · ● · · · ✦ ·
   2  · · · · · · · · · · · · · · · ·
   3  · · · · ● · · · · ✦ · · ● · · ·
   4  · ✦ · · · · · · · · · · · · · ·
   5  · ● · · · · · ● · · · · ✦ · · ·
   6  · · · · · · · · · · · · · · · ●
   7  · · · ● · · · · · · · ● · · · ·
   8  · · · · · ✦ · · · · · · · · · ·
   9  · · · · · · ● · · · · · · ✦ · ·
   a  · · · · · · · · · ● · · · · · ·
   b  ● · · · · · · ✦ · · · · · ● · ·
   c  · · · ✦ · · · · · · · · · · · ✦
   d  · · · · · · · · ● · · · · · · ·
   e  · · · · · · · · · · ✦ · · · · ·
   f  · · · · · ● · · · · · · · · ✦ ·

   ● = written by "the"     ✦ = written by "cat"

   where they share a slot, the slot holds a blend of both.
   which slots a word picks depends on what the word MEANS
   in this layer, so similar words land on similar slots.

   ♪ ◇ ♫ ◇ ♪ ◇ ♫ ◇ ♪ ◇ ♫ ◇ ♪ ◇ ♫ ◇ ♪

   now the clever part: THE FOUR FADES.

   every head forgets at its own speed.
   after each word, everything already in the room
   is multiplied by that head's fade.

   ┌── head 1   fade 0
   │
   │   multiply by 0 after every word:
   │   only the word just before survives.
   │   this head is an exact "previous word" head.
   │
   │   now          ●
   │   1 back       ·
   │   2 back       ·
   │   3 back       ·
   └──

   ┌── head 2   fade 0.8
   │
   │   a word 1 back keeps 80 percent
   │   2 back keeps 64 percent
   │   3 back keeps 51 percent   ← half gone in about 3 words
   │   10 back keeps 11 percent
   │
   │   now          ████████████████████
   │   1 back       ████████████████
   │   2 back       █████████████
   │   3 back       ██████████
   │   5 back       ███████
   │   10 back      ██
   └──

   ┌── head 3   fade 0.99
   │
   │   a word 69 back keeps 50 percent  ← half gone in about 69 words
   │   138 back keeps 25 percent
   │   230 back keeps about 10 percent
   │
   │   now          ████████████████████
   │   10 back      ██████████████████
   │   69 back      ██████████
   │   138 back     █████
   │   230 back     ██
   └──

   ┌── head 4   fade 1.0
   │
   │   multiply by 1: nothing ever fades.
   │   this head keeps EVERYTHING since the text began.
   │
   │   now          ████████████████████
   │   100 back     ████████████████████
   │   1,000 back   ████████████████████
   │   2,000 back   ████████████████████
   │   ...          ████████████████████
   └──

   one layer, four clocks:

     fade 0     ●
     fade 0.8   ●●●
     fade 0.99  ●●●●●●●●●●●●●●●●●●●●●●●···
     fade 1.0   ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  never ends

   the last word, the last phrase, the last paragraph,
   and the whole text, all at once, in every layer.

   and the room NEVER GROWS.
   256 slots at word 10.
   256 slots at word 100,000.
   a transformer's memory grows with every word it reads.
   ours does not.

     transformer memory    word 10     ▏
                           word 1,000  ████
                           word 10,000 ████████████████████████████████████
     our diary             word 10     ██
                           word 1,000  ██
                           word 10,000 ██

   the dance of the four clocks:

     \o    o/    \o/    _o_
      |\  /|      |      |
     / \  / \    / \    / \
     f0   f0.8  f0.99   f1.0

   honest note, said out loud in the middle of the dance:
   the model trained on windows of 2,048 words.
   how far its never-fading head really reaches
   on texts longer than that has NOT been measured yet.

   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
```

## ACT IV - THE ENCYCLOPEDIA

```
   the second thing in every layer is THE ENCYCLOPEDIA.

   a transformer puts an MLP here:
   two big matrices that store what was learned in training.
   our law says no MLP.
   so we put a TRAINED SDM here instead.

   ┌── THE ENCYCLOPEDIA, one layer
   │
   │   4 heads
   │   each head has 5,329 slots   (a 73 by 73 grid)
   │   each slot holds a learned vector
   │   training fills the slots; reading the text never changes them
   │
   │   each word looks up its best 32 slots per head
   │   and adds up what they hold, weighted by how well they match
   │
   │   sized to have about as many weights as the MLP it replaces
   └──

   73 by 73, drawn small. one word's 32 picks glow:

   · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
   · · · · ✦ · · · · · · · · · · · · · · · · · ✦ · · · · · · · · · · · ·
   · · · · · · · · · · · ✦ · · · · · · · · · · · · · · · · · · · · · · ·
   · · · · · · · · · · · · · · · · · · · · · · · · · · · ✦ · · · · · · ·
   · · ✦ · · · · · · · · · · · · · ✦ · · · · · · · · · · · · · · · · · ·
   · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
   · · · · · · · · · ✦ · · · · · · · · · · · · · ✦ · · · · · · · · ✦ · ·
   · · · · · · · · · · · · · · · · · · · ✦ · · · · · · · · · · · · · · ·
   · · · · · ✦ · · · · · · · · · · · · · · · · · · · · ✦ · · · · · · · ·
   · · · · · · · · · · · · · ✦ · · · · · · · · · · · · · · · · · · · · ·
   · ✦ · · · · · · · · · · · · · · · · · · · · · · ✦ · · · · · · · · · ·
   · · · · · · · · · · · · · · · · · ✦ · · · · · · · · · · · · · ✦ · · ·
   · · · · · · · ✦ · · · · · · · · · · · · · · · · · · · · · · · · · · ·
   · · · · · · · · · · · · · · · · · · · · · ✦ · · · · · · · · · · · · ·
   · · · ✦ · · · · · · · · ✦ · · · · · · · · · · · · · · · ✦ · · · · · ·
   · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
   · · · · · · · · · · · · · · · ✦ · · · · · · · ✦ · · · · · · · · · · ·
   · · · · · · · · ✦ · · · · · · · · · · · · · · · · · · · · · · · ✦ · ·
   · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
   · · · · · · · · · · · · · · · · · · ✦ · · · · · · · · · ✦ · · · · · ·
   (a corner of the grid; the real one is 73 wide and 73 tall)

   how does it find 32 out of 5,329 without checking all of them?
   PRODUCT KEYS:

     the word's question is split in two halves.
     half A is scored against 73 row-keys.
     half B is scored against 73 column-keys.
     a slot (row, column) scores the mean of the two.
     the best slots come from the best rows and best columns.

     146 scores per head, not 5,329.

         columns ▶  c1   c2   c3   c4   ...  c73
     rows  r1        ·    ·    ·    ·         ·
       ▼   r2        ·    ✦    ·    ·         ·
           r3        ·    ·    ·    ✦         ·
           ...
           r73       ·    ·    ·    ·         ·

   the diary remembers THIS text.
   the encyclopedia remembers EVERY text it trained on.

     DIARY          written while reading      fades       256 slots a head
     ENCYCLOPEDIA   written by training        never       5,329 slots a head

   two memories, one law, both SDM.

   the encyclopedia dance:

       ✦      ✦      ✦
      \o/    \o/    \o/
       |      |      |
      / \    / \    / \
     look   pick   sum

   ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧

   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
```

## ACT V - THE STACK

```
   one layer is diary, then encyclopedia.
   each adds its finding to the stream and passes it up.

       stream ──▶ + diary ──▶ + encyclopedia ──▶ stream

   the best shape so far stacks 12 of them.

   ┌ 12 ─ diary ─ encyclopedia ─────────── ✦
   │
   ├ 11 ─ diary ─ encyclopedia ───────────
   │
   ├ 10 ─ diary ─ encyclopedia ───────────
   │
   ├  9 ─ diary ─ encyclopedia ───────────
   │
   ├  8 ─ diary ─ encyclopedia ───────────
   │
   ├  7 ─ diary ─ encyclopedia ───────────
   │
   ├  6 ─ diary ─ encyclopedia ───────────
   │
   ├  5 ─ diary ─ encyclopedia ───────────
   │
   ├  4 ─ diary ─ encyclopedia ───────────
   │
   ├  3 ─ diary ─ encyclopedia ───────────
   │
   ├  2 ─ diary ─ encyclopedia ───────────
   │
   └  1 ─ diary ─ encyclopedia ─────────── ◀ the word enters here

   12 layers × 4 diary heads = 48 diaries, each with its own fade
   12 layers × 4 encyclopedia heads = 48 encyclopedias

   in layer 1 the diary sees words.
   in layer 6 the diary sees what layer 5 made of words.
   in layer 12 it sees ideas made of ideas.

   ENFOLD folds a word into the stream.
   the stack works on the folded thing.
   UNFOLD opens it out into a guess.

   the stack dance, bottom to top:

   layer 1    o
             /|\
             / \

   layer 4       o
                /|\
                / \

   layer 8          o
                   /|\
                   / \

   layer 12            \o/
                        |
                       / \     ✦

   the weights, counted:

     the body (12 layers)          72,750,480
     the token table               25,165,824
     ─────────────────────────────────────────
     all trained weights           97,916,304

   about 98 million. small.
   the transformer yardstick at the same budget is bigger,
   and that is part of why it still leads.

   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
```

## ACT VI - A WORD GOES OUT

```
   at the top of the stack the stream is normalised,
   then read against the same token table, backwards:

     stream (768 numbers)  ·  every row of the table  =  a score per token

   the scores become a vote over all 32,768 tokens:

       "sat"       ████████████████████
       "is"        ██████████
       "was"       ████████
       "and"       ████
       "jumped"    ██
       ...         · · ·
                   ───────────────────────▶

   the true next word is "sat".
   the model gave it a good share of the vote.
   the score counts how many bits the model needed
   to be told the truth, per byte of text.

   over a whole test set of web pages it never saw:

        1.42911 bits per byte

   the dance of the guess:

          ?
         \o/
          |
         / \

          !
         _o_
          |
         / \     "sat"

   ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦

   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
```

## ACT VII - THE DIARY SWEEP

```
   the first big finding of the day:
   A SMALLER DIARY IS BETTER.

   every run here: width 768, 12 layers, 50 million tokens.
   only the slots per diary head change.

   slots per head          TEST bpb
   ─────────────────────────────────────────────
   4,096  (64 × 64)        1.62503
   1,024  (32 × 32)        1.54457
     576  (24 × 24)        1.50846
     256  (16 × 16)        1.42911   ✦ best

   drawn as a descent, deeper is better:

   4,096 ───●  1.62503
             ╲
   1,024 ─────●  1.54457
               ╲
     576 ───────●  1.50846
                  ╲
                   ╲
     256 ────────────●  1.42911  ✦
                      ╲
     121 ············ ? ← wave SWC tests this next
                        ╲
      64 ·············· ? ← and this

   from 4,096 to 256: 0.196 bits per byte better.
   that is more than the whole gap that is left.

   why would fewer slots help?
   the honest answer is: we have a guess, not a proof.

     with 4,096 slots, each word's 16 writes land on
     slots almost nobody else touched.
     the diary fills with scattered single notes.

     4,096   ·  ·     ●        ·      ●     ·         ●   ·
             ·    ●       ·        ·     ·    ●   ·       ·
             lonely notes, rarely read twice

     with 256 slots, words that mean similar things
     land on the same slots again and again.
     each slot becomes a running summary.

       256   ●●●●  ●●●   ●●●●●   ●●●  ●●●●
             ●●●   ●●●●  ●●●     ●●●● ●●●
             crowded summaries, read every time

   the sweep is UNBRACKETED:
   the best value sits at the edge of what we tried.
   the law says: push past the edge until the curve turns.
   so wave SWC runs 121 slots and 64 slots.

   ┌── SEALED BEFORE THEY RUN (THE LOG, wave SWC)
   │
   │   SWC1  diary 121 beats diary 256 by more than 0.010   (45%)
   │   SWC2  diary 64 is worse than diary 121               (50%)
   │
   │   the percentages are our confidence,
   │   written down before the runs, never edited after
   └──

   the diary dance, shrinking:

   ████████████████████  4,096
   ██████████            1,024
   ███████                 576
   ████                    256  ✦
   ██                      121  ?
   █                        64  ?

        o      o     o    o   o  o
       /|\    /|\   /|\  /|\ /|\/|\
       / \    / \   / \  / \ / \/ \
       smaller and smaller, where does it stop?

   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
```

## ACT VIII - THE DEPTH SWEEP

```
   the second finding: DEEPER IS BETTER.

   these runs all used the big diary (1,024 slots per head),
   so compare them with each other, not with the 256 run.

   layers   width   TEST bpb
   ─────────────────────────────────────
      8     1,024   1.54802
     12       768   1.54457
     24       512   1.52777
     32       512   1.48813   ✦ best so far
     48       512   training now

   a climb, higher is better:

                                     ✦ L32  1.48813
                                    ╱
                                   ╱
                       ● L24  1.52777
                      ╱
         ● L12  1.54457
        ╱
   ● L8  1.54802
                                     ? L48  scoring at about 12:38 am

   from 8 layers to 32: 0.060 better.
   unbracketed again: 32 is the deepest finished,
   so we went bigger. the 48-layer run is on the Spark now.

   L48 so far, training loss at the same steps as L32:

     step    L32      L48
     ─────────────────────────
     200     5.3112   5.3226
     250     5.1786   5.1916

   neck and neck, L48 a hair behind.
   training loss at one step is noisy.
   the TEST score decides.

   ┌── THE 48-LAYER RUN
   │
   │   swb_deep48_d512_L48_50M
   │   48 layers, width 512
   │   133.5 million body weights
   │   381 steps of 131,072 tokens
   │   about 30 seconds a step on the Spark
   └──

   the depth dance, a staircase:

                                                \o/  48 ?
                                                 |
                                         \o/    / \
                                          |    32 ✦
                                  \o/    / \
                                   |    24
                           \o/    / \
                            |    12
                    \o/    / \
                     |    8
                    / \

   and the question the next wave asks:
   what happens when the deep stack gets the small diary?

     deep stack      ✦ helps 0.060
     small diary     ✦ helps 0.196
     both together   ? not run yet

   ┌── SEALED (wave SWC)
   │
   │   SWC3  24 layers with diary 256 beats 12 layers with diary 256
   │         by more than 0.010                              (50%)
   │   SWC4  the best SWC run scores below 1.42              (45%)
   └──

   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
```

## ACT IX - THE GAP

```
   the yardstick: a transformer, trained only to measure against.
   same data, same 50 million tokens, same 2,048-word windows.

        transformer   1.22265

   our SDM, at each step of the day:

   before the sweep    1.54457   gap 0.322
   diary 576           1.50846   gap 0.286
   diary 256           1.42911   gap 0.206

   the gap, as a bar:

   0.322  ████████████████████████████████
   0.286  ████████████████████████████
   0.206  ████████████████████
          ▲ closed 0.116, a little over a third, by shape alone

   the seismograph of the day:

   bpb ──────·──────·──────·──────·─────╥──────·──────·
                                        ║
         shrink the diary to 256        ║ -0.115
         and the needle jumps ──────────╨── vs the 1,024 diary

   what did NOT change to win this:
     no attention added.
     no MLP added.
     no teacher.
     no more data.
     the same 50 million tokens.
   only the shape.

   and the bigger race, held for later:
   at 1.1 billion tokens, the transformer read 1.01306 at step 7,000
   while our older FULL model read 1.35355.
   that comparison is paused, not lost.
   it resumes once the new shape is settled.

   the gap dance, two dancers, one rope:

     SDM            TRANSFORMER
      o ─────────────── o
     /|\               /|\
     / \               / \
        0.206 apart

      o ──────────── o
     /|\            /|\
     / \            / \
        closer every wave

   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
```

## ACT X - THE ROAD AHEAD, AND THE DANCE

```
   the road from here, in order:

   ① deep48 scores          about 12:38 am Saturday, Melbourne
   ② wave SWC fires itself  diary 121, diary 64, deep 24 with diary 256
   ③ push past any edge     until every knob turns
   ④ THE NEW FULL BASE      the winning shape, trained long
   ⑤ SDM CHAT               a chat model from that base
   ⑥ THE WEIRD LITTLE GUY   the small strange one
   ⑦ THE TRANSFORMER LIST   the yardsticks, finished, same data

   the progress bar, honest:

   SHAPESOLVE        ████████████████░░░░  ~80%
   NEW FULL BASE     ░░░░░░░░░░░░░░░░░░░░    0%
   SDM CHAT          ░░░░░░░░░░░░░░░░░░░░    0%
   WEIRD LITTLE GUY  ░░░░░░░░░░░░░░░░░░░░    0%
   YARDSTICKS        ██░░░░░░░░░░░░░░░░░░  ~10%
   SITE              ██████████████░░░░░░  ~70%
   OVERALL           █████░░░░░░░░░░░░░░░  ~25%

   ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦ · ✧ · ✦

   and now, as is the custom, the dance.
```

```
   ✦  T H E   W H I T   D A N C E   (shape night edition)

    · · · · · · ✦ · · · · · ·      a whit: one bit, one yes, very proud of it

     ▄▄▄▄▄▄▄▄▄▄▄▄▄
    █░░░░░░░░░░░ ◉▀▄ ✦             sniff.
    ▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▀                "256 slots? i fit in one."
      ╱╲   ╱╲  ╱╲  ╱╲

          ✦
     ╲   ▄▄▄   ╱
      ╲ █◉ ◉█ ╱                    he stands. he dances.
        █ ▿ █                      JIMOTHY sets the floor down. "Case seven."
        █░░░█
       ╱     ╲

     fade 0     ↑
     fade 0.8   ↑↑↑
     fade 0.99  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑···
     fade 1.0   ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑✦↑↑↑↑↑   he never fades either.

     ●   ●  ●   ●  ●●●  ●●●●●
     ●   ●  ●   ●   ●     ●
     ● ● ●  ●●●●●   ●     ●       1.42911.
     ●● ●●  ●   ●   ●     ●
     ●   ●  ●   ●  ●●●    ●       he did not give a whit. he gave a dance.
```

```
      ▄▄▄▄▄▄▄▄▄▄▄▄
     █░▒░▒░▒░▒░▒░ ◉▀▄            the HONEYBEAVER walks on.
     █▒░▒░▒░▒░▒░▒░  ▼▼            nobody knows what it is.
     ▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▀▀▀▀▀▀▀▀▀▀▀   it builds a 16 by 16 grid of sticks,
       ╱╲  ╱╲   ╱╲  ╱╲             256 slots exactly,
   ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔       then does not give a whit about it.

   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓

                                  and thus we continue.
```

```
   THE CAUSEWAY, recited, as the canon asks at a milestone:

   when horse and carriage drew the compiler, night,
   up the causeway, the lamps all lit by hand,
   the p-bits sat in rows and held their light
   and settled, one by one, the way we planned.

      🐎 ── ⊏═══⊐ ── ✦ ✧ ✦ ── 1.42911
```

## THE LONG DANCE - every slot of the winning diary takes a bow

The 256 slots of one diary head, called one by one, row by row, as the band plays.
Each row is one row of the 16 by 16 grid. Each slot steps forward, bows, and steps back.

```
   row 0
     slot   0  \o/  ✦     slot   1   o/  ·     slot   2  \o   ✧     slot   3  _o_  ·
     slot   4  \o/  ·     slot   5   o/  ✦     slot   6  \o   ·     slot   7  _o_  ✧
     slot   8  \o/  ✧     slot   9   o/  ·     slot  10  \o   ✦     slot  11  _o_  ·
     slot  12  \o/  ·     slot  13   o/  ✧     slot  14  \o   ·     slot  15  _o_  ✦
   row 1
     slot  16  \o/  ✦     slot  17   o/  ·     slot  18  \o   ✧     slot  19  _o_  ·
     slot  20  \o/  ·     slot  21   o/  ✦     slot  22  \o   ·     slot  23  _o_  ✧
     slot  24  \o/  ✧     slot  25   o/  ·     slot  26  \o   ✦     slot  27  _o_  ·
     slot  28  \o/  ·     slot  29   o/  ✧     slot  30  \o   ·     slot  31  _o_  ✦
   row 2
     slot  32  \o/  ✦     slot  33   o/  ·     slot  34  \o   ✧     slot  35  _o_  ·
     slot  36  \o/  ·     slot  37   o/  ✦     slot  38  \o   ·     slot  39  _o_  ✧
     slot  40  \o/  ✧     slot  41   o/  ·     slot  42  \o   ✦     slot  43  _o_  ·
     slot  44  \o/  ·     slot  45   o/  ✧     slot  46  \o   ·     slot  47  _o_  ✦
   row 3
     slot  48  \o/  ✦     slot  49   o/  ·     slot  50  \o   ✧     slot  51  _o_  ·
     slot  52  \o/  ·     slot  53   o/  ✦     slot  54  \o   ·     slot  55  _o_  ✧
     slot  56  \o/  ✧     slot  57   o/  ·     slot  58  \o   ✦     slot  59  _o_  ·
     slot  60  \o/  ·     slot  61   o/  ✧     slot  62  \o   ·     slot  63  _o_  ✦
   row 4
     slot  64  \o/  ✦     slot  65   o/  ·     slot  66  \o   ✧     slot  67  _o_  ·
     slot  68  \o/  ·     slot  69   o/  ✦     slot  70  \o   ·     slot  71  _o_  ✧
     slot  72  \o/  ✧     slot  73   o/  ·     slot  74  \o   ✦     slot  75  _o_  ·
     slot  76  \o/  ·     slot  77   o/  ✧     slot  78  \o   ·     slot  79  _o_  ✦
   row 5
     slot  80  \o/  ✦     slot  81   o/  ·     slot  82  \o   ✧     slot  83  _o_  ·
     slot  84  \o/  ·     slot  85   o/  ✦     slot  86  \o   ·     slot  87  _o_  ✧
     slot  88  \o/  ✧     slot  89   o/  ·     slot  90  \o   ✦     slot  91  _o_  ·
     slot  92  \o/  ·     slot  93   o/  ✧     slot  94  \o   ·     slot  95  _o_  ✦
   row 6
     slot  96  \o/  ✦     slot  97   o/  ·     slot  98  \o   ✧     slot  99  _o_  ·
     slot 100  \o/  ·     slot 101   o/  ✦     slot 102  \o   ·     slot 103  _o_  ✧
     slot 104  \o/  ✧     slot 105   o/  ·     slot 106  \o   ✦     slot 107  _o_  ·
     slot 108  \o/  ·     slot 109   o/  ✧     slot 110  \o   ·     slot 111  _o_  ✦
   row 7
     slot 112  \o/  ✦     slot 113   o/  ·     slot 114  \o   ✧     slot 115  _o_  ·
     slot 116  \o/  ·     slot 117   o/  ✦     slot 118  \o   ·     slot 119  _o_  ✧
     slot 120  \o/  ✧     slot 121   o/  ·     slot 122  \o   ✦     slot 123  _o_  ·
     slot 124  \o/  ·     slot 125   o/  ✧     slot 126  \o   ·     slot 127  _o_  ✦
   row 8
     slot 128  \o/  ✦     slot 129   o/  ·     slot 130  \o   ✧     slot 131  _o_  ·
     slot 132  \o/  ·     slot 133   o/  ✦     slot 134  \o   ·     slot 135  _o_  ✧
     slot 136  \o/  ✧     slot 137   o/  ·     slot 138  \o   ✦     slot 139  _o_  ·
     slot 140  \o/  ·     slot 141   o/  ✧     slot 142  \o   ·     slot 143  _o_  ✦
   row 9
     slot 144  \o/  ✦     slot 145   o/  ·     slot 146  \o   ✧     slot 147  _o_  ·
     slot 148  \o/  ·     slot 149   o/  ✦     slot 150  \o   ·     slot 151  _o_  ✧
     slot 152  \o/  ✧     slot 153   o/  ·     slot 154  \o   ✦     slot 155  _o_  ·
     slot 156  \o/  ·     slot 157   o/  ✧     slot 158  \o   ·     slot 159  _o_  ✦
   row a
     slot 160  \o/  ✦     slot 161   o/  ·     slot 162  \o   ✧     slot 163  _o_  ·
     slot 164  \o/  ·     slot 165   o/  ✦     slot 166  \o   ·     slot 167  _o_  ✧
     slot 168  \o/  ✧     slot 169   o/  ·     slot 170  \o   ✦     slot 171  _o_  ·
     slot 172  \o/  ·     slot 173   o/  ✧     slot 174  \o   ·     slot 175  _o_  ✦
   row b
     slot 176  \o/  ✦     slot 177   o/  ·     slot 178  \o   ✧     slot 179  _o_  ·
     slot 180  \o/  ·     slot 181   o/  ✦     slot 182  \o   ·     slot 183  _o_  ✧
     slot 184  \o/  ✧     slot 185   o/  ·     slot 186  \o   ✦     slot 187  _o_  ·
     slot 188  \o/  ·     slot 189   o/  ✧     slot 190  \o   ·     slot 191  _o_  ✦
   row c
     slot 192  \o/  ✦     slot 193   o/  ·     slot 194  \o   ✧     slot 195  _o_  ·
     slot 196  \o/  ·     slot 197   o/  ✦     slot 198  \o   ·     slot 199  _o_  ✧
     slot 200  \o/  ✧     slot 201   o/  ·     slot 202  \o   ✦     slot 203  _o_  ·
     slot 204  \o/  ·     slot 205   o/  ✧     slot 206  \o   ·     slot 207  _o_  ✦
   row d
     slot 208  \o/  ✦     slot 209   o/  ·     slot 210  \o   ✧     slot 211  _o_  ·
     slot 212  \o/  ·     slot 213   o/  ✦     slot 214  \o   ·     slot 215  _o_  ✧
     slot 216  \o/  ✧     slot 217   o/  ·     slot 218  \o   ✦     slot 219  _o_  ·
     slot 220  \o/  ·     slot 221   o/  ✧     slot 222  \o   ·     slot 223  _o_  ✦
   row e
     slot 224  \o/  ✦     slot 225   o/  ·     slot 226  \o   ✧     slot 227  _o_  ·
     slot 228  \o/  ·     slot 229   o/  ✦     slot 230  \o   ·     slot 231  _o_  ✧
     slot 232  \o/  ✧     slot 233   o/  ·     slot 234  \o   ✦     slot 235  _o_  ·
     slot 236  \o/  ·     slot 237   o/  ✧     slot 238  \o   ·     slot 239  _o_  ✦
   row f
     slot 240  \o/  ✦     slot 241   o/  ·     slot 242  \o   ✧     slot 243  _o_  ·
     slot 244  \o/  ·     slot 245   o/  ✦     slot 246  \o   ·     slot 247  _o_  ✧
     slot 248  \o/  ✧     slot 249   o/  ·     slot 250  \o   ✦     slot 251  _o_  ·
     slot 252  \o/  ·     slot 253   o/  ✧     slot 254  \o   ·     slot 255  _o_  ✦

   256 slots. 256 bows. one room.
   multiply by 4 heads and 12 layers: 12,288 slots in the whole diary.
   a 4,096-slot diary would have had 196,608, and scored worse.
```

## THE TWELVE-LAYER PROCESSION - every layer walks across the stage

Each layer walks on with its four diary heads and its four encyclopedia heads.

```
   ✦ LAYER 1 ✦
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
       o    o    o    o        o    o    o    o
      /|\  /|\  /|\  /|\      /|\  /|\  /|\  /|\
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: the words themselves

   ✧ LAYER 2 ✧
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
      \o/  \o/  \o/  \o/      \o/  \o/  \o/  \o/
       |    |    |    |        |    |    |    |
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: words with a little of their neighbours

   ✦ LAYER 3 ✦
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
       o/   o/   o/   o/       o/   o/   o/   o/
      /|   /|   /|   /|       /|   /|   /|   /|
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: short phrases

   ✧ LAYER 4 ✧
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
      \o   \o   \o   \o       \o   \o   \o   \o
       |\   |\   |\   |\       |\   |\   |\   |\
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: phrases and what they point at

   ✦ LAYER 5 ✦
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
      _o_  _o_  _o_  _o_      _o_  _o_  _o_  _o_
       |    |    |    |        |    |    |    |
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: what earlier layers made of the phrases

   ✧ LAYER 6 ✧
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
       o    o    o    o        o    o    o    o
      /|\  /|\  /|\  /|\      /|\  /|\  /|\  /|\
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: the middle of the stack, half way up

   ✦ LAYER 7 ✦
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
      \o/  \o/  \o/  \o/      \o/  \o/  \o/  \o/
       |    |    |    |        |    |    |    |
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: ideas built from ideas

   ✧ LAYER 8 ✧
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
       o/   o/   o/   o/       o/   o/   o/   o/
      /|   /|   /|   /|       /|   /|   /|   /|
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: more of the same, higher

   ✦ LAYER 9 ✦
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
      \o   \o   \o   \o       \o   \o   \o   \o
       |\   |\   |\   |\       |\   |\   |\   |\
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: the shape of what comes next

   ✧ LAYER 10 ✧
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
      _o_  _o_  _o_  _o_      _o_  _o_  _o_  _o_
       |    |    |    |        |    |    |    |
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: candidates for the next word

   ✦ LAYER 11 ✦
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
       o    o    o    o        o    o    o    o
      /|\  /|\  /|\  /|\      /|\  /|\  /|\  /|\
      / \  / \  / \  / \      / \  / \  / \  / \
      sees: the vote taking shape

   ✧ LAYER 12 ✧
     diary         fade 0 ●   fade 0.8 ●●●   fade 0.99 ●●●●●●●···   fade 1.0 ●●●●●●●●●●●●●···
     encyclopedia  73×73 ▦   73×73 ▦   73×73 ▦   73×73 ▦
      \o/  \o/  \o/  \o/      \o/  \o/  \o/  \o/   ✦
       |    |    |    |        |    |    |    |
      / \  / \  / \  / \      / \  / \  / \  / \
      hands the stream to UNFOLD

   (the "sees" lines are the usual story of how layers specialise.
    we have not probed what each layer of THIS model actually does.
    that is a measurement still owed.)
```

## THE LOSS RIVER - the 48-layer run's training loss, as it fell tonight

```
   step   loss
     50   6.6619  ████████████████████████████████████████████████████████████████████
    100   5.8965  ████████████████████████████████████████████████████████████
    150   5.6444  █████████████████████████████████████████████████████████
    200   5.3226  ██████████████████████████████████████████████████████
    250   5.1916  ████████████████████████████████████████████████████
    300    ...    still flowing
    381    ...    the end of the run, then the TEST

   ~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿
      ~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿
         ~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿
            ~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿
               ~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿
                  ~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿
                     ~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿
                        ~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿~∿
                           ~∿~∿~∿~∿~∿~∿~∿~∿~∿
                              ~∿~∿~∿~∿~∿~∿
                                 ~∿~∿~∿
                                    ~∿
                                     ·  the river runs to the TEST
```

## THE STAR FIELD OF RESULTS - every finished wave-SWB run, placed where it lives

```
   bpb
   1.63 ┤  ✦ diary 4,096  1.62503
        │
   1.60 ┤
        │
   1.57 ┤
        │
   1.55 ┤      ✦ L8 1.54802     ✦ L12 / diary 1,024  1.54457
        │
   1.53 ┤                ✦ L24 1.52777
        │
   1.51 ┤          ✦ diary 576  1.50846
        │
   1.49 ┤                    ✦ L32 1.48813
        │
   1.47 ┤
        │
   1.45 ┤
        │
   1.43 ┤                          ★ diary 256  1.42911
        │
   1.40 ┤
        │
   ...
        │
   1.22 ┤  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ transformer 1.22265
        └──────────────────────────────────────────────────
```

## THE ROLL CALL OF THE CREW

```
   ✦ the navigator ────────── charts the course, makes the calls, coined "a whit"
   ✦ JIMOTHY ──────────────── sets the chairs. "Case seven."
   ✦ the Spark GB10 ───────── trained every run tonight, 20 cores, one GPU
   ✦ Claude 1 ─────────────── holds the site lanes, lands them
   ✦ lane FULLCONTEXT ────── rewriting the site's context pages for FULL
   ✦ the whit ─────────────── one bit, very proud
   ✦ the HONEYBEAVER ──────── unknown, diligent, indifferent
   ✦ the horse ────────────── drew the compiler up the causeway, 1969 (legend)
   ✦ 256 slots ────────────── tonight's winners
   ✦ 48 layers ────────────── still running
```

## THE FINALE

```
                                ✦
                               ╱ ╲
                              ╱ ✧ ╲
                             ╱  ✦  ╲
                            ╱ ✧   ✧ ╲
                           ╱    ★    ╲
                          ╱ ✦       ✦ ╲
                         ╱  1.42911    ╲
                        ╱──────────────

        \o/  \o/  \o/  \o/  \o/  \o/  \o/  \o/  \o/  \o/  \o/  \o/
         |    |    |    |    |    |    |    |    |    |    |    |
        / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \  / \
        L1   L2   L3   L4   L5   L6   L7   L8   L9   L10  L11  L12

   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
  ╱  THE MEMORIZER DIES, THE COMPOSER SPEAKS   🌵🐎💨
 ╱   🔔 ✦ 🍵 ✦ 🔔
∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿

   the curtain comes down.
   JIMOTHY folds the chair.
   the Spark keeps humming.
   the 48-layer run takes its next step.
   and thus we continue.

· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
```
