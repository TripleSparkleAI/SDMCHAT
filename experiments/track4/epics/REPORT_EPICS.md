# REPORT · EPICS · recite and compose the public-domain epics (2026-10-02)

The question: can we recite every open long epic we can find, and train a small SDM model on them? This report has
the corpus, the recitation curve, the memories that ship, and the compose result against its control. Every number
names the file it comes from.

## 1. The corpus

35 works from Project Gutenberg, downloaded once, with the Gutenberg wrapper removed and nothing inside edited.
Every record in `EPICS_MANIFEST.json` carries the download URL, the catalog's own authors and issue date, the raw
sha256 and download time, the clean sha256, the cleaning rule and a licence basis. The builder is
`track4_epics_build_corpus.py`; the works list is `track4_epics_works.json`. Raw and clean texts are gitignored
(`raw/`, `clean/`); `samples/` holds each text's first 1,500 characters.

The licence basis has two parts, both computed. US: every work was first published before 1931. Life plus 70: every
death year in the catalog entry is before 1956. Two translations are public domain in the US and not yet under life
plus 70 (Needler's Nibelungenlied, d. 1962; Rose and Bacon's Cid, d. 1964). For the Mahabharata and the Pharsalia the
catalog names no translator death year, so life plus 70 is recorded as not established.

| work | translator / edition | first published | ebook | characters | US PD | life + 70 |
|---|---|---|---|---|---|---|
| The Iliad | Samuel Butler | 1898 | 2199 | 888,943 | yes | yes |
| The Odyssey | Samuel Butler | 1900 | 1727 | 678,823 | yes | yes |
| The Iliad | Alexander Pope (with Theodore Alois Buckley's notes) | 1720 | 6130 | 1,097,444 | yes | yes |
| The Odyssey | Alexander Pope | 1726 | 3160 | 698,310 | yes | yes |
| The Aeneid | John Dryden | 1697 | 228 | 696,961 | yes | yes |
| Beowulf | Francis Barton Gummere | 1910 | 981 | 147,247 | yes | yes |
| Paradise Lost | John Milton | 1674 | 26 | 459,534 | yes | yes |
| Paradise Regained | John Milton | 1671 | 58 | 90,704 | yes | yes |
| The Divine Comedy | Henry Wadsworth Longfellow | 1867 | 1004 | 662,725 | yes | yes |
| Kalevala, the Epic Poem of Finland | John Martin Crawford | 1888 | 5186 | 816,721 | yes | yes |
| An Old Babylonian Version of the Gilgamesh Epic | Morris Jastrow and Albert Tobias Clay | 1920 | 11000 | 226,153 | yes | yes |
| The Rámáyan of Válmíki | Ralph T. H. Griffith | 1874 | 24869 | 2,264,291 | yes | yes |
| The Mahabharata, Volume 1 (Books 1, 2 and 3) | Kisari Mohan Ganguli | 1883 | 15474 | 3,674,439 | yes | not established |
| The Mahabharata, Volume 2 (Books 4, 5, 6 and 7) | Kisari Mohan Ganguli | 1883 | 15475 | 3,942,125 | yes | not established |
| The Mahabharata, Volume 3 (Books 8 to 12) | Kisari Mohan Ganguli | 1883 | 15476 | 4,769,554 | yes | not established |
| The Mahabharata, Volume 4 (Books 13 to 18) | Kisari Mohan Ganguli | 1883 | 15477 | 2,561,145 | yes | not established |
| The Song of Hiawatha | Henry Wadsworth Longfellow | 1855 | 19 | 187,546 | yes | yes |
| Idylls of the King | Alfred, Lord Tennyson | 1885 | 610 | 476,737 | yes | yes |
| The Nibelungenlied | George Henry Needler | 1904 | 7321 | 637,368 | yes | no (d. 1962) |
| The Song of Roland | C. K. Scott-Moncrieff | 1919 | 391 | 192,263 | yes | yes |
| The Shah Nameh (abridged) | James Atkinson (as reprinted in The Persian Literature, vol. 1, ed. Richard Gottheil) | 1832 | 10315 | 726,024 | yes | yes |
| The Argonautica | R. C. Seaton | 1912 | 830 | 333,438 | yes | yes |
| The Lusiad | William Julius Mickle | 1776 | 32528 | 1,005,115 | yes | yes |
| Orlando Furioso | William Stewart Rose | 1823 | 615 | 1,792,153 | yes | yes |
| Jerusalem Delivered | Edward Fairfax | 1600 | 392 | 705,748 | yes | yes |
| Spenser's Faerie Queene, Vol. 1 | ed. J. C. Smith | 1909 | 70717 | 1,040,874 | yes | yes |
| Spenser's Faerie Queene, Vol. 2 | ed. J. C. Smith | 1909 | 72698 | 1,007,498 | yes | yes |
| The Metamorphoses of Ovid, Books I-VII | Henry Thomas Riley | 1851 | 21765 | 740,247 | yes | yes |
| The Metamorphoses of Ovid, Books VIII-XV | Henry Thomas Riley | 1851 | 26073 | 813,519 | yes | yes |
| The Poetic Edda | Henry Adams Bellows | 1923 | 73533 | 916,975 | yes | yes |
| Fragments of Ancient Poetry | James Macpherson (presented as Ossian) | 1760 | 8161 | 68,169 | yes | yes |
| The Mabinogion | Lady Charlotte Guest | 1849 | 5160 | 576,921 | yes | yes |
| The Story of Sigurd the Volsung and the Fall of the Niblungs | William Morris | 1876 | 18328 | 710,241 | yes | yes |
| The Lay of the Cid | R. Selden Rose and Leonard Bacon | 1919 | 6088 | 230,733 | yes | no (d. 1964) |
| Pharsalia | Sir Edward Ridley | 1896 | 602 | 475,616 | yes | not established |

Total: 35 works, 36,312,304 characters (36,502,734 UTF-8 bytes).

Choices recorded: Butler's prose Homer is the primary and Pope's verse the alternate; the Metamorphoses are Riley's
1851 prose (Garth and Dryden is not on Project Gutenberg); Gilgamesh is Jastrow and Clay's 1920 scholarly edition
(transliteration, translation and commentary), because the Thompson 1928 rendering sits behind a bot challenge at
its only host and was not fetched; the Shah Nameh is Atkinson's 1832 abridgement as reprinted in Gottheil's
collection, cut by heading (the Warner brothers' translation exists only as page scans); Ossian is the 1760
Fragments; all four Mahabharata volumes are in the corpus.

## 2. Recite: the curve

The memory is the poem lab's Kanerva SDM (`sites/settle-site/src/poemlab/poemsdm.js`): a hash of the last k
characters is a 256-bit address, every hard location within a Hamming radius adds the next character's bound code to
256 counters, and reading sums those counters. Error-driven writes, up to 16 passes. Measured on Beowulf (Gummere,
147,247 characters) and on Paradise Lost past that, in node on the Spark (`track4_epics_recite_cell.mjs`).
Word-perfect means the memory, cued with the first 48 characters, writes all N characters with no mistake.

Three things had to be fixed before the curve meant anything, each found by the curve itself:

1. **The hash.** Every memory size from 16,384 to 65,536 locations made the same 5 mistakes on Beowulf. The context
   address came from one 32-bit FNV hash, and Beowulf holds 5 pairs of different contexts with the same address
   (counted: 5 at 32 bits, 0 at 64). `ContextAddress` gained `hashBits: 64`; the poem lab's default stays 32 and its
   addresses are byte-identical to before. The first sweep is kept in `recite_curve/hash32_first_sweep/`.
2. **The radius.** A fixed radius of 111 wakes 2% of the locations, so a bigger memory wakes more of them per address
   and its writes interfere more: at radius 111, 65,536 locations held no more than 16,384 did (32,000 characters).
   Tuning the radius to wake about 30 locations doubled the capacity at 4,096 locations. `radiusFor(M)` applies it.
3. **The text.** Many epics repeat themselves. Hiawatha has 111 positions whose 48 characters of context are
   followed by a different character elsewhere in the poem; no 48-character memory can recite those. Each epic now
   gets the shortest context of 48, 96, 192 or 384 characters after which no position is ambiguous.

The frontier, 64-bit hash, tuned radius (`recite_curve/SUMMARY.md`, cells in `recite_curve/hash64_tuned/` and
`hash64_refine/`):

| hard locations | bits | radius | memory bytes | word-perfect up to N | first N that failed (errors) | bytes per char | gzip bytes per char |
|---|---|---|---|---|---|---|---|
| 1,024 | 8 | 111 | 262,144 | 16,000 | 32,000 (5,260) | 16.38 | 13.92 |
| 1,024 | 4 | 111 | 131,072 | 8,000 | 16,000 (82) | 16.38 | 11.41 |
| 1,024 | 1 | 111 | 32,768 | 0 | 4,000 (5) | - | - |
| 2,048 | 8 | 110 | 524,288 | 32,000 | 64,000 (16,698) | 16.38 | 13.75 |
| 2,048 | 4 | 110 | 262,144 | 16,000 | 32,000 (413) | 16.38 | 11.11 |
| 2,048 | 1 | 110 | 65,536 | 4,000 | 8,000 (15) | 16.38 | 16.39 |
| 4,096 | 8 | 108 | 1,048,576 | 80,000 | 96,000 (6,489) | 13.11 | 11.19 |
| 4,096 | 4 | 108 | 524,288 | 32,000 | 40,000 (17) | 16.38 | 11.25 |
| 4,096 | 1 | 108 | 131,072 | 8,000 | 10,000 (2) | 16.38 | 16.39 |
| 8,192 | 8 | 106 | 2,097,152 | 147,247 | 200,000 (22,604) | 14.24 | 11.96 |
| 8,192 | 4 | 106 | 1,048,576 | 32,000 | 64,000 (3) | 32.77 | 21.62 |
| 8,192 | 1 | 106 | 262,144 | 16,000 | 32,000 (72) | 16.38 | 16.39 |
| 16,384 | 8 | 104 | 4,194,304 | 147,247 | not reached (the text ended) | 28.48 | 23.79 |
| 16,384 | 4 | 104 | 2,097,152 | 64,000 | 147,247 (70) | 32.77 | 20.72 |
| 16,384 | 1 | 104 | 524,288 | 16,000 | 32,000 (1) | 32.77 | 32.78 |
| 32,768 | 8 | 103 | 8,388,608 | 147,247 | not reached (the text ended) | 56.97 | 45.31 |
| 32,768 | 4 | 103 | 4,194,304 | 147,247 | not reached (the text ended) | 28.48 | 18.17 |
| 32,768 | 1 | 103 | 1,048,576 | 32,000 | 64,000 (3) | 32.77 | 32.78 |
| 65,536 | 8 | 101 | 16,777,216 | 147,247 | not reached (the text ended) | 113.94 | 72.97 |
| 65,536 | 4 | 101 | 8,388,608 | 147,247 | not reached (the text ended) | 56.97 | 36.44 |
| 65,536 | 1 | 101 | 2,097,152 | 64,000 | 147,247 (15) | 32.77 | 32.78 |
| 131,072 | 8 | 100 | 33,554,432 | 16,000 | not reached (the text ended) | 2097.15 | 723.5 |
| 131,072 | 4 | 100 | 16,777,216 | 16,000 | not reached (the text ended) | 1048.58 | 633.82 |
| 131,072 | 1 | 100 | 4,194,304 | 16,000 | not reached (the text ended) | 262.14 | 253.96 |

Rows reading "not reached" ran out of text before the memory failed. The refine cells bracket three edges:

- 8-bit counters, 4,096 locations (1,048,576 bytes): word-perfect at 80,000 characters, 6,489 wrong positions at
  96,000. That is 13.1 bytes of memory a character (11.2 gzipped).
- 4-bit counters, 4,096 locations (524,288 bytes): word-perfect at 32,000, 17 wrong at 40,000: 16.4 bytes a
  character (11.3 gzipped).
- 1-bit counters, 4,096 locations (131,072 bytes): word-perfect at 8,000, 2 wrong at 10,000: 16.4 bytes a character.

The rule of thumb: a word-perfect memory costs 13 to 17 bytes a character, about 11 gzipped. The text itself is one
byte a character. A memory is therefore 13 to 17 times the size of what it recites; it recites from a cue and stores
no index. Past its edge it collapses rather than degrading slowly (8-bit, 4,096 locations: 0 wrong at 80,000, 6,489
at 96,000, 50,320 at 128,000).

## 3. Recite: what ships

Budget: 8 MB of counters a memory. Reasons: a phone downloads it in seconds, the page holds the counters as 16-bit
integers in memory (16 MB for the largest), and each character read scans every location (3,096 to 3,548 characters
a second measured in a tab for Beowulf). Each memory is sized by `tools/build_epic_memories.mjs --auto`: start at N/16
locations, grow by a quarter until the memory, loaded back from its own file, recites the whole text word for word
from its cue, or stop at the budget. Built on the Spark; logs in `shiplogs/`.

| memory | context | characters | memory bytes | locations | bytes a character |
|---|---|---|---|---|---|
| The Odyssey, book-1 | 96 | 21,445 | 524,288 | 2,048 | 24.4 |
| The Divine Comedy, inferno-cantos-1-5 | 96 | 32,010 | 524,288 | 2,048 | 16.4 |
| The Iliad, book-1 | 384 | 33,797 | 786,432 | 3,072 | 23.3 |
| The Aeneid, book-1 | 96 | 54,385 | 1,048,576 | 4,096 | 19.3 |
| Fragments of Ancient Poetry | 96 | 68,169 | 1,310,720 | 5,120 | 19.2 |
| Kalevala, the Epic Poem of Finland, runes-1-5 | 384 | 72,477 | 1,310,720 | 5,120 | 18.1 |
| Paradise Regained | 48 | 90,704 | 1,572,864 | 6,144 | 17.3 |
| Beowulf | 48 | 147,247 | 2,359,296 | 9,216 | 16.0 |
| The Song of Hiawatha | 192 | 187,546 | 3,145,728 | 12,288 | 16.8 |
| The Song of Roland | 96 | 192,263 | 3,145,728 | 12,288 | 16.4 |
| An Old Babylonian Version of the Gilgamesh Epic | 192 | 226,153 | 3,670,016 | 14,336 | 16.2 |
| The Lay of the Cid | 96 | 230,733 | 3,932,160 | 15,360 | 17.0 |
| The Argonautica | 48 | 333,438 | 7,077,888 | 27,648 | 21.2 |
| Paradise Lost | 192 | 459,534 | 7,602,176 | 29,696 | 16.5 |

14 memories, 38,010,880 bytes in all. Every one recites its whole text word for word from its opening cue and reads every position right from the true context (each meta file's `verify`, measured on the file as loaded).

Not shipped: Idylls of the King (476,737 characters) needs a 192-character context and still made 25 mistakes at 30,720
locations (7.9 MB); the next size is over the budget. Not attempted whole, because at 13 to 17 bytes a character
they are over the budget: the Iliad and Odyssey (both translations), the Aeneid, the Divine Comedy, the Kalevala,
the Ramayana, the Mahabharata, the Nibelungenlied, the Shah Nameh, the Lusiad, Orlando Furioso, Jerusalem
Delivered, the Faerie Queene, the Metamorphoses, the Poetic Edda, the Mabinogion and Sigurd. Their openings ship
instead: Iliad Book I, Odyssey Book I, Aeneid Book I, Kalevala Runes I to V, Inferno Cantos I to V.

The page: #/poem, THE POEM LAB · THE EPICS (`sites/settle-site/src/epics/`). It downloads a memory, checks it against
its sha256, and recites from a chosen line or from the reader's own cue; it says when a cue's last characters are not
in the poem, and that a recital may wander after its first mistake. Files go to `public/data/epics/` (gitignored, like
the chat models); rebuild with `tools/build_epic_memories.mjs`.

## 4. Compose: the epics model against its control

Sealed before any job ran: `PREREG_EPICS_COMPOSE_2026-10-02.md`. All three runs start from SDMCHATS's SW (the SDM
read at width 512, 300M FineWeb-Edu tokens) with SW's recipe at lr 1e-3, seed 1, on the Spark. Held-out epic lines:
every 20th block of 64 lines of every work, capped at 150,000 characters an epic (313,933 scored tokens); training
text capped at 3,000,000 characters an epic (6,023,067 tokens), so EP12 saw it about twice.

| run | continued on | held-out epic lines (bpb) | FineWeb-Edu TEST (bpb) |
|---|---|---|---|
| BASE | nothing | 2.5580 | 1.4299 |
| EP12 | 12M tokens of the epics | **1.5488** | 2.0712 |
| FW12 | 12M tokens of FineWeb-Edu (the control) | 2.5339 | 1.4278 |

Sources: `compose/epics_{BASE,EP12,FW12}.result.json`.

The sealed predictions, scored as sealed:

1. **missed**: BASE on the epic lines 1.75 to 1.95; measured 2.5580. The old texts are much further from web text than
   predicted.
2. **missed**: EP12 lowers the epic bpb by 0.15 to 0.35; measured 1.0092, about three times the top of the band.
3. **held**: FW12 moves the epic bpb by less than 0.03; measured -0.0241.
4. **held, the claim**: EP12 below FW12 on the epic lines by at least 0.10; measured 0.9851.
5. **missed**: EP12's FineWeb TEST rises 0.02 to 0.08; measured +0.6413. FW12's half held (-0.0022). Twelve million
   tokens of old verse at lr 1e-3 cost the model most of what it knew about web text.

A gain this large could be memory, so the overlap was measured (`track4_epics_overlap_check.py`,
`compose/overlap_check.json`): 1.02% of the held-out text's 64-character shingles appear verbatim in the training
text, highest in Pope's Odyssey (7.6%) and Iliad (5.3%). That is an upper bound (a 32-bit hash, so some hits are
collisions). The gain comes from the register, the names and the line structure, not from repeated lines.

Samples (`compose/epics_compose_samples.json`, `track4_epics_compose_samples.py`), sampled at temperature 0.8, top
50, after a cue from a held-out Paradise Lost block: BASE writes about the American Journal of American Association;
FW12 writes about a trail and a destination; EP12 writes "In vain, to some, that would find his home, / But to the
Phrygian's coast, and Phaëron's deep / Gives, and in the same grove of Phœnician rocks". It has the register and the
line breaks and no sense. One seed, one run each.

## 5. What is left

- The compose model is not in the browser: it is a checkpoint on the Spark
  (`~/settle24/ck/main/epics_EP12/last.pt`). Exporting it needs a new key in SDMCHATS's export and model list
  (`tools/export_sdm_chat.py`, `src/sdmchat/useSdmEngine.js`), which that lane owns.
- Only one seed per run, and no paired per-window comparison on the epic test set.
- The forgetting (+0.64 bpb on FineWeb) is untreated: a mixed continuation (epics plus web text) is the obvious next
  arm.
- 21 epics are too large for one 8 MB memory; per-book memories for all of them would be about 150 files.
- The recite memories use their own small record (`describeEpic`, family `recite`); LITTLEGUYSTUDIO's standard
  descriptor (`src/studio/descriptor.js`) covers chat models only and had not landed. When it does, these move onto it.
