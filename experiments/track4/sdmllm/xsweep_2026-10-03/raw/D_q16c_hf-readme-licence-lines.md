# D_q16c dataset README licence and composition lines (primary, verbatim grep)
- tool: python3 urllib, https://huggingface.co/datasets/<id>/raw/main/README.md
- fetched: 2026-10-03 10:52:26 UTC

## HuggingFaceTB/smoltalk2
- url: https://huggingface.co/datasets/HuggingFaceTB/smoltalk2/raw/main/README.md
- bytes: 23465
  > - LongAlign-64k-context-lang-annotated (lang_6): 6249 examples. We filter [LongAlign](https://huggingface.co/datasets/THUDM/LongAlign-10k) for samples up to 64k tokens.
  > | Dataset                                           |   Weight |   # examples |   % of examples |   # tokens (M) |   % of tokens |   Avg. # turns |   Avg. # tokens per example |   Avg. # tokens in context |   Avg. # tokens in response |
  > | Dataset                                     |   Weight |   # examples |   % of examples |   # tokens (M) |   % of tokens |   Avg. # turns |   Avg. # tokens per example |   Avg. # tokens in context |   Avg. # tokens in response |
  > | Dataset                                                        |   Weight |   # examples |   % of examples |   Avg. # turns |   Avg. # tokens in context |   # tokens (M) (Chosen) |   % of tokens (Chosen) |   Avg. # tokens per example (Chosen) |   Avg. # tokens in response (Chosen) |
  > ## License
  > All the new datasets (aya_dataset-Qwen3-32B, multi-turn-reasoning-if, smolagents-toolcalling-traces, smoltalk-everyday-convs-reasoning-Qwen3-32B, smoltalk-multilingual8-Qwen3-32B, smoltalk-systemchats-Qwen3-32B, table-gpt-Qwen3-32B, tulu_3_8b_pref_mix_qwen3_32b_qwen3_06b_think) are licensed under Ap

## HuggingFaceTB/smoltalk
- url: https://huggingface.co/datasets/HuggingFaceTB/smoltalk/raw/main/README.md
- bytes: 9715
  > - LongAlign: we find that finetuning the model on only short samples makes it loose long context abilities beyond 2048 tokens, so we add english samples (with less than 16k tokens) from the [LongAlign-10k](https://huggingface.co/datasets/THUDM/LongAlign-10k) dataset and train with a 8192 sequence.
  > ## License
  > All the new datasets (Smol-Magpie-Ultra, Smol-contraints, Smol-rewrite, Smol-summarize) are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). For the existing public datasets, please refer to the original dataset for the license [Dataset composition](#dataset-composition)
  > We compare SmolTalk to the recent [Orca AgentInstruct 1M](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) dataset by finetuning SmolLM2 on both datasets using the same training setup (we train for 2 epochs, using a learning rate of 3e-04, a sequence length of 8192 and a global ba

## PleIAs/common_corpus
- url: https://huggingface.co/datasets/PleIAs/common_corpus/raw/main/README.md
- bytes: 11170
  > Common Corpus is the largest open licensed text dataset, comprising 2.27 trillion tokens (2,267,302,720,836 tokens). It is a diverse dataset, consisting of books, newspapers, scientific articles, government and legal documents, code, and more. Common Corpus has been created by Pleias in association 
  > *  **Truly Open**: contains only data that is either uncopyrighted or freely licensed
  > *  **Traceable**: each individual document is associated with documented contextual information, including licensed use or lack of copyright.
  > *  **Multilingual**: mostly representing English and French data, but contains data for 8 languages with more than 10 billion tokens (German, Spanish, Italian, Polish, Greek, Latin) and 33 languages with more than 1 billion tokens.
  > The dataset in its entirety meets the requirements of the Code of Conduct of the AI Act and goes further than the current requirements for data transparency. It aims to set a new standard of openness in AI, showing that detailed provenance at a granular document level is a realistic objective, even 
  > Common Corpus makes it possible to train model compatible with [the Open Source Initiative’s definition](https://opensource.org/ai/open-source-ai-definition#:~:text=An%20Open%20Source%20AI%20is,including%20to%20change%20its%20output.) of open-source AI, which includes openness of use, meaning use is
  > *  **OpenCulture**: our largest collection at 967,018,390,906 tokens, featuring public domain books, newspapers from cultural heritage repositories and open projets like Wikisource ad Gutenberg. We're developing innovative tools of OCR correction based on Pleias Models to correct historical digitiza
  > *  **OpenGovernment**: 579,150,518,908 tokens of financial and legal documents, including Finance Commons (from sources like SEC and WTO) and Legal Commons (including Europarl, Caselaw Access Project, Chinese Case Law), providing enterprise-grade training data from regulatory bodies and administrati
  > *  **OpenSource**: 283,227,402,898 tokens of high-quality code in open source from GitHub, filtered using ArmoRM to ensure only the top 80% of submissions by quality rating are included.
  > *  **OpenScience**: 281,193,563,789 tokens of academic content from Open Alex and other open science reposiories, processed using vision-language models to preserve crucial document structure and formatting.
  > *  **OpenWeb**: 88,517,032,065 tokens from Wikipedia (official releases from the [Wikimedia Foundation](https://huggingface.co/datasets/wikimedia/wikipedia) on Huggingface), YouTube Commons and Stack-Exchange.
  > *  **Open Semantic**: 67,958,671,827 tokens from Wikidata (official releases from the [Wikimedia Foundation](https://huggingface.co/datasets/wikimedia/wikipedia) on Huggingface). The data has been reprocessed thanks to support and help of Wikidata and Wikimedia Germany. It includes the transcription
  > | OpenCulture	| cultural heritage    	| public domain books and newspapers, Wikisource                                                    	|
  > *  `license`: sharing rights for the content either uncopyrighted (public domain, US federal public domain, CC0 on Wikidata) or various free licenses (Creative Commons, MIT, French Licence ouverte, etc.)
  > *  `date`: date of creation of the resource where known. Due to the significance of public domain and other cultural heritage content, more than half of Common Corpus predates the 21st century.
  > *  `token_count`: number of tokens as calculated by Pleias official tokenizer and Gemma-3 tokenizer for Chinese, Japanese, Arabic, Korean and few additional non-Western languages.
  > All data in Common Corpus are either uncopyrighted or freely licensed and may be used for both commercial and non-commercial purposes.

## institutional/institutional-books-1.0
- url: https://huggingface.co/datasets/institutional/institutional-books-1.0/raw/main/README.md
ERROR: HTTPError: HTTP Error 401: Unauthorized

## PleIAs/SYNTH
- url: https://huggingface.co/datasets/PleIAs/SYNTH/raw/main/README.md
- bytes: 11933
  > license: cc-by-4.0
  > SYNTH includes 79,648,272 individual text samples, comprising over 41 billion words (about 75 billion tokens with Pleias tokenizer). It is based on the amplification of 58,698 articles from Wikipedia and made possible thanks to the *Structured Wikipedia* dataset from Wikimedia Enterprise.
  > * **fully open** based on seed text under open license (CC-By-SA) and generated with models allowing for output reuse. This means that SYNTH can be universally release and serve as a basis for further reproducible synthetic pipelines.
  > * **data efficient** with best results attained with only 100-200 billions tokens trained on SYNTH.
  > - **License:**
  > In contrast with organic pretraining dataset, SYNTH allows for fast convergence to the existing SOTA (about 100 billion tokens). Furthermore, SYNTH is fully releasable, only use sourced text under free license.
  > | **seed_license**        | `string` | License of the seed text (most of the time `"CC-BY-SA 4.0"`).                                                       |

## HuggingFaceTB/smol-smoltalk
- url: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/raw/main/README.md
- bytes: 2231
  > license: apache-2.0

## common-pile/comma_v0.1_training_dataset
- url: https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset/raw/main/README.md
- bytes: 5454
  > | Main stage                    | Tokens (B) | Repeats | Effective tokens (B) |
  > | Cooldown stage               | Tokens (B) | Repeats | Effective tokens (B) |
