# D_q29 primary: FinePhrase card and arXiv 2604.13977
- urls: https://huggingface.co/datasets/HuggingFaceFW/finephrase/raw/main/README.md , http://export.arxiv.org/api/query?id_list=2604.13977&max_results=2
- tool: python3 urllib
- fetched: 2026-10-03 10:56:38 UTC

## arXiv http://arxiv.org/abs/2604.13977v2 published 2026-04-15
title: How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data
authors: Joel Niklaus, Atsuki Yamaguchi, Michal Štefánik, Guilherme Penedo, Hynek Kydlíček, Elie Bakouch, Lewis Tunstall, Edward Emanuel Beeching, Thibaud Frere, Colin Raffel, Leandro von Werra, Thomas Wolf
abstract (verbatim): Synthetic data is a standard component in training large language models, yet systematic comparisons across design dimensions, including rephrasing strategy, generator model, and source data, remain absent. We conduct extensive controlled experiments, generating over one trillion tokens, to identify critical factors in rephrasing web text into synthetic pretraining data. Our results reveal that structured output formats, such as tables, math problems, FAQs, and tutorials, consistently outperform both curated web baselines and prior synthetic methods. Notably, increasing the size of the generator model beyond 1B parameters provides no additional benefit. Our analysis also demonstrates that the selection of the original data used for mixing substantially influences performance. By applying our findings, we develop \textbf{\textsc{FinePhrase}}, a 486-billion-token open dataset of rephrased web text. We show that \textsc{FinePhrase} outperforms all existing synthetic data baselines while reducing generation costs by up to 30 times. We provide the dataset, all prompts, and the generation framework to the research community.

## FinePhrase card (verbatim lines)
> license: odc-by
> - HuggingFaceFW/fineweb-edu/sample-350BT
> - Model: [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) (`main`)
> - Source dataset: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), config `sample-350BT`, split `train`
> - Source documents in input split: `339,347,842` (≈339.3M)
> - Output samples across all configs: `1,354,044,711` (≈1.35B)
> - Completion tokens across all configs: `486,367,076,933` (≈486.4B)
> Blog post: [FinePhrase](https://huggingface.co/spaces/huggingface/finephrase)

HF API: createdAt 2026-02-15, lastModified 2026-03-31, licence odc-by, downloads 129328 (from D_q28).
## column check
- url: https://datasets-server.huggingface.co/info?dataset=HuggingFaceFW/finephrase&config=faq
- fetched: 2026-10-03 10:57:34 UTC
- ERROR: HTTPError: HTTP Error 500: Internal Server Error. The card YAML lists configs (all, faq, math, table, tutorial) but this read did not establish whether rows carry a source-document id.
