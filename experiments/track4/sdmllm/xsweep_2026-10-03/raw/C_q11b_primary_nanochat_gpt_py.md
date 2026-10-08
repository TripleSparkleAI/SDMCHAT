# C_q11b primary check: nanochat/gpt.py at master

- url: https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/gpt.py
- tool: python3 urllib
- fetched (UTC, status): 2026-10-03T10:51:19.533099Z 200
- last commits touching the file (GitHub API):
  - 92d63d4e 2026-07-03T22:54:57Z clean up fragile code
  - a9d0a862 2026-07-03T19:57:47Z unify the two optimizer implementations into one and prevent bugs like the one that just happened where i updated one an
  - 2fc63c96 2026-07-03T17:38:16Z Merge pull request #540 from svlandeg/fix/kernel
  - 3616a029 2026-07-03T17:34:12Z add a comment clarifying that RoPE here rotates by -theta (transpose of the textbook convention), which is functionally 
  - a3ca42a6 2026-04-13T12:17:23Z add comment
  - 9822cc74 2026-04-13T12:03:18Z use nn.init and initialize smear gate's weight as well
  - 94b73ad2 2026-04-03T20:39:55Z fix: initialize smear and backout lambdas in init_weights
  - a825e63f 2026-03-14T17:03:06Z Autoresearch round 2: smear, backout, and hyperparameter tuning

## Verbatim grep lines (line number: text)
```
7:- untied weights for token embedding and lm_head
32:    vocab_size: int = 32768
175:            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
178:        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
209:        wte (embedding):     normal, std=1.0
210:        lm_head:             normal, std=0.001
212:            attn.c_q:        uniform, std=1/sqrt(n_embd)
213:            attn.c_k:        uniform, std=1/sqrt(n_embd)
214:            attn.c_v:        uniform, std=1/sqrt(n_embd)
216:            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
220:        # Embedding and unembedding
221:        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
222:        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
267:            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
275:            device = self.transformer.wte.weight.device
318:        return self.transformer.wte.weight.device
404:        wte = sum(p.numel() for p in self.transformer.wte.parameters())
406:        lm_head = sum(p.numel() for p in self.lm_head.parameters())
409:        total = wte + value_embeds + lm_head + transformer_matrices + scalars
412:            'wte': wte,
414:            'lm_head': lm_head,
420:    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
426:        embedding_params = list(self.transformer.wte.parameters())
427:        lm_head_params = list(self.lm_head.parameters())
431:        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)
439:            # AdamW groups (embeddings, lm_head, scalars)
440:            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
441:            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
442:            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
472:        x = self.transformer.wte(idx) # embed current token
511:        # Forward the lm_head (compute logits)
512:        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
513:        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
515:        logits = logits.float() # switch to fp32 for logit softcap and loss computation
516:        logits = softcap * torch.tanh(logits / softcap) # squash the logits
```
