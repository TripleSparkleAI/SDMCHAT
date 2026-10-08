# B_q30_github-deepseek-engram-issues

- query url: https://api.github.com/repos/deepseek-ai/Engram/issues?state=all&sort=updated&per_page=40
- tool: python3 urllib, GitHub REST repos/deepseek-ai/Engram/issues (state=all, sorted updated)
- fetched_utc: 2026-10-03T10:52:34Z

ERROR: HTTPError: HTTP Error 403: rate limit exceeded

## Fallback after the 403 (unauthenticated GitHub core rate limit)
- tool: WebFetch https://github.com/deepseek-ai/Engram/issues
- fetched_utc: 2026-10-03T10:53Z (approx; same minute as the WebFetch call)
- issues visible (number, title, state, date as rendered):
  - #26 Related work: post-hoc fact writes into an Engram-style table | Open | Sep 5, 2026
  - #25 关于n-gram与适配v4pro而诞生的极简模式破解冻结新手问题的思考 | Open | Aug 16, 2026
  - #24 问题讨论：怎么看待Engram | Open | Apr 25, 2026
  - #22 User requests related to memory / persistence (from DeepSeek-V3 issues) | Open | Mar 18, 2026
  - #21 关于 engram_vocab_size 中词表系数的设置 | Open | Mar 6, 2026
  - #20 engram训练loss更低但是评测差的问题 | Open | Mar 5, 2026
  - #17, #16 training code (Feb 13 2026), #11, #10, #9 Why use 5x learning rate and zero weight decay for Engram parameters? (Jan 16 2026), #8, #6, #4, #3, #2
- In-window issues: 2 (#25, #26). Primary follow-ups opened: #9, #20 (both outside window, quoted in band file).
