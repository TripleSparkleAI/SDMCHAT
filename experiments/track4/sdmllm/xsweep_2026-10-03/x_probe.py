import os, json, sys, urllib.request, urllib.parse, urllib.error, datetime
tok = os.environ["X_BEARER_TOKEN"]
q = sys.argv[1]; out = sys.argv[2]
params = {"query": q + " -is:retweet", "max_results": "50",
  "tweet.fields": "created_at,public_metrics,author_id,conversation_id,lang,note_tweet,referenced_tweets,entities",
  "expansions": "author_id", "user.fields": "username,name,public_metrics"}
url = "https://api.x.com/2/tweets/search/recent?" + urllib.parse.urlencode(params)
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok, "User-Agent": "curl/8.4.0"})
rec = {"query": q, "fetched_utc": datetime.datetime.utcnow().isoformat() + "Z"}
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        rec["status"] = r.status
        rec["rate"] = {k: v for k, v in r.headers.items() if k.lower().startswith("x-rate")}
        rec["body"] = json.loads(r.read())
except urllib.error.HTTPError as e:
    rec["status"] = e.code
    rec["rate"] = {k: v for k, v in e.headers.items() if k.lower().startswith("x-rate")}
    rec["body"] = e.read().decode("utf-8", "replace")
json.dump(rec, open(out, "w"), indent=1)
print(rec["status"], rec["rate"], str(rec["body"])[:300])
