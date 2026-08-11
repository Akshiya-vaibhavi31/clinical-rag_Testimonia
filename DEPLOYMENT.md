# Deployment Guide

## Local development vs. production

People sometimes treat these as the same thing with a different URL.
They're not. Running `uvicorn --reload` on a laptop and running the same
code on a public server involve different assumptions about who can
reach it, where secrets live, and what happens when something goes wrong.

On a laptop, only you can hit the API, so a wide-open CORS policy and a
plain `.env` file sitting in the project folder are harmless. On a real
server, that same `.env` file needs to become actual environment
variables set on the machine (never committed, never copied as a file
onto the server if it can be avoided), CORS needs to be locked to the
actual frontend domain instead of `*`, and error responses need to stay
generic instead of showing stack traces to whoever happens to send a bad
request.

Data persistence is the other big difference. Locally, the `data/`
folder just sits on disk and nothing touches it unless you do. In
production, if the hosting platform doesn't guarantee that folder
survives a restart or redeploy, the vector database and the SQLite
database both disappear the next time the app restarts — which, on some
platforms, happens automatically and often.

| | Local | Production |
|---|---|---|
| Reachable by | just you | anyone with the URL |
| Secrets | `.env` file | real environment variables on the host |
| Data | your local disk | needs a persistent volume that survives restarts |
| CORS | `*` is fine | should be the real frontend domain only |
| Restarts | you control them | should happen automatically, without losing data |

## What this app actually needs to run

Before picking a host, it's worth being specific about what this
particular app requires, rather than picking whatever's popular.

The embedding model and the cross-encoder reranker both run on CPU
through `torch`. That's real memory pressure — realistically somewhere
in the 1.5-3GB range, not the 512MB a lot of free web-service tiers hand
out. The vector database (Chroma) and the SQLite database both write to
local disk and need that disk to still be there after a restart — this
rules out any host whose free tier gives you a throwaway filesystem.
Gemini itself doesn't need hosting since it's called over the network,
but the same free-tier rate limits this project ran into directly during
development (the retry logic in `rag_pipeline.py` exists because of a
real 429 error, not a hypothetical one) apply just as much in production.

## Checking what's actually free right now

Hosting free tiers change often enough that citing a number from memory
is a bad idea — a few of the most commonly recommended options have
changed meaningfully in the last two years, and I checked current status
before writing anything down here rather than relying on older
knowledge.

Fly.io is the clearest case: it's still recommended constantly in
tutorials, but the free allowance it used to offer is gone. New accounts
get a short trial (a couple of hours, or a week, whichever ends first),
and after that everything is billed. It's not a free option anymore,
full stop.

Render still has a genuine free web service tier, but two of its limits
work against this project specifically. There's no persistent disk on
the free plan, so anything written to local disk vanishes on restart —
that breaks the vector database and the SQLite log outright. The RAM
ceiling (512MB) is also tight for a `torch`-based embedding pipeline. It
could work as a stripped-down demo if you're willing to rebuild the
vector index every time the service spins back up and accept that query
history won't persist, but it's not a good fit as the main deployment
target.

Oracle Cloud's "Always Free" tier turned out to be the better match, with
one important caveat: Oracle actually cut this tier back in June 2026.
Older guides (and a lot of them are still floating around) quote the
previous numbers — 4 CPUs, 24GB of RAM. What's actually available now is
2 ARM CPUs and 12GB of RAM, plus 200GB of persistent block storage and a
generous bandwidth allowance. That's still comfortably more than this
app needs, and — unlike Render's free tier — it's a real always-on VM
with a real disk, not a service that spins down and wipes local state.
It's also not a trial; there's no expiration date attached to it. Signing
up asks for a card for identity verification (a $1 hold, not a charge).

The tradeoff is setup effort. This isn't a git-push-to-deploy platform —
it's a VM you configure yourself, which means handling the OS, the
firewall, and running Docker on it by hand. More work upfront, but
nothing about the free tier is likely to quietly change out from under
you the way Fly.io's and Render's terms already have.

## Setting it up on Oracle Cloud

```bash
# Provision an Always Free Ampere A1 instance (Ubuntu) from the Oracle
# Cloud console first, then SSH in:
ssh ubuntu@<your-vm-public-ip>

# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
# log out and back in for the group membership to take effect

# Open port 8000 — Oracle's default firewall blocks it even after you
# open it in the console's security list, so both need doing
sudo iptables -I INPUT 6 -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save

# Get the code onto the machine
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO

# Real secrets go directly on the server, not committed anywhere
cp .env.example .env
nano .env

# Build the data pipeline once (or copy an already-built data/ folder up via scp)
docker compose run api python -m src.ingest_trials
docker compose run api python -m src.ingest_pubmed
docker compose run api python -m src.build_corpus
docker compose run api python -m src.chunk_corpus
docker compose run api python -m src.embed_corpus
docker compose run api python -m src.build_vector_db
docker compose run api python -m src.build_source_index

# Bring it up in the background, set to restart on crash
docker compose up -d
```

The app is now reachable at `http://<your-vm-public-ip>:8000/`.

## Two settings to change before calling it production

```bash
CORS_ORIGINS=https://your-actual-domain.com   # not "*"
LOG_LEVEL=WARNING                              # INFO gets noisy at real traffic volume
```

## What I couldn't verify myself

I don't have a cloud account or the ability to provision a real VM from
here, so none of the steps above were actually run end-to-end by me — I
can't personally confirm the exact firewall commands or Docker install
steps work flawlessly on Oracle's current Ubuntu image. The pricing and
resource-limit numbers were checked against live sources at the time of
writing rather than pulled from memory, specifically because this is an
area that keeps changing — but you should still double-check current
terms yourself at the actual moment you deploy.
