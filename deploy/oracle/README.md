# Deploying SHRUTI on Oracle Cloud Always Free

Free forever, no ongoing charge: **4 ARM cores / 24 GB RAM**. The app uses 555 MB, so this is
roughly 40× the headroom it needs — the constraint stops being memory and becomes your patience
with Oracle's capacity queue.

Your part is the account and the VM. Everything after that is one script.

---

## Before you start — two things that catch everyone

**1. "Out of host capacity" is normal, not a bug.** Always Free ARM (Ampere A1) is heavily
oversubscribed. Mumbai and Hyderabad are frequently full. You may retry for hours. Two mitigations:
pick a less busy home region at signup, and if it fails, wait and retry rather than assuming you
did something wrong. **Your home region cannot be changed later**, so choose deliberately.

**2. The microphone needs real HTTPS.** `getUserMedia` refuses to run over `http://` or behind a
self-signed certificate — the Speak button will simply do nothing. That is why this uses a free
DuckDNS subdomain plus Let's Encrypt rather than the bare IP.

---

## Step 1 — Oracle account (~15 min)

1. Sign up at **https://signup.cloud.oracle.com**
2. Card required for **identity verification only**. Always Free resources are never charged, and
   the account stays free unless you deliberately upgrade to Pay As You Go.
3. **Choose your home region carefully** — it is permanent. If Mumbai/Hyderabad fail on capacity,
   Singapore, Osaka, Amsterdam and Phoenix are usually easier. A non-Indian region adds latency
   from your laptop, which is fine: the SLO is server-measured and the benchmark already runs from
   a US GitHub runner.

## Step 2 — Create the VM (~10 min, plus retries)

Console → **Compute → Instances → Create Instance**

| field | value |
|---|---|
| Image | **Ubuntu 24.04** (or 22.04) |
| Shape | **VM.Standard.A1.Flex** ← must be Ampere ARM, not the x86 micro |
| OCPUs | **4** |
| Memory | **24 GB** |
| Boot volume | 50 GB is plenty |
| SSH keys | upload your public key, or let Oracle generate one and **download it** |

Note the **public IP** when it finishes.

> Hitting `Out of host capacity`? Try 1 OCPU / 6 GB — smaller shapes are often available, and the
> app only needs 555 MB. Or retry the 4-core shape periodically; availability fluctuates hourly.

## Step 3 — Open the firewall in the console (~2 min)

**This is the step everyone forgets**, and it makes a perfectly working VM look dead.

Console → **Networking → Virtual Cloud Networks → your VCN → Security Lists → Default**
→ **Add Ingress Rules**, twice:

| Source CIDR | Protocol | Destination Port |
|---|---|---|
| `0.0.0.0/0` | TCP | `80` |
| `0.0.0.0/0` | TCP | `443` |

The setup script opens the VM's own iptables, but Oracle blocks at the cloud layer too. **Both are
required.**

## Step 4 — Free domain (~2 min)

1. Go to **https://www.duckdns.org**, sign in with GitHub
2. Create a subdomain, e.g. `shruti-het`
3. Put your VM's public IP in the box and press **update ip**

You now have `shruti-het.duckdns.org`. Confirm it resolves before continuing:

```bash
dig +short shruti-het.duckdns.org     # must print your VM IP
```

## Step 5 — Deploy (~10 min, mostly unattended)

SSH in:

```bash
ssh -i /path/to/your_key ubuntu@<VM_PUBLIC_IP>
```

Create the secrets file:

```bash
sudo mkdir -p /opt/shruti && sudo chown $USER:$USER /opt/shruti
cat > /opt/shruti/.env <<'EOF'
SARVAM_API_KEY=your_key
CEREBRAS_API_KEY=your_key
GROQ_API_KEY=your_key
SHRUTI_EMBED_LANE=fast
SHRUTI_SEARCH_MODE=hnsw
SHRUTI_WARMUP_QUERIES=20
EOF
chmod 600 /opt/shruti/.env
```

Run it:

```bash
curl -fsSL https://raw.githubusercontent.com/Het161/shruti/main/deploy/oracle/setup.sh \
  | bash -s -- --domain shruti-het.duckdns.org --email you@example.com
```

Then verify:

```bash
curl https://shruti-het.duckdns.org/api/health
```

Expect `"ready":true` and `"corpus_passages":310582`.

---

## What changes versus the Modal deployment

| | Modal (x86) | Oracle (ARM) |
|---|---|---|
| cost | $30/mo free credit | **free forever** |
| cold start | ~15 s (scales to zero) | **none — always on** |
| CPU | 2 vCPU | 4 ARM cores |
| RAM | 2 GB | 24 GB |
| int8 ONNX | AVX512-VNNI available | **not on ARM — falls back to fp32** |

That last row is the one real regression: the e5 and reranker int8 exports are compiled for
AVX512-VNNI and will not run on aarch64. `app/stages/embed.py` and `rerank.py` already fall back to
fp32 automatically, so nothing breaks — the reranker just runs slower than the 561 ms measured on
x86. It is off by default and lives in Tier 2, so the headline SLO is unaffected.

**Re-run the benchmark after deploying** and publish the ARM numbers. Do not reuse the Modal
figures — different silicon, different numbers:

```bash
gh variable set SHRUTI_URL --body "https://shruti-het.duckdns.org"
gh workflow run benchmark
python bench/testset.py --url https://shruti-het.duckdns.org
```

Always-on is a genuine upgrade for a judged demo: no cold start means nobody waits 15 seconds, and
there is no warm-container cost to reason about.

---

## Operating it

```bash
sudo docker logs -f shruti                    # logs
sudo docker restart shruti                    # restart
cd /opt/shruti && git pull \
  && sudo docker build -t shruti:latest . \
  && sudo docker restart shruti               # deploy an update
sudo certbot renew --dry-run                  # TLS auto-renews; this proves it
```

## If it doesn't work

| symptom | cause |
|---|---|
| connection times out | VCN ingress rules missing (Step 3) — the usual culprit |
| certbot fails | DNS not propagated; check `dig +short <domain>` returns the VM IP |
| Speak button dead | not on `https://`, or the browser denied mic permission |
| voice connects then drops | nginx missing the WebSocket `Upgrade` headers — the script sets these |
| container exits at boot | `sudo docker logs shruti`; usually a missing `/opt/shruti/.env` |
