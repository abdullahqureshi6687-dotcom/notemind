# Deploying NoteMind

Two options:

| | **A. Streamlit Community Cloud** (recommended) | **B. Hugging Face Spaces (Docker)** |
|---|---|---|
| Cost | Free | New compute Spaces need a paid plan (see below) |
| Deploys from | your GitHub repo | a Space git repo, using the `Dockerfile` |
| Resources | up to ~2.7 GB RAM, 2 CPU cores | 2 vCPU, 16 GB RAM (CPU Basic) |
| Sleeps after | 12 hours without traffic | 48 hours without traffic |
| Secrets | TOML in the app's settings | Settings → Variables and secrets |

> **What changed (checked October 2026).** Hugging Face has deprecated Streamlit as a built-in
> Space SDK; Streamlit apps now use the **Docker** SDK with a Streamlit template. Its hardware
> page also states that *"creating a new Space that runs on compute (Gradio or Docker) requires
> a paid plan. Static Spaces are free for everyone."* Free accounts can therefore no longer
> create a new Space that runs NoteMind, which is why Streamlit Community Cloud is the
> recommended free option. Re-check both pages before deploying, because these policies change:
> [Streamlit Spaces](https://huggingface.co/docs/hub/spaces-sdks-streamlit),
> [Spaces hardware](https://huggingface.co/docs/hub/main/spaces-gpus).

---

## Before you deploy (both options)

1. **Choose the LLM for the public demo.** A public app shares your API key's quota with every
   visitor. Gemini's free tier allows only about 20 requests per day, so prefer a provider with
   a larger free allowance (OpenRouter `:free` models, Groq or Mistral). Visitors can also paste
   their own key under **⚙️ Settings** in the app; it stays in their session only.
2. **Use a separate API key for the deployment** (for example `notemind-demo`), so you can
   revoke it without touching your local key. Set a low spending limit if the provider offers one.
3. **Make sure no secrets are committed:** `git ls-files | Select-String "\.env$"` must print nothing.
4. **Push everything to GitHub** and check that the tests are green in the Actions tab.

The provider and model can be chosen without editing `config.yaml`, through two
environment variables that NoteMind reads at startup:

| Variable | Example | Effect |
|---|---|---|
| `NOTEMIND_PROVIDER` | `openrouter` | replaces `generation.provider` |
| `NOTEMIND_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | replaces that provider's model ID |

This is useful because free model IDs are retired often: you can switch models from the
hosting dashboard without a new commit.

---

## A. Streamlit Community Cloud (free)

### Files it uses

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | entrypoint |
| `app/requirements.txt` | dependencies for the cloud. Community Cloud reads the dependency file next to the entrypoint before the root one. It adds the **CPU-only PyTorch** index, so the server doesn't download ~3 GB of CUDA libraries. A test keeps its pins identical to `requirements.txt`. |
| `.streamlit/config.toml` | upload size limit and theme |
| `assets/demo/*.pdf` | the **Try demo notes** button |

The app adds `src/` to Python's import path itself, so no `pip install -e .` step is needed.

### Steps

1. Go to **https://share.streamlit.io** and sign in with GitHub. Authorise access to your repositories.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `abdullahqureshi6687-dotcom/notemind`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** choose something short, e.g. `notemind-yourname`
4. Open **Advanced settings**:
   - **Python version:** **3.12** (the version everything was tested with).
   - **Secrets:** paste TOML like this, with your real key and only the providers you use:

     ```toml
     OPENROUTER_API_KEY = "sk-or-..."
     NOTEMIND_PROVIDER = "openrouter"
     NOTEMIND_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
     ```

     Keep these **at the top level** (no `[section]` header): Streamlit exposes top-level secrets
     as environment variables, and the app also copies them into the environment at startup.
5. Click **Deploy**. The first build installs PyTorch and the other packages and takes several
   minutes. If the log shows `uv` failing and `pip` retrying, that's expected: Community Cloud
   falls back to pip automatically.
6. When the app opens, click **Try demo notes** and ask a question. The very first question also
   downloads the embedding model (~90 MB), so allow up to a minute.

To change secrets later: app menu (⋮) → **Settings** → **Secrets**. Saving restarts the app.

### Avoiding slow starts

- The embedding model and config are cached with `st.cache_resource`, so they load **once per
  server**, not once per visitor.
- Indexes are cached on disk by content hash, so **Try demo notes** is instant after the first visitor.
- Apps sleep after 12 hours without traffic; the next visitor wakes it, which takes a short while.
  Open the app yourself before sharing the link (for example before an interview).

---

## B. Hugging Face Spaces (Docker)

Requires an account that is allowed to create compute Spaces (currently a paid plan for new
Spaces; see the note at the top).

### Files it uses

- `Dockerfile`: Python 3.12 slim image, runs as user ID 1000 (Spaces requirement), installs
  CPU-only PyTorch first, **bakes the embedding model into the image** so cold starts don't
  download it, and serves Streamlit on port **7860** (the Docker Spaces default).
- `.dockerignore`: keeps `.env`, `venv/`, indexes, your own PDFs, tests and notebooks out of the image.

The image disables Streamlit's XSRF check (`STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false`),
because its cookie is blocked inside the Hugging Face iframe, which breaks file uploads there.
Remove that line when deploying the image anywhere else.

### Steps

1. Create a Space at **https://huggingface.co/new-space**: choose **Docker** as the SDK, a
   **Blank** template, **CPU Basic** hardware, and Public visibility.
2. In the Space: **Settings → Variables and secrets**.
   - **New secret:** `OPENROUTER_API_KEY` (or your provider's key variable).
   - **New variable:** `NOTEMIND_PROVIDER` = `openrouter`, and optionally `NOTEMIND_MODEL`.

   Secrets and variables are injected as environment variables when the container runs.
3. The Space's `README.md` must start with this YAML block. Add it to the **top** of your
   README before pushing (GitHub shows it as a small table, which is harmless):

   ```yaml
   ---
   title: NoteMind
   emoji: 📚
   colorFrom: indigo
   colorTo: blue
   sdk: docker
   app_port: 7860
   pinned: false
   short_description: Chat with your lecture notes, with page citations
   ---
   ```

4. Push the repository to the Space. Use a Hugging Face **access token with write permission**
   (Settings → Access Tokens) as the password when Git asks:

   ```powershell
   git remote add space https://huggingface.co/spaces/YOUR-HF-USERNAME/notemind
   git push space main
   ```

   If the Space already has a commit (its generated README), run `git push space main --force`
   once, after checking that you want to replace it.
5. Watch the **Logs** tab. The first build takes several minutes (PyTorch and the embedding
   model); later builds that only change code reuse the cached dependency layers.

Data written inside the container (for example cached indexes) is lost when the Space restarts.
That's fine for NoteMind, which rebuilds indexes on demand.

---

## Running the Docker image locally (any host)

```powershell
docker build -t notemind .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=sk-or-... -e NOTEMIND_PROVIDER=openrouter notemind
```

Then open http://localhost:7860. The Dockerfile follows Hugging Face's documented Docker
setup, but the complete image hasn't been built end to end yet (the environment it was written
in couldn't reach Docker Hub), so treat your first build as the real test and check the logs.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "No API key for …" warning | The secret name doesn't match the provider (see the table in the README), or it is inside a `[section]` in the TOML. |
| "free-tier quota is used up" | The shared key hit its daily limit. Switch `NOTEMIND_MODEL`/`NOTEMIND_PROVIDER`, or paste your own key in ⚙️ Settings. |
| `LLMBadRequestError` with 404 | The free model ID was retired. Pick a current `:free` ID at openrouter.ai/models and update `NOTEMIND_MODEL`. |
| App runs out of memory | Very large uploads. Keep PDFs under the 50 MB upload limit and avoid uploading dozens of files at once. |
| Uploads fail with 403 on Hugging Face | Make sure the XSRF line in the Dockerfile is present. |
| Build fails installing torch | Check that `app/requirements.txt` (Streamlit Cloud) or the Dockerfile still points at the CPU index and that `torch==2.14.1` is available for Python 3.12. |
