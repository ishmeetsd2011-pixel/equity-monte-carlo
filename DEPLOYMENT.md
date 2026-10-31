# GitHub and Streamlit Community Cloud deployment

The application entrypoint is **app/main.py**. Publish the **project root**, including `app/`, `core/`, `requirements.txt`, and `.streamlit/config.toml`. Do not upload just the main script. No API keys or Cloud secrets are required for the current application.

## 1. Prepare Git on this Mac

Open Terminal. Git currently cannot run on this machine because Apple's Command Line Tools are missing. Install them:

```bash
xcode-select --install
```

Complete the macOS installation dialog, then open a new Terminal and verify:

```bash
git --version
```

If Git already works, skip the installation. Set the name and email you want attached to commits (a GitHub-provided private email is also valid):

```bash
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR EMAIL"
```

## 2. Check the application locally

```bash
cd /Users/ishmeetsingh/Documents/Codex/2026-11-01/mak
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app/main.py
```

For a fresh checkout, first create a virtual environment with Python 3.13: `python3.13 -m venv .venv`. Exact runtime versions are pinned in `requirements.txt`; development/test dependencies are in `requirements-dev.txt`. `.python-version` records the intended version for local version managers; select Python 3.13 explicitly in Community Cloud's Advanced settings as described below.

## 3. Create an empty GitHub repository

1. Sign in at https://github.com and choose **New repository** (or open https://github.com/new).
2. Choose a repository name, for example `equity-monte-carlo`.
3. Select Public or Private. For a private repository, grant Streamlit access when connecting your GitHub account.
4. Leave the options to initialize a README, .gitignore and license unchecked, since this project already contains files.
5. Click **Create repository** and copy its HTTPS URL, such as `https://github.com/YOUR_USERNAME/equity-monte-carlo.git`.

## 4. Commit and push the project

From the project directory, run:

```bash
cd /Users/ishmeetsingh/Documents/Codex/2026-11-01/mak
git init -b main
git add .
git status --short
```

Review the staged list. It should include source, tests, documentation, requirements, `.python-version`, and `.streamlit/config.toml`. It must exclude `.venv/`, `.cache/`, `.streamlit/secrets.toml`, local `.env` files, generated reports, and scratch downloads. `.gitignore` excludes these automatically; `work/benchmark.py` is intentionally included.

Then run the following, replacing `YOUR_USERNAME` and the repository name with the real values:

```bash
git commit -m "Prepare equity Monte Carlo app for Streamlit deployment"
git remote add origin https://github.com/YOUR_USERNAME/equity-monte-carlo.git
git push -u origin main
```

For HTTPS authentication, use your GitHub credential manager/browser sign-in when offered, or a GitHub personal access token in place of a password if prompted. GitHub account passwords do not authenticate Git pushes. Do not embed a token in the remote URL or commit it. SSH is also supported if you already have a GitHub SSH key configured.

If this folder already has a Git repository, skip `git init`; inspect `git status`, `git branch --show-current` and `git remote -v` first. Use the existing intended branch and remote rather than overwriting them. If `origin` already exists, inspect it rather than running `git remote add origin` again. These instructions assume the empty repository created in step 3; an existing remote with commits requires reconciling histories before pushing. No force push is needed.

Verify on GitHub that `app/main.py`, `core/`, `requirements.txt` and `.streamlit/config.toml` are present.

## 5. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io and sign in/connect GitHub. Grant access to the repository.
2. Click **Create app**, then **Yup, I have an app** if that question appears.
3. Select the repository `YOUR_USERNAME/equity-monte-carlo`.
4. Set **Branch** to `main` (or the branch you pushed).
5. Set **Main file path** to `app/main.py`.
6. Open **Advanced settings** and select **Python 3.13**, matching the tested environment. The Cloud Python selector is authoritative; `.python-version` is for local tooling. Leave Secrets empty.
7. Optionally choose an available app subdomain, then click **Deploy**.
8. Watch build logs until dependency installation and startup finish. The result is a URL ending in `.streamlit.app`.

The root `requirements.txt` is used for runtime installation; do not select `requirements-dev.txt` or `requirements.lock.txt`. The latter is a historical full local environment snapshot, including unused tooling; it is not the Cloud dependency manifest. No `packages.txt` or system packages are needed by this application. `.streamlit/config.toml` sets headless mode, disables usage statistics, and sets the theme; do not hardcode a server port for Cloud.

## 6. Check the deployed application

1. Enable **Synthetic offline demo data**, use 1,000 paths and a short horizon, and click **Run simulation**. Check all five tabs and a CSV/HTML download.
2. Disable demo, run `AAPL, RELIANCE.NS` with weights `0.5, 0.5`, and verify the displayed data date. This checks Yahoo access from the Cloud host separately from simulation correctness.
3. Try a benchmark comparison and monthly contribution, then test **Load 10Y government yield**. Verify the displayed yield observation date.
4. Start normal usage around 10,000 paths and one year. Large path counts, horizons, comparisons, report generation and concurrent visitors can exceed Cloud memory despite the app's 600 MB stored-output guard. The guard is not a total process-memory guarantee.

Local price caches are disposable and shared within the server, not durable cloud storage or private per-user storage. Reboots/redeployments may discard them. Simulation results live in each Streamlit session and disappear when that session ends or the process restarts. No broker credentials or holdings are persisted to disk by the application.

Yahoo may rate-limit shared cloud IP addresses even when local downloads work. Clear provider errors are shown; retry later or use the labelled synthetic demo. Government yields have explicit indicative fallbacks. Neither a working demo nor a fallback rate establishes live-data availability.

## 7. Publish updates

Edit files, run the tests, then push:

```bash
source .venv/bin/activate
python -m pytest -q
git add app core tests README.md DEPLOYMENT.md requirements.txt requirements-dev.txt .streamlit/config.toml .python-version .gitignore
git commit -m "Describe the update"
git push
```

Streamlit follows the configured GitHub branch. Check the app/build logs after a push, especially when requirements change. Use **Manage app** to inspect logs or reboot a stuck process. If the deployment Python version needs changing, follow Streamlit's Python upgrade instructions, which may require deleting and redeploying the app.

## Troubleshooting

- **Git fails with xcode-select:** complete step 1; installing Python does not install Apple's Git tooling.
- **Repository is absent from the Cloud selector:** check GitHub permissions; enter repository coordinates manually if necessary.
- **ModuleNotFoundError for core/app:** confirm the entire repository was pushed and the main file path is `app/main.py`.
- **Installation fails:** confirm Python 3.13 and root requirements.txt; inspect the first build error. The tests here ran on macOS, not Community Cloud's Linux image; review Linux wheel availability if dependency versions change.
- **App restarts or exceeds resources:** lower paths/horizon and disable comparisons; begin with the demo at 1,000 paths.
- **No data/stale quote:** verify suffixes, dates and provider availability; Refresh data retries downloads.
- **No working local Python 3.13 command:** install Python 3.13 or select it using your Python version manager before creating a fresh environment.

## Official references

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.github.com/en/get-started/git-basics/set-up-git
- https://docs.github.com/en/repositories/creating-and-managing-repositories/adding-locally-hosted-code-to-github
