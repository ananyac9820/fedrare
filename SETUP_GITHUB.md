# Pushing this to GitHub

I cannot create the repository or push on your behalf — that needs your GitHub
credentials, which I should never handle. These are the exact steps; it takes about
two minutes.

## 1. Create an empty repo on GitHub

Go to https://github.com/new

- Repository name: `fedrare` (or your preference)
- Visibility: **Private** for now — make it public when the paper is submitted
- Do **not** tick "Add a README", "Add .gitignore", or "Choose a license".
  Those files already exist here and would cause a conflict.

## 2. Push from your machine

Unzip this folder, open a terminal inside it, then:

```bash
git init
git add .
git commit -m "Initial scaffold: data loading, metrics, config, project plan"
git branch -M main
git remote add origin https://github.com/<your-username>/fedrare.git
git push -u origin main
```

If prompted for a password, GitHub needs a Personal Access Token, not your account
password: Settings -> Developer settings -> Personal access tokens -> Tokens (classic),
with `repo` scope.

## 3. Add your teammates

Settings -> Collaborators -> Add people, and invite the rest of the team.

## 4. Confirm .gitignore is working

```bash
git status --ignored | head -30
```

The dataset must never be committed. It is ~9 GB, licence-restricted, and GitHub rejects
files over 100 MB. `.gitignore` already excludes `data/`, image files, and checkpoints —
verify before your first push that no image files are staged.
