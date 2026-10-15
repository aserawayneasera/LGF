# Update the existing GitHub repository

Target repository: https://github.com/aserawayneasera/LGF

The supplied ZIP is an updated repository tree, not a Git history export. Your remote repository has not been changed by preparing this package.

## Create a review branch

Clone the repository, or use your existing clean checkout. Keep any uncommitted work in a separate branch or backup first.

```bash
git clone https://github.com/aserawayneasera/LGF.git
cd LGF
git switch -c prl-major-revision-2026
```

Extract the ZIP into a separate directory. Copy the **contents** of its `lgf-prl-revision/` directory into your checkout. Include `.gitignore`. Keep your checkout's `.git/` directory.

The older top-level `train_lgf.py`, `notebooks/`, `scripts/`, and `docs/` files also exist under `legacy/` in the new package. Remove obsolete top-level files only after confirming their preserved copies, so readers start from the updated README and dispatcher. Do not delete your Git history.

On Linux/macOS, `rsync` copies the new tree without deleting unrelated files:

```bash
rsync -a /absolute/path/to/lgf-prl-revision/ /absolute/path/to/LGF/
```

Then inspect the changed tree:

```bash
git status --short
git diff --stat
python tools/check_release.py
python -m unittest discover -s tests -v
```

## Commit source and documentation

Dataset images, checkpoint files, private PDFs, raw predictions, and local environments are excluded by default. Keep dataset/checkpoint folders beneath ignored `workspace/` or `data/` directories. Inspect staged files before committing.

```bash
git add README.md run.py requirements*.txt .gitignore CITATION.cff LICENSE \
  docs experiments notebooks legacy tools tests configs provenance CHECKSUMS.sha256
git diff --cached --stat
git commit -m "Add PRL revision experiments and reproducibility documentation"
git push -u origin prl-major-revision-2026
```

Open a pull request into `main` and review the source and documentation. The branch workflow retains your old repository history and provides a reviewable update.

## Large reproducibility artifacts

Publish only artifacts you own and intend to share. Source code belongs in Git. Checkpoints and large prediction/annotation manifests belong in a separate release or research archive with checksums and provenance. Link the archive from the README after publication. No release URLs or dataset download permissions are invented in this package.

The MIT license from the existing repository is preserved. Ultralytics and the datasets have separate licensing terms. Do not replace those terms with the repository's license.
