# GitHub Release Checklist for PrionNER

This checklist is specific to the staged public package in `final_submission_PrionNER/`.

## 1. Decide What The Public GitHub Repo Will Contain

- Use `final_submission_PrionNER/` as the starting point for the public repository, not the full working directory.
- Keep the released repository focused on:
  - `data/`
  - `metadata/`
  - `code/`
  - `docs/`
  - top-level `README.md`
- Exclude internal-only content already outside this package, such as logs, caches, checkpoints, temporary folders, and large experiment outputs.

## 2. Resolve Release Blockers Before Publishing

- Choose and add the real license text.
- Decide whether the dataset and code use the same license or separate licenses.
- Add citation metadata.
- Confirm that publishing the released text and annotations on GitHub is permitted.

Notes:
- `LICENSE_NOT_SET.md` currently says the package must not be published as-is.
- Some source texts appear to be derived from article abstracts, so copyright and redistribution terms should be reviewed carefully before release.

## 3. Fix Repository Hygiene Issues In The Staged Package

- Replace machine-local absolute links in documentation with relative links.
- Normalize mixed document IDs in the released data.
- Add a repository `.gitignore`.
- Add a small top-level `LICENSE` file once the license is chosen.

Current status:
- GitHub-incompatible absolute links in `README.md`, `docs/dataset_card.md`, and `docs/reproducibility.md` have been replaced with relative links.
- Some released test documents still use numeric-only IDs such as `1`, `2`, `7`, `8`, `9`, `14`, and `16`, while most files use `prion_XXXX`.

## 4. Add Standard GitHub Metadata Files

- `LICENSE`
- `CITATION.cff`
- `.gitignore`
- `README.md` polish pass for GitHub rendering
- Optional:
  - `CODEOWNERS`
  - `.github/ISSUE_TEMPLATE/`
  - `.github/workflows/` for basic validation

Suggested `CITATION.cff` fields:
- title
- authors
- version
- date-released
- repository URL
- preferred citation text
- license

## 5. Sanity-Check The Data Package

- Verify file counts across raw, raw_text, fine, and coarse views.
- Verify JSON `doc_id` values match filenames after any renaming.
- Verify BRAT, JSON, and CoNLL exports are synchronized.
- Decide whether the raw zip archives should stay in the repo or move to a GitHub Release asset.

Current package status:
- Approximate size: `15M`
- This is small enough for a normal GitHub repository.

## 6. Bootstrap The Actual Repository

- Create a new GitHub repository, likely named `PrionNER`.
- Copy or move the contents of `final_submission_PrionNER/` to the repo root.
- Initialize git if starting from a local folder.
- Commit the cleaned release package.
- Push to GitHub.
- Create a tagged release, if desired.

Suggested first tags:
- `v1.0.0` for the first public dataset release
- `v1.0.1` and later only for metadata or packaging fixes that do not change annotations

## 7. Nice-To-Have Improvements

- Add a short "How to cite" section to `README.md`.
- Add a "Data statement" or "Ethics / limitations" section to the dataset card.
- Add a minimal validation script or GitHub Action that checks:
  - paired BRAT files exist
  - JSON filenames match `doc_id`
  - split counts match `metadata/dataset_summary.json`
- Decide whether model checkpoints should be published in a separate repository or release artifact.

## Recommended Order

1. Finalize license and redistribution policy.
2. Fix broken doc links and naming inconsistencies.
3. Add `CITATION.cff`, `.gitignore`, and final metadata.
4. Initialize the GitHub repo from `final_submission_PrionNER/`.
5. Run one last validation pass.
6. Push and create the first release.
