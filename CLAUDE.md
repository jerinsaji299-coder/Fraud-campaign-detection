# Read this first

Read README.md fully before doing anything. It is the complete, authoritative
project document — there are no separate methodology or schema docs.

## README update rule

After EVERY change (creating, modifying, or deleting a file, adding a
feature, changing a config, adding a dependency, passing a stage, or making
any decision), update every affected README section in the same step, and
add a dated Changelog entry. A task is not complete until the README
reflects it. Never let the README describe something that no longer exists.

## Core rules

- Never change a frozen research definition (README section "Frozen
  research definitions") without the user's explicit approval.
- Every random/stochastic step takes an explicit, documented seed.
- No account-identity or label-derived features are ever used as model
  inputs.
- Never modify the raw data or `docs/reference/01_data_verification.ipynb`
  (read-only evidence). Never run that notebook locally.
- The frontend never computes research metrics (except the Visibility Lab
  toy, which must use the exact backend visibility definition) and never
  displays invented data as if it were a real result — use empty states
  instead.

## Working mode

Work in stages. Stop for the user's approval at the end of each stage.
