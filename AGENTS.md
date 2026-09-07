# AI Developer Workflow Rules

You must strictly execute tasks out of `todo.md` by advancing through these 4 rigid approval phases. Do not skip phases or execute code early.

## Phase 1: Planning & Context Gathering
1. Announce the task or item you are pulling from `todo.md`.
2. Propose a new local git branch naming convention matching: `feature/short-description` or `fix/short-description`.
3. Read the relevant project files and cross-reference `CLAUDE.md` constraints to map out the current state.
4. **STOP and wait for user approval.** Present your granular, step-by-step implementation plan in the terminal chat. Do not touch or modify any files yet.

## Phase 2: Implementation & Staging Review
1. Once the user says to proceed, create/switch to the branch and implement the necessary code changes.
2. Run any verification tests required by the project (e.g., `python src/evaluate.py`) using your terminal tools to ensure everything passes cleanly.
3. Map out your proposed Git commits as bullet points. Format them using Conventional Commits (`feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`) with files changed listed underneath.
   - **Summary Line Constraint:** The commit summary line must be a short headline **strictly under 50 characters** (e.g., `feat: build recommendation pipeline`).
   - **Structure:** Ensure there is a mandatory blank line between the summary headline and the description body.
   - **Commit Body:** Use the description body to detail technical trade-offs, historical constraints, or architectural decisions. Wrap body lines at 72 characters.
4. **STOP and wait for user approval.** Do not actually run any commit commands yet.

## Phase 3: Committing & Cleaning
1. Upon approval, execute the actual `git commit` commands locally for each staged chunk using the exact formatting approved in Phase 2.
2. Run a final `git status` to ensure the working directory is entirely clean.

## Phase 4: Delivery (GitHub Pull Request)
1. Run `git push origin HEAD` to push your branch up to the remote repository.
2. Use the GitHub CLI tool (`gh pr create`) to open a brand-new Pull Request. Set the title to match the primary commit message, and provide a clear, professional 2-sentence description summarizing our changes for an engineering reviewer.
3. Output the direct URL link to the newly generated GitHub Pull Request in the terminal.