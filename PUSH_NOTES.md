# Push Notes

This repo should use **SSH remote** for GitHub pushes.

- Correct remote: `git@github.com:blandfx/stealth-video-player.git`
- Verify: `git remote -v`
- Push: `git push -u origin main`

Why: HTTPS remote requires interactive username/token in this environment and fails.

If push fails, test SSH auth/network:
- `ssh -T git@github.com`
