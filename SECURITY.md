# Security

**Report a vulnerability privately.** Use GitHub's "Report a vulnerability" button on the Security tab. Please don't open a public issue.

How this repository is protected:
- **Only the owner can change it.** There are no collaborators, deploy keys or webhooks. Rulesets block everyone else from updating `main`, creating or deleting branches and tags, force-pushing and rewriting history.
- **Pull requests are proposals only.** Nothing from a pull request is merged automatically. Changes are reviewed by hand and applied in the upstream app, then published by the sync.
- **CI is read-only and can't be hijacked.** It uses only GitHub's own actions, pinned to exact commits, with a read-only token. Workflow runs from outside contributors need the owner's approval.
- **Every published change is scanned** for secrets and personal data before it is pushed. GitHub secret scanning with push protection and Dependabot alerts are on.

- **Every commit and release tag is signed.** GitHub shows them as **Verified**, and unsigned commits are rejected on `main` for everyone, the owner included.

**When you install Kobox:** use a tagged release, and check that it shows as Verified.
