# Secret And Path Audit

Generated: 2026-06-11

Source matches were reviewed by pattern before packaging. The release candidate uses empty placeholders in `.env.example`; generated outputs, local caches, model weights, logs, raw datasets, complete third-party repositories, and local experiment outputs are excluded.

## Final Release-Candidate Results

- High-confidence credential scan: pass, zero matches.
- Machine-specific path scan: pass, zero matches.
- Broad audit scan: review pass; remaining matches are empty env placeholders, API parameter names, token-related variable names, and audit metadata.
- Legacy-name public scan: pass, zero matches outside migration, rename, and inventory provenance files.

## Source Audit Counts

- Source files scanned: 3291
- Included source files: 75
- Excluded source files: 3216
- Source files with legacy-name markers before migration: 158
- Source files with sensitive/path audit markers before packaging: 562
