Unused file organization (2026-04-19)

Policy:
- Non-destructive cleanup by moving likely-unused generated artifacts to archive.
- Keep experiment folders and datasets unchanged.

Moved from repository root:
- XGB_possessed_perspective_model.joblib
- XGB_seer_perspective_model.joblib
- XGB_villager_perspective_model.joblib
- XGB_werewolf_perspective_model.joblib
Destination:
- archive/unused_candidates_20260419/root_model_exports/

Moved from notebooks/:
- XGB_human_perspective_model.joblib
- XGB_seer_perspective_model.joblib
- XGB_werewolf_perspective_model.joblib
Destination:
- archive/unused_candidates_20260419/notebook_model_exports/

Removed cache directories:
- All __pycache__ directories under repository tree.

Notes:
- If you need to restore moved files, copy them back from the archive folders.
