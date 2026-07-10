# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
# Run MediaPipe inference on every N-th frame. Skipped frames reuse cached
# landmarks and replay the last-drawn skeleton at ~zero cost.
INFERENCE_EVERY = 2

# ---------------------------------------------------------------------------
# Squat thresholds
# ---------------------------------------------------------------------------
# Depth: hip-Y must pass below knee-Y by this many pixels to qualify as "deep"
DEPTH_HIP_BELOW_KNEE_PX = -5

# Knee-over-toe: knee x may extend beyond ankle x by at most this fraction of
# hip width before it's flagged as excessive forward travel
KNEE_TOE_SLACK_FRACTION = 0.08

# Valgus: ankle_width - knee_width beyond this fraction of ankle_width = collapse
VALGUS_SLACK_FRACTION = 0.04

# Back angle (torso vs vertical): above this many degrees = leaning too far
BACK_LEAN_MAX_FROM_VERTICAL = 50.0

# ---------------------------------------------------------------------------
# Rep-counter state machine thresholds (fractions of frame height)
# ---------------------------------------------------------------------------
DESCENT_THRESHOLD  = 0.04
BOTTOM_THRESHOLD   = 0.15
ASCENT_THRESHOLD   = 0.05
COMPLETE_THRESHOLD = 0.06

# ---------------------------------------------------------------------------
# Scoring deductions (points off per issue per frame)
# ---------------------------------------------------------------------------
DEDUCT_DEPTH    = 25
DEDUCT_KNEE_TOE = 15
DEDUCT_VALGUS   = 20
DEDUCT_BACK     = 15

# ---------------------------------------------------------------------------
# Voice / feedback timing
# ---------------------------------------------------------------------------
VOICE_COOLDOWN_SEC     = 4.0   # seconds before re-saying same cue
GOOD_REP_VOICE_COOLDOWN = 8.0  # don't compliment too often
CONSECUTIVE_FRAMES     = 6     # frames of bad form before alerting

# ---------------------------------------------------------------------------
# Coaching messages (cycled in order to avoid repetition)
# ---------------------------------------------------------------------------
ISSUE_MESSAGES: dict[str, list[str]] = {
    "Go deeper": [
        "Go deeper. Hip must pass below your knee.",
        "Not deep enough. Drop those hips.",
        "Sit lower. Break parallel.",
    ],
    "Knees too far forward": [
        "Knees too far forward. Push your hips back.",
        "Shift weight into your heels.",
        "Sit back more. Load the hips, not the knees.",
    ],
    "Push knees out": [
        "Push your knees out. Don't let them cave in.",
        "Drive your knees outward. Follow your toes.",
        "Knees collapsing. Spread the floor with your feet.",
    ],
    "Keep chest up": [
        "Keep your chest up. Don't lean forward.",
        "Proud chest. Upright torso.",
        "Less forward lean. Stay tall.",
    ],
    "Good rep": [
        "Great rep!",
        "Perfect form. Keep going.",
        "Beautiful squat.",
        "Solid rep. Stay tight.",
    ],
    "Rep complete": [
        "Rep complete.",
        "Nice one.",
        "Good work.",
    ],
}
