# scripts/video_defaults.sh — ONE source of truth for the deliverable-video knob defaults.
#
# Sourced by BOTH scripts/demo_video.sh (local driver) and scripts/pod_make_video.sh (on-pod
# worker) so the same semantic knob cannot silently diverge between them again (the class of bug
# where demo_video defaulted COHERENCE=1 while a direct pod_make_video run defaulted it to 0 and
# rendered raw, unsmoothed poses). Override any of these via the environment; scripts must keep
# the `VAR="${VAR:-$VIDEO_*_DEFAULT}"` pattern.

VIDEO_FRAMES_DEFAULT=60
VIDEO_STITCH_DEFAULT=1
VIDEO_COHERENCE_DEFAULT=1
VIDEO_CAMERAS_DEFAULT=broadcast,sideline,top,goal
VIDEO_DEVICE_DEFAULT=gpu
VIDEO_RES_X_DEFAULT=1280
VIDEO_RES_Y_DEFAULT=720
VIDEO_SAMPLES_DEFAULT=32
VIDEO_FPS_DEFAULT=25
VIDEO_STEP_DEFAULT=1
