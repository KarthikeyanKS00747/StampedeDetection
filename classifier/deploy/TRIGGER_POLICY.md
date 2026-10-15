# Experimental trigger and clearance policy

**Status:** draft for measurement, not validated for automatic activation.

Each observation covers 30 source frames; successive windows start 15 frames
apart. At 25 FPS, each window covers about 1.2 seconds of frames and
successive window starts are 0.6 seconds apart.
Offline timestamps denote the **start** of a window, not when an operator
could receive an alert. Processing and queue delay must be measured live.

## Proposed state machine

| State | Observation | Action |
|---|---|---|
| Clear | 3 consecutive `abnormal` windows | Open one experimental alert episode and notify an operator in a separate channel. |
| Clear | Any `normal` window | Reset the abnormal streak. |
| Alert | 5 consecutive `normal` windows | Close the episode. |
| Alert | Any `abnormal` window | Reset the normal streak; do not open another episode. |

Missing, corrupt, or delayed frames are unknown observations. Reset both
streaks, record the gap, and require a fresh full window before further
decisions. Do not treat missing data as normal. The proposed 3/5 thresholds
have not been tuned. Do not map the observation to a risk tier or buzzer.

## Independent normal footage evaluation

Use footage from separate sources or recording sessions that were never used
to train, select, or tune this model. Include routine walking, stationary
crowds, empty/near-empty scenes, lighting changes, and camera movement.
Keep each clip's source, dates, FPS, duration, and scenario. Do not repeat
evaluation on training clips.

For each normal clip, report false-positive windows / total evaluable windows.
Apply the proposed state machine in time order. Count each transition from
Clear to Alert as one false-alert episode. Divide the number of episodes by
the total independent normal footage hours for episodes/hour, and report the
per-scenario counts too. Overlapping windows are correlated, so a window
fraction is not an independent event rate. Report uncertainty across clips
and sites when enough independent material is available.

The observed 0/345 on replayed training footage is only an **in-sample
false-positive window count/fraction**. Independent false-positive rate and
false-alert episodes/hour are unknown. Live detection delay is also unknown.

## Before activation

1. Evaluate independent normal footage and review every proposed episode.
2. Tune the trigger/clearance thresholds on separate development footage,
   then evaluate the frozen policy on a new independent set.
3. Measure end-to-end live delay from source frame capture to operator notice.
4. Verify detector weights, feature order, FPS, and windowing against the
   packaged model. Keep the existing baseline primary during experiment.
