# Dataset curation update — 14 October 2026

## What is now recorded

All 50 catalogue IDs have human observations. V001–V024 are Karthikeyan's review. V025 contains both Karthikeyan's cuts concern and an unnamed teammate's size concern. V026–V050 contain the teammate's review. These are separate from the earlier assistant's 8 browser-sampled assessments. A human 'useful' opinion is not training approval or a risk label.

Original review wording is paraphrased in `data/review/video_candidates_50.json` without converting event identities into risk labels. The existing `user_decision: Pending` represents final dataset acceptance, not absence of feedback.

## Curation decisions

- Start further motion/occupancy assessment with V005, V007, V008, V009, V013, V015, V018, V019, V020, V021, V022, V023, V033, V040, V043, V044, V045, V046, V047. This is a review route, not a download list or a count of independent events.
- Keep a separate routine-crowd/hard-negative pool. Seated crowds, quiet gatherings and empty areas can help detect false alarms. Jumping, waving, chanting and flashing lights should not automatically trigger danger.
- V020/V022 may duplicate each other: conservatively group them until checked. Duplicate clips and crops must never cross train/test.
- V007/V019 need a crowd ROI: large stage/background areas should not contribute motion features.
- Small distant people in V008/V015 require actual detector/count validation. A pleasing overhead view alone does not make YOLO counts accurate.
- V009/V018/V023 and other moving/zooming views need camera-motion assessment. Cropping does not remove camera motion.
- V024/V025/V026/V048 contain reported cuts. Split into uninterrupted shots first. V026's suggested 0–4s is only a proposed interval and is short.
- Flashing illumination in V001/V002/V007/V013 can distort optical flow. Keep these outside the clean initial motion baseline until tested; they may later be useful challenge examples. Grayscale alone does not solve flashing lights.

No ROI masking, camera stabilisation, illumination correction or automatic cut detector is implemented by this curation update. These remain preprocessing work, not solved by the timing fix.

## Your 10 local files

I inspected nine evenly spaced decoded frames per file, not full continuous playback. Sampled previews preserve neither proof of incident identity nor exact onset timing. Original files in Downloads were not edited, moved or deleted. The new manifest records metadata, SHA-256 file hashes, proposed intervals and unresolved provenance.

| ID | File | Duration | Preliminary route |
| --- | --- | ---: | --- |
| L001 | 04_jun_25_speech_ifrmit_1749042567_720 (online-video-cutter.com).mp4 | 35.32s | Incident reference / difficult-view candidate |
| L002 | 2.mp4 | 25.926s | Priority routine-flow candidate |
| L003 | 3.mp4 | 25.926s | Priority dense-flow candidate |
| L004 | Charlie Kirk shooting_ Video captures panic at Utah campus as gunshot heard.mp4 | 64.267s | Potential onset/escape case-study candidate |
| L005 | ezyZip.mp4 | 21.855s | Incident reference / difficult-view candidate |
| L006 | gettyimages-2193687205-640_adpp (online-video-cutter.com).mp4 | 39.206s | Routine-action hard-negative candidate; rights hold |
| L007 | gettyimages-2203499024-640_adpp.mp4 | 8.008s | Routine-action hard-negative candidate; rights hold |
| L008 | ssvid.net--Video-Stampede-storms-entrance-of-Astroworld-festival_v720P.mp4 | 14.982s | Potential running/surge candidate |
| L009 | Video Project 10.mp4 | 11.867s | Priority entrance-congestion review candidate |
| L010 | Video Project 11.mp4 | 12.6s | Potential motion-transition reference |

### Best next candidates

- **L002 (`2.mp4`) and L003 (`3.mp4`)**: strongest starting pair for routine versus denser flow. Same visible structural scene, so one conservative event group. Neither is automatically labelled High Risk.
- **L009 (`Video Project 10.mp4`)**: entrance/railings crowd candidate. Visible occlusion and PTI branding; review behaviour and rights.
- **L008 (Astroworld-named clip)**: visible running/approach candidate, but handheld motion and black sidebars. Source/event identity must be checked; do not confuse an entrance surge with the later fatal crowd crush.
- **L004 (campus incident-named clip)**: possible dispersal/transition reference. Long introduction, pan and changing viewpoint; proposed 24–40s and 40–60s intervals need exact onset review.
- **L010 (`Video Project 11.mp4`)**: separate the early street shot from the later wide shot. The sampled transition occurs between approximately 4.7s and 6.3s; these are not an exact cut boundary.
- **L001/L005**: crowded entrance case-study candidates with substantial overlays, occlusion and shot changes.
- **L006/L007**: visible Getty watermarks; hold out of training/evaluation absent explicit ML permission. Raised arms/activity may supply hard-negative examples if cleared, not automatic incident labels.

Local preview contact sheets are in `outputs/local_video_review/`; the source catalogue is `data/review/local_video_manifest.json`. Two differently named files are not necessarily independent events. Video resolutions/crops also change pixel-motion scale, even after timing is corrected.

## Rights: what stops acquisition/training

[Pexels terms, section 8](https://www.pexels.com/terms-of-service/) restrict unauthorised automatic extraction for ML and systematic copying. Do not bulk-download the 50 links on the strength of a free-download badge.

[Getty's agreement, section 3.11](https://www.gettyimages.com/eula) requires explicit authorisation for ML/AI use, including model evaluation. A downloaded watermarked preview or general royalty-free purchase is not evidence of that authorisation.

For the local news/incident files, no source URL or training-use permission has been supplied. Cropping does not grant rights. All 10 local entries therefore retain `training_approved: false`. No watermarks were removed.

## Implementation completed while permissions are pending

- Live and offline motion now share displacement-to-pixels/second conversion. Still flow samples are retained consistently.
- Recorded-video feature timing now follows constant-frame-rate media time rather than detector execution speed. Both extraction paths default to a 5 FPS sampling target.
- Variable-frame-rate source timing and edited-speed footage are not validated. Metadata/frame counts are not proof of original real-time playback.
- Label templates leave risk/reviewer/event ID blank and rights false by default.
- Training requires reviewed labels, event IDs, explicit rights approval/evidence and consistent motion units.
- Training splits by event, refuses leakage-prone one-event-per-class evaluation and checks that observed classes are present on both sides. Each observed class needs at least two event groups; this is a minimal guard, not sufficient data for reliable performance.
- A future trained model records feature order, units and train/test event IDs beside its model file.
- Preview generation was corrected to preserve aspect ratio, and all 10 contact sheets were regenerated. These remain sparse review images, not quantitative geometry measurements.

Heuristic thresholds still need recalibration after the motion-unit change. The feature rows are sampled snapshots, not full temporal-window aggregates; no model accuracy is established.

## Verification

18 focused tests passed: feature, fog unit, motion conversion and training-validation tests. CLI help for edge/training/feature extraction and Python compilation passed. The complete suite stalled in the existing fog integration tests and was stopped. Those tests and other teammates' fog changes were not altered. Do not report the whole suite as passing.

No trained risk model was produced. No remote video was downloaded during this update.

## Minimal information needed from you now

Provide the original source links and any reuse/ML permission information for **`2.mp4`, `3.mp4`, `Video Project 10.mp4` and `Video Project 11.mp4` first**. If these are your own recordings, state that and any editing/speed changes. If they are downloaded stock/news footage, identify the source. 'Downloaded for free' is not licence evidence.

The proposed intervals in the local manifest are review targets, not final risk annotations. Next we can examine uninterrupted windows and record observable behaviour with uncertainty. Do not relabel a whole video solely from its title or the fact that an incident happened at that event.
