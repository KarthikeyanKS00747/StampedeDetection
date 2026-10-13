# Dataset policy and first benchmark

The first benchmark is the **University of Minnesota Unusual Crowd Activity (UMN)** dataset. Its eleven sequences start with normal walking and end with sudden collective escape motion. It is appropriate for validating the system's motion-escalation behaviour, but it is **not** a real stampede dataset and must not be described as one.

The downloader fetches the official AVI file only after you run it. Keep the source/attribution with any report or demo. Do not mix unlicensed web clips into this project.

After download, run the preview extractor and inspect the contact sheet. Karthikeyan should mark any unsuitable source in `data/review/clip_review.csv`; this prevents it from entering training.
