"""Person tracking shared by live edge ingestion and offline dry runs."""

from __future__ import annotations

from edge.motion import velocities


class PersonTracker:
    def __init__(self) -> None:
        import supervision as sv

        self._supervision = sv
        self._tracker = sv.ByteTrack()
        self._previous_centres: dict[int, tuple[float, float]] = {}

    def update(self, yolo_result, delta_s: float) -> tuple[int, list[tuple[float, float]]]:
        detections = self._supervision.Detections.from_ultralytics(yolo_result)
        tracked = self._tracker.update_with_detections(detections)
        centres: dict[int, tuple[float, float]] = {}
        vectors = []
        if tracked.tracker_id is not None:
            for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
                centre = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
                ident = int(track_id)
                centres[ident] = centre
                if ident in self._previous_centres:
                    old = self._previous_centres[ident]
                    vectors.extend(velocities(
                        [(centre[0] - old[0], centre[1] - old[1])], delta_s
                    ))
        self._previous_centres = centres
        return len(tracked), vectors
