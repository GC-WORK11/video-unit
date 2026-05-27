"""Evaluation context for VideoUnit evaluators."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:
    from typing_extensions import TypeAlias


@dataclass
class ObjectTrack:
    """Represents a tracked object's trajectory over time.

    Attributes:
        object_id: Unique identifier for this object instance.
        class_name: Semantic class (e.g., "person", "car", "ball").
        frames: List of frame numbers where this object was detected.
        timestamps: Corresponding timestamps for each frame (ISO format strings).
        bboxes: List of [x1, y1, x2, y2] bounding boxes per frame.
        confidences: Detection confidence per frame (0.0 to 1.0).
        mask_paths: Optional paths to segmentation masks per frame.
    """

    object_id: str
    class_name: str
    frames: list[int]
    timestamps: list[str]
    bboxes: list[list[float]]
    confidences: list[float]
    mask_paths: Optional[list[Optional[str]]] = None

    def __post_init__(self):
        if self.mask_paths is None:
            self.mask_paths = [None] * len(self.frames)

    def duration_frames(self) -> int:
        """Number of frames the object is present."""
        return len(self.frames)

    def get_bbox_at(self, frame_number: int) -> Optional[list[float]]:
        """Get bounding box at a specific frame, or None if not present."""
        try:
            idx = self.frames.index(frame_number)
            return self.bboxes[idx]
        except ValueError:
            return None


@dataclass
class PerceptionResult:
    """Results from AETHER's neural perception pipeline.

    Attributes:
        tracks: Object tracks from CoTracker3.
        masks: Semantic segmentation masks from SAM2.
        depth: Optional depth maps.
        optical_flow: Optional optical flow fields.
        scene_detected: Whether scene change was detected at any frame.
        scene_cut_frames: List of frame numbers where scene cuts occur.
        metadata: Additional perception metadata.
    """

    tracks: list[ObjectTrack] = field(default_factory=list)
    masks: Optional[list[Any]] = None
    depth: Optional[list[Any]] = None
    optical_flow: Optional[list[Any]] = None
    scene_detected: bool = False
    scene_cut_frames: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _convert_cotracker_to_tracks(cotracker_result: dict, fps: float = 30.0) -> list[ObjectTrack]:
    """Convert CoTracker3 output format to ObjectTrack list.

    CoTracker3 returns tracks as:
        {
            "tracks": [  # list of frames
                [  # list of track points per frame
                    {"id": 0, "x": 100.0, "y": 200.0, "visibility": 0.9},
                    ...
                ],
                ...
            ],
            "frame_count": N,
            "track_count": M
        }

    Each unique "id" represents one tracked point across all frames.
    """
    if not cotracker_result or "tracks" not in cotracker_result:
        return []

    frames_data = cotracker_result["tracks"]
    if not frames_data:
        return []

    # Group track points by their id across frames
    # track_points[track_id] = [(frame_idx, x, y, visibility), ...]
    track_points: dict[int, list[tuple]] = {}

    for frame_idx, frame_tracks in enumerate(frames_data):
        if not frame_tracks:
            continue
        for point in frame_tracks:
            track_id = point.get("id", point.get("point_id", 0))
            if track_id not in track_points:
                track_points[track_id] = []
            track_points[track_id].append((
                frame_idx,
                point.get("x", 0.0),
                point.get("y", 0.0),
                point.get("visibility", 0.5),
            ))

    # Convert to ObjectTrack objects
    object_tracks = []
    for track_id, points in track_points.items():
        if not points:
            continue
        # Sort by frame index
        points.sort(key=lambda p: p[0])
        frames = [p[0] for p in points]
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        confidences = [p[3] for p in points]

        # Build bboxes as small squares around each point (10x10)
        # since CoTracker3 only tracks points, not bounding boxes
        bboxes = [[x - 5, y - 5, x + 5, y + 5] for x, y in zip(xs, ys)]

        # Build timestamps
        timestamps = [f"{int(f // 3600):02d}:{int((f % 3600) // 60):02d}:{f % 60:06.3f}" for f in frames]

        object_tracks.append(ObjectTrack(
            object_id=f"track_{track_id}",
            class_name="unknown",
            frames=frames,
            timestamps=timestamps,
            bboxes=bboxes,
            confidences=confidences,
            mask_paths=None,
        ))

    return object_tracks


def _build_perception_result(perception_data: Any) -> PerceptionResult:
    """Build a PerceptionResult from various input formats.

    Handles:
    - None: returns empty PerceptionResult
    - dict: extracts tracks from cotracker-style dict, builds ObjectTrack list
    - PerceptionResult: returns as-is
    """
    if perception_data is None:
        return PerceptionResult()

    if isinstance(perception_data, PerceptionResult):
        return perception_data

    if isinstance(perception_data, dict):
        # Handle AetherNeuralCore.run() output format
        # which has "tracking": {"tracks": [...], "track_count": N, ...}
        tracking = perception_data.get("tracking", {})
        if tracking and isinstance(tracking, dict):
            tracks = _convert_cotracker_to_tracks(tracking)
        else:
            tracks = []

        # Handle segmentation masks
        segmentation = perception_data.get("segmentation", {})
        masks = segmentation.get("masks") if isinstance(segmentation, dict) else None

        # Handle depth
        depth_data = perception_data.get("depth", {})
        depth = depth_data.get("depth") if isinstance(depth_data, dict) else None

        return PerceptionResult(
            tracks=tracks,
            masks=masks,
            depth=depth,
            scene_detected=False,
            scene_cut_frames=[],
            metadata={
                "frame_count": perception_data.get("frame_count", 0),
                "total_time_s": perception_data.get("total_time_s", 0),
                "vram_peak_gb": perception_data.get("vram_peak_gb", 0),
            },
        )

    return PerceptionResult()


@dataclass
class EvaluationContext:
    """Context passed to evaluators containing all video and analysis data.

    Attributes:
        video_path: Path to the video file being evaluated.
        contract: Parsed contract YAML as a dictionary.
        perception_result: Results from AETHER's perception pipeline.
        tracks: Shortcut to perception_result.tracks.
        frames_dir: Directory containing extracted video frames.
        run_dir: Directory for this evaluation run's outputs.
        available_inputs: Set of input names available for evaluation.
        video_metadata: Video properties (fps, duration, resolution, etc.).
    """

    video_path: str
    contract: dict
    perception_result: Union[PerceptionResult, dict, None]
    frames_dir: Path
    run_dir: Path
    tracks: list[ObjectTrack] = field(default_factory=list)
    available_inputs: set[str] = field(default_factory=set)
    video_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Resolve perception_result to PerceptionResult if needed
        if not isinstance(self.perception_result, PerceptionResult):
            resolved = _build_perception_result(self.perception_result)
            self.perception_result = resolved

        # Provide convenient access to tracks from perception_result
        if not self.tracks and self.perception_result:
            self.tracks = self.perception_result.tracks

        # Build available inputs from what's present
        if not self.available_inputs:
            self.available_inputs = set()
            if self.perception_result and self.perception_result.tracks:
                self.available_inputs.add("tracks")
            if self.perception_result and self.perception_result.masks is not None:
                self.available_inputs.add("masks")
            if self.perception_result and self.perception_result.depth is not None:
                self.available_inputs.add("depth")
            if self.perception_result and self.perception_result.optical_flow is not None:
                self.available_inputs.add("optical_flow")
            if self.frames_dir and hasattr(self.frames_dir, 'exists') and self.frames_dir.exists():
                self.available_inputs.add("frames")
            elif self.frames_dir:
                self.available_inputs.add("frames")
            if self.video_metadata:
                self.available_inputs.add("metadata")

    def get_frame_path(self, frame_number: int) -> Path:
        """Get the path to a specific frame image.

        Args:
            frame_number: Zero-indexed frame number.

        Returns:
            Path to the frame image file.
        """
        frames_dir = Path(self.frames_dir) if not isinstance(self.frames_dir, Path) else self.frames_dir
        return frames_dir / f"frame_{frame_number:06d}.jpg"

    def get_thumbnail_path(self, frame_number: int, suffix: str = "") -> Path:
        """Get path for a thumbnail generated during evaluation.

        Args:
            frame_number: Frame number.
            suffix: Optional suffix for the thumbnail filename.

        Returns:
            Path for the thumbnail image.
        """
        suffix_part = f"_{suffix}" if suffix else ""
        run_dir = Path(self.run_dir) if not isinstance(self.run_dir, Path) else self.run_dir
        return run_dir / f"thumb_{frame_number:06d}{suffix_part}.jpg"

    def frame_to_timestamp(self, frame_number: int) -> str:
        """Convert frame number to ISO timestamp string.

        Args:
            frame_number: Zero-indexed frame number.

        Returns:
            Timestamp string in format "HH:MM:SS.fff".
        """
        fps = self.video_metadata.get("fps", 30.0)
        total_seconds = frame_number / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
