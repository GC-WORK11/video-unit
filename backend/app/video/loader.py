"""Video loading and frame extraction using PyAV — optimized for speed."""
import logging
from pathlib import Path
from typing import Literal

import av
import numpy as np

from app.core.config import settings

log = logging.getLogger(__name__)


class VideoMetadata:
    def __init__(self, path: str):
        self.path = Path(path)
        self._probe()

    def _probe(self):
        try:
            with av.open(str(self.path)) as container:
                stream = container.streams.video[0]
                self.fps = float(stream.average_rate)
                self.width = stream.width
                self.height = stream.height
                self.duration = float(container.duration / 1_000_000)
                self.frame_count = stream.frames or int(self.duration * self.fps)
                self.codec = str(stream.codec_context.name)
        except Exception as e:
            log.warning(f"Could not probe video: {e}")
            self.fps = 30.0
            self.width = 1920
            self.height = 1080
            self.duration = 0.0
            self.frame_count = 0
            self.codec = "unknown"

    def __repr__(self):
        return f"VideoMetadata({self.path.name}, {self.width}x{self.height}, {self.fps:.1f}fps, {self.duration:.1f}s)"


class FrameExtractor:
    """Extract frames from video — optimized for speed using seek().

    Strategy:
    - Always cap at max_frames (default 10, max 20 for slow machines)
    - Use av.Stream.seek() to jump to keyframe positions
    - Only decode enough frames to get N evenly-spaced samples
    """

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self.metadata = VideoMetadata(str(self.video_path))

    def extract_all_frames(
        self,
        output_dir: str | Path,
        every_nth: int = 1,
        max_frames: int | None = None,
    ) -> list[Path]:
        """Extract frames efficiently.

        Args:
            output_dir: Where to save PNG frames
            every_nth: Extract every N frames (1 = all frames)
            max_frames: Hard cap — stops after this many frames are saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []

        total = self.metadata.frame_count
        fps = self.metadata.fps

        # Cap max frames to avoid long decode times (allow up to 3000 for full video coverage)
        cap = min(max_frames or settings.max_frames, settings.max_frames)
        every_nth = max(1, total // cap) if total > cap else 1

        try:
            with av.open(str(self.video_path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"  # Enable frame-level threading

                saved = 0
                frame_idx = 0

                # Use seek() to jump to approximate position
                target_positions = [
                    int(i * every_nth)
                    for i in range(min(cap, (total // every_nth) + 1))
                ]

                for target in target_positions:
                    if saved >= cap:
                        break

                    # Seek to the nearest keyframe before target
                    # time = frame_idx / fps
                    seek_time = max(0, target / fps)
                    container.seek(int(seek_time * 1_000_000))  # microseconds

                    # Decode forward to exact target frame
                    for frame in container.decode(stream):
                        if frame_idx < target:
                            frame_idx += 1
                            continue

                        # Save this frame
                        output_path = output_dir / f"frame_{saved:05d}.png"
                        frame.to_image().save(output_path, optimize=True)
                        extracted.append(output_path)
                        frame_idx += 1
                        saved += 1

                        if saved >= cap:
                            break

                        break  # Exit decode loop, go to next target

        except Exception as e:
            log.error(f"Frame extraction failed: {e}")

        log.info(f"Extracted {len(extracted)} frames to {output_dir}")
        return extracted
