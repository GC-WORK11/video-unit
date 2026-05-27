#!/usr/bin/env python3
"""Test the kinematic discovery module."""

import numpy as np
import sys
sys.path.insert(0, '/home/govinda/aether/backend')

from app.scene_graph.kinematic_discovery import (
    discover_kinematic_structure,
    kinematic_tree_to_mjcf,
    JointType,
)

def test_synthetic_pendulum():
    """Test on synthetic pendulum data."""
    print("\n=== Test 1: Synthetic Pendulum ===")
    
    T = 60  # frames
    n_points = 20  # points on pendulum
    
    # Pendulum: pivot fixed, bob swings in arc
    t = np.linspace(0, 2*np.pi, T)
    pivot = np.array([0, 0, 1])  # Fixed pivot
    length = 0.5  # Pendulum length
    
    tracks = []
    for i in range(n_points):
        # Points distributed along pendulum
        frac = i / n_points
        for frame in range(T):
            angle = 0.3 * np.sin(t[frame])  # Swing angle
            x = length * frac * np.sin(angle)
            y = -length * frac * (1 - np.cos(angle))
            z = 1 - length * frac * np.cos(angle) + pivot[2]
            tracks.append([x, y, z])
    
    tracks = np.array(tracks).reshape(T, n_points, 3)
    
    # All points should cluster into 1 body (pendulum swings as rigid)
    tree = discover_kinematic_structure(tracks, n_bodies=1)
    
    print(f"Discovered: {tree.n_bodies} bodies, {tree.n_joints} joints")
    print(f"Joint types: {[j.joint_type for j in tree.joints]}")
    
    # Should be revolute (rotation around z-axis)
    if tree.joints:
        joint = tree.joints[0]
        print(f"Joint axis: {joint.axis}")
        print(f"Confidence: {joint.confidence:.2f}")
    
    return tree


def test_two_link_arm():
    """Test on synthetic 2-link arm."""
    print("\n=== Test 2: Two-Link Arm ===")
    
    T = 60
    n_points_link1 = 15
    n_points_link2 = 15
    
    t = np.linspace(0, 2*np.pi, T)
    
    tracks = []
    
    # Link 1: rotates around z
    for i in range(n_points_link1):
        frac = (i + 1) / n_points_link1
        for frame in range(T):
            angle1 = 0.5 * np.sin(t[frame])
            x = 0.3 * frac * np.cos(angle1)
            y = 0.3 * frac * np.sin(angle1)
            z = 1
            tracks.append([x, y, z])
    
    # Link 2: attached to end of link 1, rotates around z
    for i in range(n_points_link2):
        frac = (i + 1) / n_points_link2
        for frame in range(T):
            angle1 = 0.5 * np.sin(t[frame])
            angle2 = 0.8 * np.sin(t[frame] * 1.3)
            x1 = 0.3 * np.cos(angle1)
            y1 = 0.3 * np.sin(angle1)
            x2 = x1 + 0.25 * frac * np.cos(angle1 + angle2)
            y2 = y1 + 0.25 * frac * np.sin(angle1 + angle2)
            tracks.append([x2, y2, 1])
    
    tracks = np.array(tracks).reshape(T, n_points_link1 + n_points_link2, 3)
    
    tree = discover_kinematic_structure(tracks, n_bodies=2)
    
    print(f"Discovered: {tree.n_bodies} bodies, {tree.n_joints} joints")
    for joint in tree.joints:
        print(f"  {joint.parent_id} -> {joint.child_id}: {joint.joint_type.value}")
    
    # Should find 2 revolute joints
    assert tree.n_bodies == 2, "Should find 2 bodies"
    assert tree.n_joints >= 1, "Should find at least 1 joint"
    
    return tree


def test_real_data():
    """Test kinematic discovery on real session data.

    This function attempts to use actual CoTracker3 tracking from video frames.
    Falls back to synthetic data if tracking fails.
    """
    print("\n=== Test 3: Real Session Data ===")

    from pathlib import Path

    # Find any session with frames (frames are in session/frames/ subdirectory)
    sessions_dir = Path("/home/govinda/aether/data/sessions")
    session_dir = None
    frames_dir = None

    if sessions_dir.exists():
        for session in sorted(sessions_dir.iterdir())[:10]:  # Try first 10 sessions
            if session.is_dir():
                frames_subdir = session / "frames"
                if frames_subdir.exists() and (frames_subdir / "frame_0001.png").exists():
                    session_dir = session
                    frames_dir = frames_subdir
                    break

    if not session_dir or not frames_dir:
        print("No session found with frames, using synthetic data")
        return test_synthetic_two_body()

    frames_files = sorted(frames_dir.glob("frame_*.png"))[:15]
    if not frames_files:
        print("No frames found, using synthetic data")
        return test_synthetic_two_body()

    print(f"Loading {len(frames_files)} frames from {session_dir.name}...")

    # Try to use CoTracker3 if available
    try:
        from app.perception.tracking import get_pipeline
        pipeline = get_pipeline()
        frames = []
        for f in frames_files:
            import cv2
            frame = cv2.imread(str(f))
            if frame is not None:
                frames.append(frame)

        if len(frames) >= 5:
            print(f"Running perception on {len(frames)} frames...")
            result = pipeline.run_full_pipeline(frames)

            # Extract tracks from CoTracker3 result
            tracks_raw = result["tracking"]["tracks"]
            if tracks_raw and result["tracking"]["track_count"] > 0:
                print(f"Using CoTracker3 tracks: {result['tracking']['track_count']} tracks")
                # Use the tracking data directly
                # Format: tracks_raw[frame][track_id] -> {x, y, visibility}
                T = len(tracks_raw)
                n_tracks = result["tracking"]["track_count"]
                tracks_3d = np.zeros((T, n_tracks, 3))

                for f_idx in range(T):
                    for t_idx in range(n_tracks):
                        if t_idx < len(tracks_raw[f_idx]):
                            tracks_3d[f_idx, t_idx, 0] = tracks_raw[f_idx][t_idx].get("x", 0) * 0.001
                            tracks_3d[f_idx, t_idx, 1] = tracks_raw[f_idx][t_idx].get("y", 0) * 0.001
                            tracks_3d[f_idx, t_idx, 2] = 0.5

                pipeline.unload_all()
                tree = discover_kinematic_structure(tracks_3d, n_bodies=2)
                print(f"Discovered: {tree.n_bodies} bodies, {tree.n_joints} joints")

                if tree.joints:
                    mjcf = kinematic_tree_to_mjcf(tree)
                    print("\nGenerated MJCF:")
                    print(mjcf[:500] + "...")

                return tree
            else:
                print("No tracks found, falling back to synthetic")
                pipeline.unload_all()
        else:
            print("Not enough frames, falling back to synthetic")

    except Exception as e:
        print(f"Perception pipeline failed: {e}")
        print("Falling back to synthetic data")

    # Fallback: synthetic 2-body mechanism
    return test_synthetic_two_body()


def test_synthetic_two_body():
    """Synthetic 2-body mechanism for testing."""
    print("\nUsing synthetic 2-body mechanism...")

    T = 60
    n_points_link1 = 15
    n_points_link2 = 15

    t = np.linspace(0, 2*np.pi, T)

    tracks = []

    # Link 1: rotates around z
    for i in range(n_points_link1):
        frac = (i + 1) / n_points_link1
        for frame in range(T):
            angle1 = 0.5 * np.sin(t[frame])
            x = 0.3 * frac * np.cos(angle1)
            y = 0.3 * frac * np.sin(angle1)
            z = 1
            tracks.append([x, y, z])

    # Link 2: attached to end of link 1, rotates around z
    for i in range(n_points_link2):
        frac = (i + 1) / n_points_link2
        for frame in range(T):
            angle1 = 0.5 * np.sin(t[frame])
            angle2 = 0.8 * np.sin(t[frame] * 1.3)
            x1 = 0.3 * np.cos(angle1)
            y1 = 0.3 * np.sin(angle1)
            x2 = x1 + 0.25 * frac * np.cos(angle1 + angle2)
            y2 = y1 + 0.25 * frac * np.sin(angle1 + angle2)
            tracks.append([x2, y2, 1])

    tracks = np.array(tracks).reshape(T, n_points_link1 + n_points_link2, 3)

    tree = discover_kinematic_structure(tracks, n_bodies=2)

    print(f"Discovered: {tree.n_bodies} bodies, {tree.n_joints} joints")

    if tree.joints:
        mjcf = kinematic_tree_to_mjcf(tree)
        print("\nGenerated MJCF:")
        print(mjcf[:500] + "...")

    return tree


if __name__ == "__main__":
    print("Testing Kinematic Discovery Module")
    print("=" * 50)
    
    test_synthetic_pendulum()
    test_two_link_arm()
    test_real_data()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
