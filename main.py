#!/usr/bin/env python3

import argparse
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Optional: probe video HxW via OpenCV
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("lerobot-converter")


@dataclass
class ConverterConfig:
    input_dir: Path
    output_dir: Path
    joint_names: List[str]
    fps: float = 20.0
    action_mode: str = "next_state"  # "next_state" or "delta_next"
    task_text: str = "Pick up the bottle from the table"
    camera_key: str = "observation.images.ego_view"
    video_codec_hint: str = "avc1"
    default_video_shape: Tuple[int, int, int] = (224, 224, 3)  # (H, W, C)
    compat_extra_files: bool = False  # write stats.json and tasks/*.parquet


class LeRobotConverter:
    def __init__(self, cfg: ConverterConfig):
        self.cfg = cfg

        # Output dirs
        (self.cfg.output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.cfg.output_dir / "videos" / "chunk-000" / self.cfg.camera_key).mkdir(parents=True, exist_ok=True)
        (self.cfg.output_dir / "meta").mkdir(parents=True, exist_ok=True)

        # Running state
        self.total_episodes = 0
        self.total_frames = 0
        self.episode_info = []  # dicts: {"episode_index", "length", "has_video"}
        self._global_index_offset = 0
        self._video_shape_cache: Optional[Tuple[int, int, int]] = None  # (H, W, C)

    # ---------- CSV Reading ----------
    def _read_episode_csv(self, csv_path: Path) -> pd.DataFrame:
        expected = self.cfg.joint_names
        epi_candidates = {"episode", "episode_number", "episode id", "episode_id"}  # lowercased
        alias_map = {
            "motor_b": "motor_base",
            "episode_number": "episode_number",
            "episode": "episode_number",
            "episode id": "episode_number",
            "episode_id": "episode_number",
        }
        # 1) try header
        try:
            df0 = pd.read_csv(csv_path, sep=None, engine="python")
            df0.columns = [c.strip() for c in df0.columns.astype(str)]
            rename_map = {}
            for col in df0.columns:
                key = col.lower()
                if key in alias_map:
                    rename_map[col] = alias_map[key]
            if rename_map:
                df0 = df0.rename(columns=rename_map)
            drop_cols = [c for c in df0.columns if c.lower() in epi_candidates]
            if drop_cols:
                df0 = df0.drop(columns=drop_cols)
            if all(j in df0.columns for j in expected):
                return df0[expected].copy()
        except Exception:
            pass
        # 2) headerless
        df1 = pd.read_csv(csv_path, header=None, sep=None, engine="python")
        ncols = df1.shape[1]
        if ncols < len(expected):
            raise ValueError(
                f"{csv_path.name}: only {ncols} columns found; need at least {len(expected)}. "
                "Check delimiter or file integrity."
            )
        if ncols >= len(expected) + 1:
            first_col = df1.iloc[:, 0]
            int_like = pd.to_numeric(first_col, errors="coerce").dropna().apply(float.is_integer).all()
            if ncols == len(expected) + 1 or int_like:
                df1 = df1.iloc[:, 1:1 + len(expected)].copy()
            else:
                df1 = df1.iloc[:, -len(expected):].copy()
        else:
            df1 = df1.iloc[:, :len(expected)].copy()
        df1.columns = expected
        if all(isinstance(x, str) for x in df1.iloc[0].tolist()):
            row0 = [s.strip().lower() for s in df1.iloc[0].tolist()]
            exp_set = set([s.lower() for s in expected] + list(alias_map.keys()))
            if all(val in exp_set for val in row0):
                df1 = df1.iloc[1:].reset_index(drop=True)
        return df1

    # ---------- Build obs/actions ----------
    def _build_obs_and_actions(self, states_f32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs = states_f32
        act = np.empty_like(obs)
        if obs.shape[0] == 1:
            act[0] = 0.0 if self.cfg.action_mode == "delta_next" else obs[0]
            return obs, act
        if self.cfg.action_mode == "delta_next":
            act[:-1] = obs[1:] - obs[:-1]
            act[-1] = 0.0
        else:
            act[:-1] = obs[1:]
            act[-1] = act[-2]
        return obs, act

    # ---------- Video helpers ----------
    def _probe_video_shape(self, video_path: Path) -> Optional[Tuple[int, int, int]]:
        if not _HAS_CV2:
            return None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return (h, w, 3)
        except Exception:
            pass
        return None

    def _maybe_cache_video_shape(self, video_path: Optional[Path]):
        if self._video_shape_cache is not None:
            return
        if video_path and video_path.exists():
            shp = self._probe_video_shape(video_path)
            self._video_shape_cache = shp if shp is not None else self.cfg.default_video_shape
        else:
            self._video_shape_cache = self.cfg.default_video_shape

    def _find_video_for_episode(self, ep_idx: int) -> Optional[Path]:
        candidates = [
            self.cfg.input_dir / f"episode_{ep_idx}.mp4",
            self.cfg.input_dir / f"Episode_{ep_idx}.mp4",
            self.cfg.input_dir / f"episode_{ep_idx:06d}.mp4",
            self.cfg.input_dir / f"Episode_{ep_idx:06d}.mp4",
        ]
        for p in candidates:
            if p.exists():
                return p
        pat = re.compile(rf"(^|[^0-9])0*{ep_idx}([^0-9]|$)")
        for p in self.cfg.input_dir.glob("*.mp4"):
            if pat.search(p.stem):
                return p
        return None

    def _copy_video(self, video_path: Path, ep_idx: int) -> bool:
        try:
            out = self.cfg.output_dir / "videos" / "chunk-000" / self.cfg.camera_key / f"episode_{ep_idx:06d}.mp4"
            shutil.copy2(video_path, out)
            return True
        except Exception as e:
            log.warning(f"Failed to copy video for episode {ep_idx}: {e}")
            return False

    # ---------- Parquet writing ----------
    def _write_episode_parquet(self, ep_idx: int, obs: np.ndarray, act: np.ndarray) -> Tuple[int, Path]:
        n = obs.shape[0]
        if n == 0:
            raise ValueError(f"Episode {ep_idx}: zero frames. Remove or fix the CSV.")
        fps = float(self.cfg.fps)
        timestamps = (np.arange(n, dtype=np.float32) / fps)
        frame_idx = np.arange(n, dtype=np.int64)
        ep_col = np.full(n, ep_idx, dtype=np.int64)
        gidx = np.arange(self._global_index_offset, self._global_index_offset + n, dtype=np.int64)
        self._global_index_offset += n
        next_done = np.zeros(n, dtype=bool); next_done[-1] = True

        arr_state = pa.array(obs.tolist(), type=pa.list_(pa.float32()))
        arr_action = pa.array(act.tolist(), type=pa.list_(pa.float32()))
        table = pa.table({
            "observation.state": arr_state,
            "action":            arr_action,
            "timestamp":         pa.array(timestamps),
            "episode_index":     pa.array(ep_col),
            "frame_index":       pa.array(frame_idx),
            "index":             pa.array(gidx),
            "next.done":         pa.array(next_done),
            "task_index":        pa.array(np.zeros(n, dtype=np.int64)),
        })
        out_path = self.cfg.output_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, out_path)
        return n, out_path

    # ---------- Per-episode stats (count shape = (1)) ----------
    def _episode_stats(self, obs: np.ndarray, act: np.ndarray) -> dict:
        n, J = int(obs.shape[0]), int(obs.shape[1])
        def as_vec_1d(x: np.ndarray) -> List[float]:
            return np.asarray(x, dtype=np.float32).reshape(J,).tolist()
        def stats_for(x: np.ndarray) -> dict:
            mean = x.mean(axis=0); std  = x.std(axis=0); vmin = x.min(axis=0); vmax = x.max(axis=0)
            d = {"count": [n], "mean": as_vec_1d(mean), "std": as_vec_1d(std), "min": as_vec_1d(vmin), "max": as_vec_1d(vmax)}
            assert isinstance(d["count"], list) and len(d["count"]) == 1
            for k in ("mean","std","min","max"): assert isinstance(d[k], list) and len(d[k]) == J
            return d
        return {"observation.state": stats_for(obs), "action": stats_for(act)}

    # ---------- Optional compatibility outputs ----------
    def _write_tasks_parquet(self):
        out_dir = self.cfg.output_dir / "meta" / "tasks" / "chunk-000"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"task_index": [0], "task": [self.cfg.task_text]})
        df.to_parquet(out_dir / "file_000.parquet", index=False)

    # (A) WRITE stats.json IN THE SHAPE YOU REQUESTED
    def _write_stats_json(self):
        """
        Writes meta/stats.json with the structure:
          - For scalar features (index, timestamp, frame_index, episode_index, task_index):
              {"min":[x], "max":[x], "mean":[x], "std":[x], "count":[N]}
          - For vector features (observation.state, action):
              {"min":[J], "max":[J], "mean":[J], "std":[J], "count":[N]}
          - For the declared camera feature (e.g., observation.images.ego_view):
              channel-first 3x1x1 stats:
                min/max/mean/std: [[[c0]], [[c1]], [[c2]]],  count:[N_with_video]
              NOTE: We do NOT decode MP4s; we assume normalized [0,1].
        """
        meta_dir = self.cfg.output_dir / "meta"
        data_dir = self.cfg.output_dir / "data" / "chunk-000"

        # Aggregators
        total_N = 0
        # scalar sums
        scalars = {
            "index":        {"sum":0.0, "sumsq":0.0, "min":None, "max":None},
            "timestamp":    {"sum":0.0, "sumsq":0.0, "min":None, "max":None},
            "frame_index":  {"sum":0.0, "sumsq":0.0, "min":None, "max":None},
            "episode_index":{"sum":0.0, "sumsq":0.0, "min":None, "max":None},
            "task_index":   {"sum":0.0, "sumsq":0.0, "min":None, "max":None},
        }
        # vector sums (per joint)
        vec_keys = ["observation.state", "action"]
        sums = {k: None for k in vec_keys}
        sumsqs = {k: None for k in vec_keys}
        vmins = {k: None for k in vec_keys}
        vmaxs = {k: None for k in vec_keys}

        # for image "count"
        frames_with_video = 0
        has_video_by_ep = {e["episode_index"]: e["has_video"] for e in self.episode_info}
        length_by_ep    = {e["episode_index"]: e["length"]     for e in self.episode_info}

        # Iterate over episodes
        for ep in sorted(length_by_ep.keys()):
            p = data_dir / f"episode_{ep:06d}.parquet"
            ds = pd.read_parquet(p, columns=[
                "observation.state","action","timestamp","index","frame_index","episode_index","task_index"
            ])
            # lengths
            n = len(ds)
            total_N += n

            # scalar updates
            for key in ("index","timestamp","frame_index","episode_index","task_index"):
                col = ds[key].to_numpy()
                s = scalars[key]
                s["sum"]   += float(col.sum())
                s["sumsq"] += float((col.astype(np.float64)**2).sum())
                cmin, cmax = float(col.min()), float(col.max())
                s["min"] = cmin if s["min"] is None else min(s["min"], cmin)
                s["max"] = cmax if s["max"] is None else max(s["max"], cmax)

            # vector updates
            for key in vec_keys:
                arr = np.stack(ds[key].to_list()).astype(np.float64)  # [n, J]
                if sums[key] is None:
                    sums[key]  = arr.sum(axis=0)
                    sumsqs[key]= (arr**2).sum(axis=0)
                    vmins[key] = arr.min(axis=0)
                    vmaxs[key] = arr.max(axis=0)
                else:
                    sums[key]  += arr.sum(axis=0)
                    sumsqs[key]+= (arr**2).sum(axis=0)
                    vmins[key]  = np.minimum(vmins[key], arr.min(axis=0))
                    vmaxs[key]  = np.maximum(vmaxs[key], arr.max(axis=0))

            # image frame-count if this episode had a video
            if has_video_by_ep.get(ep, False):
                frames_with_video += n

        def scalar_finalize(s):
            N = float(total_N)
            mean = s["sum"]/N if N else 0.0
            var  = s["sumsq"]/N - mean*mean if N else 0.0
            std  = float(np.sqrt(max(var, 0.0)))
            minv = float(s["min"] if s["min"] is not None else 0.0)
            maxv = float(s["max"] if s["max"] is not None else 0.0)
            return {"min":[minv], "max":[maxv], "mean":[mean], "std":[std], "count":[int(total_N)]}

        def vector_finalize(k):
            N = float(total_N)
            mean = (sums[k]/N).astype(np.float32).tolist() if N else []
            var  = (sumsqs[k]/N - (sums[k]/N)**2) if N else np.zeros_like(sums[k])
            std  = np.sqrt(np.maximum(var, 0.0)).astype(np.float32).tolist()
            minv = vmins[k].astype(np.float32).tolist()
            maxv = vmaxs[k].astype(np.float32).tolist()
            return {"min":minv, "max":maxv, "mean":mean, "std":std, "count":[int(total_N)]}

        out = {
            "index":         scalar_finalize(scalars["index"]),
            "timestamp":     scalar_finalize(scalars["timestamp"]),
            "frame_index":   scalar_finalize(scalars["frame_index"]),
            "episode_index": scalar_finalize(scalars["episode_index"]),
            "task_index":    scalar_finalize(scalars["task_index"]),
            "observation.state": vector_finalize("observation.state"),
            "action":            vector_finalize("action"),
        }

        # Camera stats with [C=3,1,1] layout; we don't decode videos, so we assume [0,1] normalized range.
        # If you later store images directly (not MP4), swap this to real stats.
        C = 3
        def ch3(v):  # [[[v]], [[v]], [[v]]]
            return [[[float(v)]] for _ in range(C)]
        if frames_with_video > 0:
            out[self.cfg.camera_key] = {
                "min":   ch3(0.0),
                "max":   ch3(1.0),
                "mean":  ch3(0.5),
                "std":   ch3(np.sqrt(1.0/12.0)),  # ~0.288675 for U[0,1]
                "count": [int(frames_with_video)],
            }

        with (meta_dir / "stats.json").open("w") as f:
            json.dump(out, f, indent=2)

    # ---------- Main convert ----------
    def convert(self):
        log.info(f"Input:  {self.cfg.input_dir}")
        log.info(f"Output: {self.cfg.output_dir}")
        log.info(f"Joints: {self.cfg.joint_names} | FPS: {self.cfg.fps} | action_mode: {self.cfg.action_mode}")

        csv_files = sorted(
            p for p in self.cfg.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv" and p.name.lower().startswith("episode_")
        )
        if not csv_files:
            raise FileNotFoundError("No episode_*.csv files found in input directory.")

        meta_dir = self.cfg.output_dir / "meta"
        episodes_jsonl = (meta_dir / "episodes.jsonl").open("w")
        tasks_jsonl = (meta_dir / "tasks.jsonl").open("w")
        episodes_stats_jsonl = (meta_dir / "episodes_stats.jsonl").open("w")

        # Single task entry
        tasks_jsonl.write(json.dumps({"task_index": 0, "task": self.cfg.task_text}) + "\n")
        tasks_jsonl.close()

        first_video: Optional[Path] = None

        for csv_file in csv_files:
            m = re.match(r"^episode_(\d+)\.csv$", csv_file.name, re.IGNORECASE)
            if not m:
                log.warning(f"Skipping non-standard CSV name: {csv_file.name}")
                continue
            ep_idx = int(m.group(1))

            df_raw = self._read_episode_csv(csv_file)
            df_num = df_raw.apply(pd.to_numeric, errors="coerce")
            if df_num.isna().any().any():
                bad = df_raw[df_num.isna().any(axis=1)]
                first_bad_idx = int(bad.index[0])
                raise ValueError(
                    f"{csv_file.name}: non-numeric values detected in joint columns at row {first_bad_idx}. "
                    f"Example row: {bad.loc[first_bad_idx].to_dict()}"
                )
            states = df_num[self.cfg.joint_names].to_numpy(dtype=np.float32)

            obs, act = self._build_obs_and_actions(states)
            n_frames, parquet_path = self._write_episode_parquet(ep_idx, obs, act)

            vid_path = self._find_video_for_episode(ep_idx)
            has_video = False
            if vid_path:
                has_video = self._copy_video(vid_path, ep_idx)
                if has_video and first_video is None:
                    first_video = vid_path

            self.total_episodes += 1
            self.total_frames += n_frames
            self.episode_info.append({"episode_index": ep_idx, "length": n_frames, "has_video": bool(has_video)})

            stats = self._episode_stats(obs, act)
            episodes_stats_jsonl.write(json.dumps({"episode_index": ep_idx, "stats": stats}) + "\n")

            episodes_jsonl.write(json.dumps({
                "episode_index": ep_idx,
                "tasks": [self.cfg.task_text],
                "length": n_frames
            }) + "\n")

            log.info(f"Episode {ep_idx:06d}: frames={n_frames} video={'yes' if has_video else 'no'} -> {parquet_path.name}")

        episodes_jsonl.close()
        episodes_stats_jsonl.close()

        self._maybe_cache_video_shape(first_video)
        self._write_info_json()     # writes info.json (with chunks_size)
        self._write_stats_json()    # (A) writes stats.json in requested shape

        if self.cfg.compat_extra_files:
            # (kept: tasks parquet for your convenience; not required for your request)
            self._write_tasks_parquet()
            log.info("Wrote compatibility file: meta/tasks/chunk-000/file_000.parquet")

        log.info(f"Done. Episodes: {self.total_episodes}  Frames: {self.total_frames}")

    # (B) info.json with chunks_size added
    def _write_info_json(self):
        meta_dir = self.cfg.output_dir / "meta"
        video_shape = self._video_shape_cache or self.cfg.default_video_shape
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(self.cfg.joint_names)],
                "names": self.cfg.joint_names
            },
            "action": {
                "dtype": "float32",
                "shape": [len(self.cfg.joint_names)],
                "names": self.cfg.joint_names
            },
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "next.done":     {"dtype": "bool",    "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
            self.cfg.camera_key: {
                "dtype": "video",
                "shape": list(video_shape),  # [H, W, 3]
                "names": ["height", "width", "channel"],
                "info": {"video.fps": float(self.cfg.fps), "video.codec": self.cfg.video_codec_hint}
            }
        }
        info = {
            "codebase_version": "v2.1",
            "robot_type": "custom_robot",
            "fps": float(self.cfg.fps),
            "total_episodes": int(self.total_episodes),
            "total_frames": int(self.total_frames),
            "total_tasks": 1,
            "total_videos": int(sum(1 for e in self.episode_info if e["has_video"])),
            "total_chunks": 1,
            "chunks_size": max(1, self.total_episodes),  # (B) required for chunk mapping
            "splits": {"train": f"0:{self.total_episodes}"},
            "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": f"videos/chunk-{{episode_chunk:03d}}/{self.cfg.camera_key}/episode_{{episode_index:06d}}.mp4",
            "features": features
        }
        with (meta_dir / "info.json").open("w") as f:
            json.dump(info, f, indent=2)


# ---------- CLI ----------
def parse_args() -> ConverterConfig:
    ap = argparse.ArgumentParser(description="Convert CSV+MP4 episodes to LeRobot v2.1 dataset")
    ap.add_argument("--input", required=True, type=Path, help="Input directory containing episode_*.csv (+ optional episode_*.mp4)")
    ap.add_argument("--output", required=True, type=Path, help="Output dataset root directory")
    ap.add_argument("--fps", type=float, default=20.0, help="Frames per second for timestamps and video metadata")
    ap.add_argument("--action-mode", choices=["next_state", "delta_next"], default="next_state",
                    help="Action encoding: 'next_state' targets or 'delta_next' deltas")
    ap.add_argument("--joint-names", nargs="+", default=["motor_base", "motor_1", "motor_2", "motor_3", "motor_e"],
                    help="Ordered joint names (and CSV column order if headerless)")
    ap.add_argument("--task", default="Pick up object and place it on target", help="Task text for tasks.jsonl / episodes.jsonl")
    ap.add_argument("--camera-key", default="observation.images.ego_view", help="Camera feature key (used in videos/...)")
    ap.add_argument("--video-codec", default="avc1", help="Codec hint for viewer (e.g., avc1, h264, mp4v)")
    ap.add_argument("--compat-extra-files", action="store_true",
                    help="Also write v2.0/v3-style files: meta/tasks/chunk-000/file_000.parquet")
    args = ap.parse_args()
    return ConverterConfig(
        input_dir=args.input.resolve(),
        output_dir=args.output.resolve(),
        joint_names=args.joint_names,
        fps=args.fps,
        action_mode=args.action_mode,
        task_text=args.task,
        camera_key=args.camera_key,
        video_codec_hint=args.video_codec,
        compat_extra_files=args.compat_extra_files,
    )


def main():
    cfg = parse_args()
    LeRobotConverter(cfg).convert()


if __name__ == "__main__":
    main()
