# Raw_to_leRobot_transformation

It’s designed for **robotics logs** (joint states) paired with ego-view videos (if available).

> ✨ Highlights: headerless/quirky CSVs supported, robust episode parsing, exact Parquet schema, optional video probing (OpenCV), and **stats/info** emitted in the layout you asked for.

---

## Why this exists

You often have piles of `episode_*.csv` with motor/joint values and sometimes `episode_*.mp4` next to them. You want a **single command** to mint a clean LeRobot-style dataset that downstream tools can load consistently. This tool does precisely that.

---

## Install

### From source
```bash
git clone https://github.com/YOUR_ORG/lerobot-converter.git
cd lerobot-converter
pip install numpy, pandas, pyarrow

python3 main.py \
  --input /path/to/raw_episodes \
  --output /path/to/dataset \
  --fps 20 \
  --action-mode next_state \
  --joint-names motor_base motor_1 motor_2 motor_3 motor_e \
  --task "Pick up object and place it on target" \
  --camera-key observation.images.ego_view
```

# Data Requirements & Outputs

## Input layout

- The input directory **must** contain files named `episode_*.csv` and may optionally include matching `episode_*.mp4`.

---

## CSV expectations

- Accepts **headerless** CSVs or CSVs **with a header**.
- If a header is present, the following columns (any case/spacing) are **ignored**:
  - `episode`, `episode_number`, `episode id`, `episode_id`
- **Aliases supported**:
  - `motor_b` → `motor_base`
- **Headerless files**: column **order** is taken from `--joint-names`.
- If line 1 looks like a mixed “names” row, it’s **auto-detected and dropped**.
- **Non-numeric cells** trigger a clear error with the **offending row** printed.

---

## Video expectations

We look for a matching MP4 for each episode using any of these patterns:

- `episode_{i}.mp4`, `Episode_{i}.mp4`
- `episode_{i:06d}.mp4`, `Episode_{i:06d}.mp4`
- **Or** any `.mp4` whose **stem contains the integer `i`** (zero-padding allowed)

If **OpenCV** is installed, we **probe** H×W once from the first found video; otherwise a **default** shape is used.

---

## Outputs

The converter writes a LeRobot v2.1-style dataset:

```
<output>/
  data/
    chunk-000/
      episode_000001.parquet
      episode_000002.parquet
      ...
  videos/
    chunk-000/
      observation.images.ego_view/
        episode_000001.mp4
        episode_000002.mp4
        ...
  meta/
    info.json
    stats.json
    episodes.jsonl
    tasks.jsonl
    episodes_stats.jsonl
```

### Per-episode Parquet schema

- `observation.state` : `List[float32]` of size **J**
- `action`            : `List[float32]` of size **J**
- `timestamp`         : `float32`
- `episode_index`     : `int64`
- `frame_index`       : `int64`
- `index`             : `int64` (global running index)
- `next.done`         : `bool` (true on the last frame of the episode)
- `task_index`        : `int64` (always 0 for single-task datasets)

### `meta/info.json`

- Includes:
  - **features schema**
  - **fps**, totals
  - **chunks_size**
  - `data_path`, `video_path`
  - video **shape/codec** hint

### `meta/stats.json`

- **Scalars** (`index`, `timestamp`, `frame_index`, `episode_index`, `task_index`):  
  `{min, max, mean, std, count}` with **each value wrapped in a single-element list**.
- **Vectors** (`observation.state`, `action`): per-joint `{min, max, mean, std}` arrays + **`count`** as a single-element list.
- **Camera key** (e.g., `observation.images.ego_view`): emits **channel-first `3×1×1`** stats with a **uniform[0,1] assumption** (frames aren’t decoded).  
  `count` equals the **total frames** across episodes that had a video.

### Other metadata

- `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/episodes_stats.jsonl`

### Optional compatibility output

- With `--compat-extra-files`, also writes:
  - `meta/tasks/chunk-000/file_000.parquet` (v2.0/v3-style convenience file)

---

## Action modes

- **`next_state` (default)**:  
  `action[t] = state[t+1]` (the last action is duplicated from `t = T-2`).
- **`delta_next`**:  
  `action[t] = state[t+1] - state[t]` and `action[T-1] = 0`.
