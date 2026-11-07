# From Impressions to Attention

The study compares billboard locations using observed traffic and potential viewing time, then asks whether the evidence changes a media buying or creative decision. Phase 1 produces auditable measurements from real footage. The included highway clip is a technical test case. Its marked road section measures traffic and road-zone time.

## Phase 1 checkpoint

Use PowerShell from this project folder. A local Python environment, the YOLO11m weights, and the licensed highway clip were prepared on the project machine. On a fresh machine with Python 3.11, install the dependencies with the commands below. The model downloads on first use if `weights/yolo11m.pt` is absent. Download the [highway clip](https://www.pexels.com/video/traffic-on-the-580-freeway-san-francisco-bay-area-19347491/) and save the 1280 × 720 version as `data/raw_videos/highway_01.mp4`. If your copy has a different resolution or crop, draw a new site annotation for that copy.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the full clip. When `--output` is omitted, the script creates a new dated results folder under `data/outputs/`.

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --video data/raw_videos/highway_01.mp4 --site config/sites/highway_01.yaml --device cpu
```

The output includes `metrics.json` with the clip summary, `tracks.parquet` with anonymous vehicle positions and timestamps, and `events.parquet` with line crossings and visits. A track counts once when its estimated bottom-centre crosses the yellow line in the specified direction. Zone time is estimated from entry and exit samples. The median includes only visits with both boundaries observed. A long tracking gap ends the visit and can leave a partial record. The pipeline uses presentation timestamps and samples at 10 FPS. The licence log records the [source and reuse conditions](https://www.pexels.com/license/).

The saved full run in `data/outputs/phase1_highway/` reports 10 AI crossings over 27.40 seconds, 1 complete zone visit and 32 partial visits. The saved run in `data/outputs/phase1_manual_check/` covers the first five seconds and reports 2 AI crossings. Use the manual check below to assess that count. The high proportion of partial visits makes the full clip's dwell estimate unsuitable as a placement benchmark.

Review a short window by hand with the predictions hidden. Follow the yellow arrow and count each vehicle when the bottom of its bounding area reaches the line. Use `C` to tally, `U` to undo, space to pause, and `N` to step while paused. Compare the tally with `vehicle_passages` from the same clip window. A discrepancy is a reason to inspect the video and note misses or duplicate IDs.

```powershell
.\.venv\Scripts\python.exe scripts/review_site.py --video data/raw_videos/highway_01.mp4 --site config/sites/highway_01.yaml --limit-seconds 5
.\.venv\Scripts\python.exe scripts/run_pipeline.py --video data/raw_videos/highway_01.mp4 --site config/sites/highway_01.yaml --device cpu --limit-seconds 5
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Prepare a real billboard case

1. Pick a fixed-camera, continuous clip with a real billboard in view. Confirm the reuse permission and add a row to `data/licence_log.csv`. Save the original clip privately in `data/raw_videos/`.
2. Identify the approaching lanes from which the display face is visible. Open the annotation tool and mark a convex road zone, the counting line across the eligible lanes, and the four display corners. Check the line direction with the review tool. With line endpoints in left-to-right screen order, `negative_to_positive` counts traffic moving from above the line to below it. Reverse the direction if needed.
3. For speed, identify four road-plane landmarks with measured XY coordinates in metres and an identifiable measurement source. Supply their eight values to `--world-points` in perimeter order and describe the source with `--calibration-source`. Click the landmarks in that same order. Speed is estimated only inside the measured quadrilateral. Leave these arguments out for an uncalibrated clip. The speed field will be `null`.
4. Keep the matching site quote and audience claim, including the booking period and the owner's definition of an impression, for the marketing comparison in Phase 2.

```powershell
.\.venv\Scripts\python.exe scripts/annotate_site.py --video data/raw_videos/my_billboard.mp4 --site config/sites/my_billboard.yaml --site-id my_billboard --scenario city_day --billboard --direction negative_to_positive --zone-basis 'Approaching lanes with a potential view of the display face'
```

The sample clip has no documented ground distances or billboard visibility zone. Its output provides a traffic test, with speed unavailable. Use a measured site for billboard-specific exposure and calibrated speed.

## Phase 2 checkpoint

Open `data/outputs/phase2_highway/marketing.md` for the saved marketing scorecard. It uses the full highway run. The count review in the same folder shows two AI crossings and two confirmed manual crossings in the first five seconds, with at least one uncertain vehicle. Count error remains unavailable until there are fully adjudicated windows. The two-frame detector and tracker reports are smoke tests on hand-drawn boxes from the real clip. Keep them out of the final benchmark.

```powershell
.\.venv\Scripts\python.exe scripts/run_marketing.py --run data/outputs/phase1_highway
.\.venv\Scripts\python.exe scripts/eval_counts.py --run data/outputs/phase1_highway
```

Pass `--run` once per Phase 1 results folder to compare locations. For a site marked `billboard_visibility`, the count for potential viewability uses line crossings made while the track is inside the annotated zone. Place the counting line inside the eligible road section. This geometry supplies a proxy for an opportunity to see. The out-of-home industry distinguishes passing traffic from opportunities within the display's line of sight. [OAAA impression measurement guidelines](https://oaaa.org/wp-content/uploads/2022/10/May2021_OOHImpressionsGuidelines.pdf)

Copy `config/marketing_evidence.yaml` to `config/marketing_private.yaml` for real quotes and other sourced inputs. Pass `--evidence config/marketing_private.yaml` to the scorecard command. The input fields are:

| Evidence path | Fields and rules |
|---|---|
| `clips.<video filename>.occupancy.<vehicle class>` | `mean` people per vehicle and `source`. Optional `low` and `high` give a source-supported sensitivity range. |
| `clips.<video filename>.pedestrians` | `site_id`, `end_s` copied from the Phase 1 `observation_s`, `passing`, `viewable`, and `source`. Use zero only when an observed count is zero. |
| `placements.<site ID>.cost` | `display_id` matching the site ID, `start_date`, exclusive `end_date`, `scope`, `amount`, `currency`, `kind` (`quote` or `paid`), and `source`. |
| `placements.<site ID>.audiences` | List of records with matching `display_id`, dates and `scope`, plus `kind` (`owner_claim` or `independent`), `metric`, `value`, and `source`. Supported metrics are `circulation`, `ots`, `likelihood_to_see`, `attentive_passages`, and `attentive_seconds`. |

Use a consistent scope description for the same display face, booking and audience. The scorecard divides cost by a positive, matching audience count and multiplies by 1,000. Owner CPM is shown by the type of impression the owner quoted. Independent OTS CPM and attentive CPM require separate matching audience evidence. The ratio of independent to owner OTS compares the same metric and booking period. A short video can report what happened during its observed window. Campaign CPM requires campaign-period evidence. The plan's APM and AER abbreviations refer to formulas in `project.txt`, which was absent from this workspace. The scorecard names its implemented cost and attention measures explicitly.

For manual count validation, use `scripts/review_site.py` with `--start-seconds` and `--limit-seconds` to choose a window in the original clip. Add the tally to `data/labels/manual_crossings.csv`. Put `0` in `possible_additional` when every crossing is adjudicated. Enter a positive number for a known upper bound or leave it blank when the upper bound is unknown. The evaluator uses intervals starting at `start_s` and ending just before `end_s`. It computes count error only for exact windows and omits zero-count windows from percentage error.

### Label footage for the detector and tracker benchmark

1. Use the original video in a local CVAT task. Create `car`, `motorcycle`, `bus`, and `truck` labels. Draw vehicle boxes in **Track** mode so each vehicle keeps the same ID. Review interpolation, especially where one vehicle passes behind another. Label every visible vehicle in the selected region and interval.
2. Start with one short, clear segment. Build toward the plan's **500 labelled evaluation frames** across different clips. At the default 10 evaluated frames per second, that takes roughly 50 seconds of footage across the tasks. Reserve separate clips for threshold tuning and final evaluation.
3. Export **CVAT for video** and extract `annotations.xml`. Save the numeric XML in `data/labels/`. The evaluator decodes the video in memory and saves scores. [CVAT video export format](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/)

The following command reproduces the existing two-frame pilot. For your benchmark, replace the XML and frame range with your fully labelled interval. Use `--site` only when the annotation covers the marked road zone. Change `--task detection` to `--task tracking` for HOTA, MOTA and IDF1. Use `--tune-thresholds` only on the development clips.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
.\.venv\Scripts\python.exe scripts/eval_vision.py --task detection --video data/raw_videos/highway_01.mp4 --annotations data/labels/highway_roi_pilot.xml --first-frame 84 --last-frame 87 --site config/sites/highway_01.yaml
```

Use `--detector rfdetr` with the optional `rfdetr==1.11.2` package to compare RF-DETR Medium on the same held-out frames and device. Report detector settings along with speed. The evaluator reports inference throughput after the first scored frame and excludes video decoding. When the tracking range starts partway into a clip, ByteTrack processes the preceding frames so it has the same history as the Phase 1 run. [RF-DETR inference reference](https://rfdetr.roboflow.com/latest/learn/run/detection/)

### Toronto pilot

The original Toronto clip is `data/raw_videos/toronto_01.mp4`. The site file marks the large curved screen at 10 Dundas East, the road zone on the screen's visible side, and a line across the far-side lane. The arrow counts right-to-left travel. Use only the first 12 seconds with these coordinates. The display rotates ads, so interpret the zone as potential visibility of the screen. An individual ad would need a separate playback timeline and viewing criteria. [Original footage by Hao Le](https://www.pexels.com/video/dynamic-urban-intersection-with-billboards-34602127/) · [Branded Cities display description](https://brandedcities.com/2020/01/101278/)

```powershell
.\.venv\Scripts\python.exe scripts/review_site.py --video data/raw_videos/toronto_01.mp4 --site config/sites/toronto_01.yaml --start-seconds 3.5 --limit-seconds 4.5 --paused
```

Use **N** to step through the clip and watch the silver convertible. For this side-on shot, use the bottom centre of the vehicle at road level as the counting point. Press **C** as it crosses the yellow line in the arrow's direction. The blue outline marks the screen and the green outline marks the road zone. The saved count review in `data/outputs/phase2_toronto/` records one AI crossing and one manual crossing in this window.

For a first CVAT exercise, open the original MP4 with no frame filtering. Select the `car` class in **Track** mode and draw a box around the same silver convertible on frames **216, 228 and 240** (3.6, 3.8 and 4.0 seconds). Keep it under one track ID. This is a practice track. For a scored interval, label every eligible vehicle in the chosen region and review the frames between keyframes.

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --video data/raw_videos/toronto_01.mp4 --site config/sites/toronto_01.yaml --limit-seconds 12 --device cpu
```

## Phase 3 checkpoint

Phase 3 tracks pedestrians in the marked sidewalk area and uses SCRFD to look for a face in each person's upper-body crop. A candidate needs at least 40 pixels across its shorter face dimension in the original frame, adequate sharpness, and acceptable brightness. 6DRepNet estimates head orientation for faces that pass. The program writes aggregate counts only. It processes face crops in memory. InsightFace's pretrained detector is used here for noncommercial research. [SCRFD and model terms](https://github.com/deepinsight/insightface) · [6DRepNet implementation](https://github.com/thohemp/6DRepNet)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-attention.txt
.\.venv\Scripts\python.exe scripts/run_attention.py --video data/raw_videos/toronto_01.mp4 --site config/sites/toronto_01.yaml --scene config/attention_sites/toronto_01.yaml
```

The first 12 seconds of Toronto produced 77 pedestrian tracks in the marked area. SCRFD associated a face at least once with 31 tracks. Every associated face was below the size threshold. Track IDs can fragment, so this describes detected tracks. The Toronto report leaves billboard-facing attention unavailable because the clip also lacks measured screen-to-camera geometry. The result shows why a clip can support traffic analysis yet yield little usable evidence about pedestrian attention.

On BIWI, 144 of the 156 sampled frames passed the face quality gate. All 24 recordings were represented in the scored sample. Mean absolute head-pose error over pitch, yaw and roll was 4.41° at native resolution. On the same accepted faces, reducing the face width to 40 pixels raised error to 4.64°, and 20 pixels raised it to 6.54°. The resize check isolates the pose model after face detection. BIWI measures head orientation in a lab setting. Billboard-facing classification in street footage needs separate validation. [BIWI dataset and terms](https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose) · [fast.ai archive documentation](https://docs.fast.ai/tutorial.vision.html#points)

The saved Toronto, benchmark, and resolution plot results are in `data/outputs/phase3_toronto/`. To reproduce the benchmark, download the [fast.ai BIWI RGB archive](https://s3.amazonaws.com/fast-ai-imagelocal/biwi_head_pose.tgz) and point the command at that file. The evaluator reads the archive in place and does not extract photos.

```powershell
.\.venv\Scripts\python.exe scripts/eval_headpose.py --archive 'PATH_TO\biwi_head_pose.tgz' --stride 100 --degrade-widths 40 20
```

The online head-turn check uses a close-up video by cottonbro studio. The camera's forward direction is the reference. Five stable intervals were manually labelled before inference. The model matched all 58 sampled frames in those intervals at the 20° threshold. The saved `online_head_turn_check.json` records this single-subject demonstration. Billboard-facing validation still requires footage with a measured relationship between the target, camera, and people. [Source video](https://www.pexels.com/video/a-man-turning-his-head-sideways-5474326/)

For calibrated scenes the `geometry` mapping takes `source`, `target_camera_m` and verified `pose_axis_signs` for horizontal and vertical head movement. Camera coordinates are metres with x to the image right, y down, and z into the scene. Supply `fixed_head_camera_m` for a measured stationary head position. For moving subjects, supply `intrinsics_px` (`fx, fy, cx, cy`), `head_plane_normal`, `head_plane_offset_m` and `depth_range_m`. Mark the pedestrian area in native image pixels and verify the orientation against a person deliberately facing the target and turning away. The default rule uses a 20° cone and at least 0.4 seconds of sustained alignment. The result is a head-facing proxy. Keep ad playback times and any marketing costs separate from it.

## Phase 6 dashboard

Double-click **Open Dashboard.cmd** in the project folder. If the browser does not open automatically, use the local address printed in the terminal. Keep the terminal open while using the dashboard. Press **Ctrl+C** in the terminal to stop it.

To install the dashboard dependency or run it from PowerShell, use the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

The dashboard reads the saved Phase 2-5 results under `data/outputs/`. Keep those folders when copying the project. You can switch between Toronto and London in the manual count review and download the Edmonton attendance chart. Open **Marketing formulas** to see the published DeeLorri Apparel CPM example and download the final report. The app runs locally without rerunning the models.
