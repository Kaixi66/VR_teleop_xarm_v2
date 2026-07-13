# UF850 VR Data Collection

For daily operation, you only need two scripts:

```bash
./easy_collect   # Record one episode
./reset_arm      # Reset the robot after recording
```

The hardware and Python environments are assumed to be installed already. The
other repository files are internal implementation and are not part of the
normal operator workflow.

## One-time local settings

Create the two private configuration files once:

```bash
cp config.example.toml config.toml
cp user_settings.env.example user_settings.env
chmod +x easy_collect reset_arm
```

Edit `user_settings.env` before recording:

```bash
DATASET_NAME="press_buttons"
TASK_TEXT="Press the target button."
MODE="real"
USE_HOTSPOT="true"
HOTSPOT_PASSWORD=""
```

- `DATASET_NAME` is the data folder name.
- `TASK_TEXT` is the task/instruction/prompt. These three names mean the same
  text in this project.
- Use English for `TASK_TEXT`.
- Leave `HOTSPOT_PASSWORD` empty to enter it securely when requested.

Also verify the robot IP, two camera serial numbers, Wi-Fi interface, workspace
limits, and reset pose in `config.toml`. Both local files are ignored by Git.

## Safety

- Keep the hardware emergency stop within reach.
- Keep people and obstacles outside the complete arm path.
- Keep the first live-test speed at `50 mm/s`.
- Never run `reset_arm` until the previous episode has finished saving.
- The normal `reset_arm` command moves the robot and may open the gripper.
- Verify the reset pose for this exact UF850 installation.

## New recording workflow

### First episode

Power and enable the UF850, clear the complete arm path, then:

1. Completely close the Quest teleoperation App.
2. Run:

   ```bash
   ./easy_collect
   ```

3. Connect Quest to the hotspot printed in the terminal, but keep the App
   closed.
4. Quest may display **No Internet**. This is normal for the robot-only
   hotspot; keep Quest connected.
5. Type:

   ```text
   START
   ```

6. Wait until the terminal prints:

   ```text
   COLLECTOR READY
   ```

7. Open the Quest App.
8. Keep both controllers still.
9. Wait until the terminal prints:

   ```text
   TELEOP ARMED
   ```

10. Start operating the robot and complete the task.
11. Press `Ctrl+C` once when the episode is finished.
12. Wait for the terminal to confirm that the episode was saved.

After `Ctrl+C`:

- robot motion is paused;
- the current episode is safely written to disk;
- the hotspot stays active;
- the bridge stays running for the next episode.

Do not close the terminal or turn off the hotspot while the episode is still
saving.

### Record the next episode

When the previous episode was saved successfully and the bridge is healthy:

1. Keep Quest connected to the hotspot.
2. Keep the Quest App open; it does not need to be restarted.
3. Run `./easy_collect` again.
4. Type `START`.
5. Keep both controllers still while the collector becomes ready.
6. Wait for `TELEOP ARMED`.
7. Record the next demonstration.
8. Press `Ctrl+C` and wait for the saved confirmation.

Repeat this section for every additional episode. The episode number increases
automatically:

```text
episode_000
episode_001
episode_002
...
```

If the hotspot was disabled, the bridge failed, or `reset_arm` was run, do not
use this fast sequence. Close the Quest App and follow the first-episode
sequence again.

## After all episodes are finished

First confirm that the final episode was saved. Then:

1. Close the Quest App.
2. Keep the emergency stop ready.
3. Run:

   ```bash
   ./reset_arm
   ```

4. Watch the countdown and the complete robot path.
5. Wait for:

   ```text
   RESET COMPLETE
   ```

`reset_arm` safely stops the paused bridge/ROS stack, checks the UF850, opens
the gripper when configured, moves to the configured reset joint pose, verifies
the final pose, enters STOP, and disconnects.

Only after `RESET COMPLETE` should you manually turn off the hotspot or follow
the normal lab power-down procedure.

Before using a reset pose for the first time, inspect it without robot motion:

```bash
./reset_arm --dry-run
```

The dry-run may stop a paused bridge, but it does not command reset motion.

## Short version

```text
FIRST EPISODE
  ./easy_collect
  connect Quest hotspot, keep App closed
  START
  COLLECTOR READY -> open Quest App
  keep controllers still -> TELEOP ARMED
  operate
  Ctrl+C -> wait until episode is saved

NEXT EPISODE
  keep Quest App open
  ./easy_collect
  START -> TELEOP ARMED
  operate
  Ctrl+C -> wait until episode is saved

ALL EPISODES FINISHED
  confirm final episode is saved
  close Quest App
  ./reset_arm
  wait for RESET COMPLETE
  then turn off the hotspot
```

## Saved data

Data is written under:

```text
data/<DATASET_NAME>/episode_NNN/
```

Each step contains:

```text
cam_0.jpg
cam_1.jpg
data.json
```

The shared prompt is stored once in `episode_meta.json` under `task`; it is not
repeated inside every step's `data.json`.

## Simple troubleshooting

- **Quest says No Internet:** keep it connected; the hotspot is intentionally a
  local robot network.
- **Waiting for COLLECTOR READY:** check the UF850 connection and both cameras.
- **Waiting for TELEOP ARMED:** make sure the Quest App is open and keep both
  controllers still.
- **The robot does not move:** close the App and repeat the first-episode order
  exactly.
- **reset_arm refuses to run:** the episode may still be recording or saving.
  Wait for `easy_collect` to finish and read the full error.
- **Anything moves unexpectedly:** press the hardware emergency stop.

## License

No license has been selected yet.
