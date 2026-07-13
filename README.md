# UF850 VR Data Collection

Only two scripts are used:

```bash
./easy_collect
./reset_arm
```

## 1. Set the dataset name and prompt

Open `user_settings.env` in the same folder as `easy_collect`.

If the file does not exist, create it once:

```bash
cp user_settings.env.example user_settings.env
```

Fill in:

```bash
DATASET_NAME="press_buttons"
TASK_TEXT="Press the target button."
```

`TASK_TEXT` is the task/instruction/prompt. Use an English sentence.

## 2. Record the first episode

1. Run `./easy_collect`.
2. Put on Quest and connect it to the robot hotspot, but do not enter the App
   yet.
3. Type `START` in the computer terminal.
4. Wait for `COLLECTOR READY`.
5. Enter the Quest App.
6. Keep both controllers still and wait for `TELEOP ARMED`.
7. Start operating the robot.
8. When recording is finished, press `Ctrl+C` once.
9. Wait until the terminal confirms that the episode was saved.

## 3. Reset after every episode

Keep the Quest App open and run:

```bash
./reset_arm
```

Wait for:

```text
RESET COMPLETE
```

Do not exit the Quest App. Quest also remains connected to the hotspot.

## 4. Record the next episode

With the Quest App still open:

1. Run `./easy_collect` again.
2. Type `START`.
3. Keep both controllers still.
4. Wait for the new `COLLECTOR READY` and `TELEOP ARMED` messages.
5. Start operating the robot.
6. Press `Ctrl+C` when finished and wait for the saved confirmation.
7. Run `./reset_arm` again and wait for `RESET COMPLETE`.

Repeat section 4 for every new episode.

## Complete loop

```text
./easy_collect
START
COLLECTOR READY
TELEOP ARMED
operate
Ctrl+C
wait until saved
./reset_arm
RESET COMPLETE
repeat
```

## Safety

- Keep the hardware emergency stop within reach.
- Keep both controllers still until `TELEOP ARMED`.
- Do not run `reset_arm` until the episode has finished saving.
- Watch the robot during reset.
- Turn off the hotspot only after the final `RESET COMPLETE`.

Data is saved under:

```text
data/<DATASET_NAME>/episode_NNN/
```
