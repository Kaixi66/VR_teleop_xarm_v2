# UF850 VR Data Collection

Only two scripts are used during collection:

```bash
./easy_collect
./reset_arm
```

Before starting, check the dataset name and English task prompt in your local
settings.

## Safety

- Keep the hardware emergency stop within reach.
- Clear people and obstacles from the complete robot path.
- Do not run `reset_arm` until the episode has finished saving.
- Watch the robot during the complete reset motion.

## Workflow for every episode

1. Close the Quest App.
2. Run:

   ```bash
   ./easy_collect
   ```

3. Connect Quest to the robot hotspot, or keep it connected if it is already
   connected. Leave the App closed.
4. Type:

   ```text
   START
   ```

5. Wait for:

   ```text
   COLLECTOR READY
   ```

6. Open the Quest App.
7. Keep both controllers still and wait for:

   ```text
   TELEOP ARMED
   ```

8. Start operating the robot.
9. When the episode is finished, press `Ctrl+C` once.
10. Wait until the terminal confirms that the episode was saved.
11. Close the Quest App.
12. Run:

    ```bash
    ./reset_arm
    ```

13. Wait for:

    ```text
    RESET COMPLETE
    ```

14. Start the next episode by repeating this workflow from step 1.

## Important

- After `Ctrl+C`, robot motion pauses while the episode is saved.
- The hotspot stays active between episodes; Quest can remain connected to the
  hotspot.
- `reset_arm` stops the old bridge, so the Quest App must be closed and opened
  again after the next `COLLECTOR READY` message.
- Do not turn off the hotspot before `RESET COMPLETE`.
- After the final episode has been saved and reset is complete, you can turn
  off the hotspot.

Data is saved automatically under:

```text
data/<dataset_name>/episode_NNN/
```
