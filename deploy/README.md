# Raspberry Pi deployment

Use one Pi as the **Fog + alarm Pi** and one as the **edge Pi**.

1. Connect both Pis to the same trusted local network and assign/reserve the fog Pi's IP address.
2. Copy this repository to each Pi, then from its directory run `bash deploy/install_pi.sh`.
3. On the fog Pi, wire the LEDs and buzzer as documented in the project README. Start `python -m fog.app`, open `http://FOG_PI_IP:8000`, and check the dashboard from a phone/laptop.
4. On the edge Pi, first test the transport with `python -m edge.main --simulate --fog-url http://FOG_PI_IP:8000`. This must make the dashboard cycle through states before connecting a camera.
5. Connect and test the camera with a saved video before using source `0`. Set a measured `--area-m2`; density is meaningless without it.
6. To run at boot, copy the relevant `.service` file to `/etc/systemd/system/`, replace `FOG_PI_IP`, then run `sudo systemctl daemon-reload` and `sudo systemctl enable --now stampede-fog` or `stampede-edge`.

Do not deploy this system as the sole safety measure at an event. Validate it with venue staff and have an explicit human response process.
