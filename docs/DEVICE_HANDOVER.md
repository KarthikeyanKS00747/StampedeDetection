# Device handover and missing-items checklist

Prepared 11 October 2026. Inventory stated by Karthikeyan: two Raspberry Pi 4 4GB boards, one buzzer, two LEDs, two resistors, charging cables for both boards and connecting wires. Other items are unconfirmed, not assumed absent.

## Who gets what now

| Person | Give/access now | Needed later |
| --- | --- | --- |
| Karthikeyan KS | Keep your RTX 4070 laptop and dataset/training workspace | Approved footage, annotations and access to edge Pi for final testing |
| Samridh Dey | Project copy and his own laptop/browser | Access to Pi B after Samrudh configures it; trusted trained model from Karthikeyan |
| Samrudh Om | Both Pis, both cables AND suitable adapters, buzzer, both LEDs/resistors and wires | Verified storage, prototyping equipment and LAN access |

This arrangement puts integration hardware together and avoids splitting the alarm parts from their controller. Samridh does not need a dedicated Pi to begin software development. If teammates work in separate locations, Samrudh can configure both first, then temporarily hand Pi B to Samridh with its documented alarm assembly; Pi A remains with Samrudh. Arrange local-network integration when the boards are reunited. Do not expose SSH/API ports publicly to compensate for separate locations.

## Confirm before handover

| Item | Quantity | Why / status |
| --- | --- | --- |
| Raspberry Pi 4 4GB | 2 | Already stated; label A edge and B fog/alarm |
| Good-quality USB-C 5V/3A power supply, or official 5.1V/3A supply | 2 | Adapters not confirmed. Cables alone cannot power the Pis |
| Boot storage, normally microSD with Raspberry Pi OS | 2 | Not confirmed; a 32GB card per Pi is a practical project choice, not a hard requirement |
| Compatible card reader | 1 shared | Needed to image microSD cards unless already prepared or another boot method is used |
| LEDs | 2 | Already stated; identify colours and polarity |
| Verified LED series resistors | 2 | Already stated but values unknown; verify 220–330 ohm values for this proposed setup |
| Jumper wires and secure breadboard/prototyping arrangement | 1 set | Wires stated; breadboard not confirmed. Appropriate connector types must fit the parts |
| Buzzer specifications or compatible driver module/circuit | 1 | Buzzer stated; voltage, current and active/passive type unknown. Leave disconnected until verified |
| Trusted Wi-Fi/router or wired LAN | 1 shared network | Needed for laptop/Pi and edge/fog communication; internet only for setup/downloads |
| Laptop for each teammate | Access | Development, SSH, browser and documentation; no dedicated GPU needed by them |

Optional, not required to start: monitor plus micro-HDMI cable/keyboard/mouse for setup, Ethernet cables, cooling appropriate to sustained loads, and a compatible USB camera for a later live demo. Start with saved video; do not buy a camera just to unblock software work. A raw phone camera feed is not automatically supported by the current code.

Before buying anything, check what each teammate already owns. Do not purchase a buzzer driver until the buzzer specifications are known. The project must not create dangerous crowd behaviour for testing.

## Shared software handover

- Complete project copy with `edge/`, `fog/`, `shared/`, `tests/`, `deploy/`, `scripts/` and documentation.
- [Samridh's handbook](HANDBOOK_SAMRIDH_DEY.md) and [Samrudh's handbook](HANDBOOK_SAMRUDH_OM.md).
- Current facts: four basic tests pass; no trained risk model yet; 50 source candidates are not 50 training files; 8 catalogue videos have preliminary visual sampling; physical Pi tests remain unverified.
- Samridh owns `fog/app.py`, `fog/alerts.py`, `fog/static/` and fog API tests. Coordinate model-interface changes in `fog/classifier.py` with Karthikeyan.
- Samrudh owns deployment/service files, verified hardware documentation and performance evidence. Coordinate `fog/gpio_alarm.py` changes with Samridh.
- Karthikeyan owns edge/features, preprocessing, labels, event-grouped training/evaluation and the schema's feature meanings. Changes to `shared/schemas.py` need agreement from all three.
- Do not share browser passwords, private accounts or unlicensed footage. Share credentials privately only when necessary, not inside code or handbooks.

## Sign-off record to fill together

Record each board's label, serial/identifier if needed, storage, supply rating, owner, handover date, condition, host/IP, OS version and expected return date. Record resistor values, LED colours, buzzer specifications and whether buzzer wiring is approved. Photograph the verified assembly and keep a copy with Samrudh's report.

## Official hardware references

- [Pi 4 specifications and power requirements](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/).
- [Raspberry Pi setup and storage guidance](https://www.raspberrypi.com/documentation/computers/getting-started.html).
- [GPIO and hardware documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html).
