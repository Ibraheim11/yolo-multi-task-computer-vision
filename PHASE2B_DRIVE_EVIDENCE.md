# Phase 2B saved Drive audit evidence

Read through connected Google Drive on 2026-11-10. No dataset was downloaded locally.

- [Original report.json](https://drive.google.com/file/d/1G-fhjQD_UWF0pKyyCtWGtuMhkhsoDJRj/view), 4,121,909 bytes.
- [Original issues.csv](https://drive.google.com/file/d/1fDLM1oH2DJuQwMLj0K36nkm1RuqpYTa8/view).
- [Original split summary](https://drive.google.com/file/d/1a7VZADDR56wi65kAMJ4_09l0vDAea8dS/view).

Confirmed classes: fire, smoke. Images: 5,596 (train 3,940; validation 1,017; test 639). Invalid files: 1/1/4. Missing labels, corrupt images and orphan labels: zero. Parsed boxes: 6,227 / 1,653 / 987. Empty background labels: 555 / 105 / 72. Audit ready: false. The 5,590 valid records exclude the six invalid files.

All seven issues have the exact rejection reason `expected 5 fields: class x_center y_center width height`:

| Split | Export-relative label | Line |
| --- | --- | ---: |
| train | `train/labels/366f79d032096cf6_jpg.rf.9450be1b6b53c48d722609a3aa828d94.txt` | 1 |
| val | `valid/labels/WaterMistFireDemonstration3447_png.rf.8b898bfa25fa838fbdea67cf21cd9556.txt` | 2 |
| test | `test/labels/barbecue602_png.rf.5e2b9b695487711ea85588846a9fc3a1.txt` | 1 |
| test | `test/labels/fire32frame52_jpg.rf.da6630cf0090b002605b5f863ce9431e.txt` | 1 |
| test | `test/labels/fire32frame52_jpg.rf.da6630cf0090b002605b5f863ce9431e.txt` | 2 |
| test | `test/labels/firesamp21_frame100_jpg.rf.45a7b54a6b0a367507eaf679be34c2fe.txt` | 4 |
| test | `test/labels/hand_held_batteries_night77_png.rf.ad70612b751c71f7b5ce14bf1ef4c8d8.txt` | 1 |

The fire32frame52 file fails on two lines. Raw rejected rows are absent from the saved report; token counts and formats require the restored export. None is yet confirmed to be segmentation, recoverable or irreparably malformed. A field-count error does not prove there are too many fields.

Follow [the diagnostic procedure](FIRE_SMOKE_LABEL_DIAGNOSIS.md). Preserve Drive originals before the notebook audit cell overwrites dataset_audit filenames.
