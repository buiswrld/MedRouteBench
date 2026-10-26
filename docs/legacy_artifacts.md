# Retained historical artifacts

The source ZIP and its Git history are preserved in the author's original archive. The cleaned public-facing snapshot excludes generated runs and the old Git object database. It does not delete or rewrite those original records.

The original working tree contains:

| Run directory | Trace files | Manifest model | Manifest backend |
|---|---:|---|---|
| `20260804T150844.181942Z_7e7e8bd3` | 0 | gpt-5-mini | azure |
| `20260804T150921.742039Z_34416e36` | 0 | gpt-5-mini | azure |
| `20260804T151117.011007Z_9ec83250` | 2 | gpt-5-mini | azure |
| `20260804T151157.957162Z_e5adb071` | 482 | gpt-5-mini | azure |
| `20260804T154247.978728Z_57dc31bd` | 2 | gpt-5-mini | azure |
| `20260804T155252.886131Z_2f21453e` | 2 | gpt-5-mini | azure |
| `20260804T155404.290533Z_338af4c7` | 482 | gpt-5-mini | azure |

There are 970 trace files plus 14 manifest/report files. The first two runs selected zero cases. Of the three two-case runs, one reports only one completed case. The two 482-case runs each report 482 completed cases. These counts describe saved artifacts, not newly measured model performance.

The seven manifests lack the current `reversed` provenance field. They retain older code hashes and generation settings. The two complete runs have different output-token limits (512 and 2048); they are not three matched paper replicates.

Reachable historical report blobs also name `llama-3.1-8b-instant`. They do not provide a verified complete final ten-model experiment. No MedCTA report blob was found among reachable historical report paths. Unreachable Git objects and external backups have not been exhaustively audited.

Machine-readable inventory: [legacy_artifact_inventory.json](legacy_artifact_inventory.json). Personal absolute machine paths and response contents are not included in this inventory. The complete originals remain in the source archive.
