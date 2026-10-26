# Data provenance

The input JSON files are copied byte-for-byte from the supplied repository. No benchmark observation was repaired, fabricated, or removed.

## PubMedQA

Source: <https://github.com/pubmedqa/pubmedqa>. The supplied `ori_pqal.json` has 1,000 records; `test_ground_truth.json` identifies 500 official test IDs. Filtering and applying the retained split rule produces 482 cases (264 yes, 165 no, 53 maybe). This local cohort check does not authenticate the original upstream download revision: that revision was not recorded in these PubMedQA files.

## MedCTA

Source: <https://huggingface.co/datasets/IVUL-KAUST/MedCTA>, revision `0be777092121a18ba2a85a0c4145a9d4fe8ed7db`, as recorded in the supplied adaptations. `subset_v1.json` has 11 cases; `fullset_v1.json` has 107. Original images are remote references and were not fetched in this cleanup. The schema is this repository's adapted replay representation, not a claim that these JSON files are verbatim upstream raw data.

The duplicated observation in full-set case 19 is preserved and documented in [known issues](reproducibility.md). Pinned source URLs do not themselves verify remote image availability today.

## File SHA-256

| File | SHA-256 |
|---|---|
| `data/pubmedqa/ori_pqal.json` | `8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d` |
| `data/pubmedqa/test_ground_truth.json` | `939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51` |
| `data/medcta/subset_v1.json` | `ccc5fd9ef8682fc336f50ecf73dbc38c34f7a092d863d04c77e38507675b8ae7` |
| `data/medcta/fullset_v1.json` | `faceb65f25fa15430122f5e5b6c7b443893c8f68b634f319f0d87c06d2e88cd3` |

Dataset and underlying-image terms remain separate from the team code license; see [third-party notices](../THIRD_PARTY_NOTICES.md).
