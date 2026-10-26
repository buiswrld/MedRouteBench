# Third-party materials

These notices do not assign a license to the team's own code.

## PubMedQA

- Upstream: <https://github.com/pubmedqa/pubmedqa>.
- Paper: Jin et al., *PubMedQA: A Dataset for Biomedical Research Question Answering*, EMNLP-IJCNLP 2019, <https://aclanthology.org/D19-1259/>.
- Included inputs: `data/pubmedqa/ori_pqal.json`, `data/pubmedqa/test_ground_truth.json`, and the small PubMedQA fixtures.
- The upstream repository publishes the MIT license, copyright 2019 pubmedqa. Its notice is retained in [`third_party/PubMedQA_LICENSE.txt`](third_party/PubMedQA_LICENSE.txt).
- Underlying PubMed abstract text retains its source provenance; this notice does not claim ownership of or separately relicense those abstracts.

## MedCTA

- Upstream: <https://huggingface.co/datasets/IVUL-KAUST/MedCTA>.
- Paper: Ashraf et al., *MedCTA: A Benchmark for Clinical Tool Agents*, 2026, <https://arxiv.org/abs/2606.11702>.
- Included adaptations: `data/medcta/subset_v1.json` and `data/medcta/fullset_v1.json`, converted into this repository's `schema_version: 1` replay format. Source revision, case IDs, and image URLs remain in the JSON.
- The dataset card and supplied provenance declare Apache-2.0. A standard copy is included in [`third_party/Apache-2.0.txt`](third_party/Apache-2.0.txt). Preserve applicable upstream notices when redistributing or refreshing adaptations.
- Images are referenced by upstream URLs rather than bundled; this does not establish unrestricted rights to source images.
- The known case 19 observation duplication is described in [reproducibility](docs/reproducibility.md).

## Libraries, model services, and reference scripts

OpenAI's Python SDK, python-dotenv, Tenacity, Pillow, and pytest retain their own licenses and notices. OpenRouter candidate access and any Azure judge access require the user's own credentials and applicable service/model terms. No model weights or service sponsorship are supplied by this repository.

Two standalone scripts under `paper_reference_files/medcta/` in the original archive lack established upstream revision/license notices in the supplied files. They are not imported by the core evaluators and are excluded from the cleaned snapshot; the unchanged source archive preserves them. Their provenance/licensing has not been resolved by this cleanup.

## Team code license

The uploaded repository has no top-level license for the team's implementation. The cleanup has not selected a license on the authors' behalf. The code owner should choose one and confirm incorporated-material attribution before announcing an open-source release.
