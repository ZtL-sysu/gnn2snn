# S9 Evidence Manifest

All hashes are SHA-256 and were computed against the final bytes in this run root.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `evidence/final-submission-audit.md` | 6333 | `90b5affc0c213591fc237854832d7da93ef8f4d990979a5e0564125e596d950c` |
| `evidence/final-checks.json` | 3112 | `d2fcd62de3cd5f0b7dadf2febf1439b01ce6f90cc0df0e567b2e6c8e6c205b3a` |
| `evidence/submission-constraints.md` | 2636 | `938469ebc39a026bbb0148b3e1c8225e8fb108173cbb774200ec57b45b6eba4c` |
| `evidence/manuscript-metadata.md` | 2694 | `5c8a8ce161db79d17f3738e8464d8d2a9b6631c600187701306253e1ac070412` |
| `evidence/upload-map.md` | 1334 | `a68bb33f25a2be325d5706660d8dde1f5efbf52afc8d3365c986cbe91e7cfd41` |
| `working/main.tex` | 38160 | `6b160fa447e2c1f453d9d7f91f57d9535671cb98e2b06a6f982cfbd451f5cf25` |
| `working/main.pdf` | 526618 | `23e1f84488c656832fc3d6839fcda07b122759f7311c1cf8b0a039049fe4d7bb` |
| `working/supplement.tex` | 20151 | `435aedb434ba8e5c61a2f49dbdd6004afed4d9064552aee84c28a02b0d1b925b` |
| `working/supplement.pdf` | 177499 | `81af234f5b6e1d17f2a14e62aaf7e2e0873ba19234d7111df647a535298768e4` |
| `working/main.log` | 27493 | `1a48ce4975925c6195179c7fe70a3e74f984eb0bb291023370fd33673f211c98` |
| `working/supplement.log` | 24309 | `21c9f70bc91842ec4d8b4c1bcdf82f834cd4d37d0da43b6c65c0a31b47b34f54` |
| `sources/main.tex` | 38877 | `efa5dba3efa114f48c9f42bab5a870dbede6f37c7cf526818a8dfd99d642738a` |
| `sources/supplement.tex` | 20151 | `435aedb434ba8e5c61a2f49dbdd6004afed4d9064552aee84c28a02b0d1b925b` |
| `sources/references.bib` | 6182 | `2e93b8b9a8c0cf98a64d3fababb2b43cb63126da83e8f9a3e73f572d7cb490cf` |
| `sources/cvpr.sty` | 17887 | `1cd7f434b13d41c475df4a1901a5d82d9ca8b80b8cf6205dc4602a5c164286b4` |
| `sources/ieeenat_fullname.bst` | 26900 | `e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc` |
| `execution-contract-v1.md` | 1727 | `d4693986c95358f5868bc9ccd9c4069de8eb5d06efdc25a785bf4f579860a73c` |
| `evidence/submission-profile.md` | 3518 | `690f693eef86d4f93d00fda6f88f6a1146fdb6e49e2731c7b645510a4115677e` |
| `evidence/claim-disposition.md` | 9084 | `3aa8b0bf4f303aed7058e408489acb5ff4273c07439eaa1d05c673675f07ad69` |
| `evidence/module-rescore-round2.md` | 4075 | `52b85abb10caea0b75cc25c0d1856aa4940bb37684f05e4da596a4537621202e` |
| `writing/lint-round2.json` | 5943 | `209b056452bc1a3a2b2deb4b085d9582fdc1eb15e2485121996324691f6a84aa` |
| `evidence/checks-round2.json` | 762 | `452970ff30fcc5c99393f844d941c96967f14924e125b0c042fe2ebaad9644e1` |
| `delivery/gnn2snn-cvpr2027-source.zip` | 1619558 | `ca7182860211685c1447c1c0c86735c293020988094e319ca3dfbd1fabe1bc33` |

## Verification summary

- Main PDF: 8 pages; supplement PDF: 4 pages.
- Main and supplement compile logs: zero undefined citations, zero undefined references, zero overfull boxes, zero LaTeX errors.
- Clean source-ZIP extraction: all input hashes match; clean builds produced 8 and 4 pages; normalized main PDF text matches the accepted working PDF.
- Anonymity: anonymous title-page blocks; no institutional affiliation, email, or correspondence metadata in either PDF.
- LLM Usage Statement: present once in the main PDF, unchanged from baseline, with all named tools and human-responsibility language verified.
- Overall S9 result: PASS with scoped limitation `L-S9-1` (no live CVPR 2027 portal available). No major issue.
