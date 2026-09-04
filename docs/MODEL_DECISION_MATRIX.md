# SatQuery AI — Model decision matrix

| Field | Value |
| --- | --- |
| Companion | `docs/TECH_STACK.md` |
| Status | Selection decisions — no implementation |
| Date | 2026-09-04 |

**Honesty rules:** Recommend/backup names below are **real** public projects. Cells that could not be verified on an official card say **Unverified**. Paper metrics are **not** SatQuery scores. Compute figures are **order-of-magnitude** from authors or common 7B-FP16 practice, not measured on our hardware.

---

## How to read this matrix

| Column | Meaning |
| --- | --- |
| RECOMMENDED MODEL | Primary adapter for that capability |
| BACKUP MODEL | Swap via config if recommend fails (license, VRAM, weights) |
| WHY | Fit to spec + engineering |
| COMPUTE REQUIREMENT | Practical inference / train notes |
| DATASET | What the **weights or training** used (authors), plus our eval set |
| BENCHMARK | Where **we** will evaluate; not a copied leaderboard |
| IMPLEMENTATION RISK | Integration / legal / weight gaps |

---

## Capability: remote-sensing adaptation (M1) + land-cover evidence

| | |
| --- | --- |
| **RECOMMENDED MODEL** | BigEarthNet v2 (reBEN) **ResNet-50**, Sentinel-1 + Sentinel-2 (`BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0`) |
| **BACKUP MODEL** | Same architecture, Sentinel-2 only: `BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0` (**MIT** on Hugging Face). Or our continued fine-tune of ResNet-50 on the official BigEarthNet split. |
| **WHY** | Satisfies M1 with a **visual** model actually trained on BigEarthNet. Used in production for land-cover and optical–SAR **scene** complementarity. Not a generic VLM. |
| **COMPUTE REQUIREMENT** | Inference: CPU or ~2–4 GB GPU. Fine-tune: ~8–12 GB GPU. Checkpoint size on the order of **~95 MB** for the `all` HF artifact listing. |
| **DATASET** | BigEarthNet v2 / reBEN ([bigearth.net](https://bigearth.net/), CDLA-Permissive-1.0). Paper: Clasen et al., arXiv:2407.03653. |
| **BENCHMARK** | Official BigEarthNet/reBEN test split (macro-AP / authors’ protocol). **Do not paste unpublished or invented AP.** |
| **IMPLEMENTATION RISK** | Medium: `configilm` / `reben_publication` loader as documented on the HF card. **License tag on `resnet50-all-v0.2.0` must be confirmed** (S2 sibling is MIT). Band mapping from arbitrary GeoTIFF to S1/S2 layout will often **warn** (wrong sensor). |

---

## Capability: remote-sensing VQA (M2)

| | |
| --- | --- |
| **RECOMMENDED MODEL** | **GeoChat-7B** — [mbzuai-oryx/GeoChat](https://github.com/mbzuai-oryx/GeoChat), weights [MBZUAI/geochat-7B](https://huggingface.co/MBZUAI/geochat-7B) |
| **BACKUP MODEL** | **RS-LLaVA** LoRA + `Intel/neural-chat-7b-v3-3` ([BigData-KSU/RS-LLaVA](https://github.com/BigData-KSU/RS-LLaVA)). Secondary eval-only: [syvlo/RSVQA](https://github.com/syvlo/RSVQA) (GPL-3.0, skip-thoughts). Not default: EarthVQA (academic/NC). |
| **WHY** | GeoChat is an **RS-specific** grounded LVLM (CVPR 2024, arXiv:2311.15826), not ImageNet-chat. HF card **apache-2.0**; code Apache 2.0. Covers VQA with published eval code. Fuse BigEarthNet labels as evidence so land-cover is not LLM-only. |
| **COMPUTE REQUIREMENT** | ~7B VLM; plan **≥16 GB VRAM** FP16 (third-party listings ~14 GB — **unverified on our box**). CPU: treat as **not_configured**. |
| **DATASET** | GeoChat-Instruct / RS multimodal mix (authors). Backup RS-LLaVA: RS-instructions (includes RSVQA-LR per authors). |
| **BENCHMARK** | **RSVQA** (LR/HR official splits); **VRSBench** VQA split. Report only metrics we compute. |
| **IMPLEMENTATION RISK** | **High:** LLaVA-era pin (`torch==2.0.1` in GeoChat `pyproject.toml`) vs modern stack; Vicuna/LLaMA **lineage terms** besides Apache card; Windows path issues; 7B download. |

---

## Capability: remote-sensing captioning (M3)

| | |
| --- | --- |
| **RECOMMENDED MODEL** | **GeoChat-7B** caption / scene-description prompts (same checkpoint, separate adapter id) |
| **BACKUP MODEL** | **RS-LLaVA** (joint captioning + VQA in paper *Remote Sensing* 2024, doi:10.3390/rs16091477) |
| **WHY** | One RS checkpoint, two contracts (`caption` vs `vqa`). Avoids a second 7B model in VRAM. |
| **COMPUTE REQUIREMENT** | Same as GeoChat VQA (shared weights in process). |
| **DATASET** | GeoChat instruction data; RS-LLaVA: UCM-caption + UAV (authors). |
| **BENCHMARK** | **VRSBench** captioning split (BLEU/METEOR/CIDEr/CLAIR **as implemented by VRSBench eval**, only if we run it). |
| **IMPLEMENTATION RISK** | Same as GeoChat. Prompting is not a second architecture — keep separate registry entries. GitHub license for RS-LLaVA is **NOASSERTION**. |

---

## Capability: text-guided grounding (M3)

| | |
| --- | --- |
| **RECOMMENDED MODEL** | **GeoChat** referring / grounded localization (boxes as authors describe for referring object detection) |
| **BACKUP MODEL** | **MGVLF** on DIOR-RSVG — [ZhanYang-nwpu/RSVG-pytorch](https://github.com/ZhanYang-nwpu/RSVG-pytorch), paper IEEE TGRS 2023 doi:10.1109/TGRS.2023.3250471. Alternative if weights appear: **GeoGround** [arXiv:2411.11904](https://arxiv.org/abs/2411.11904), [VisionXLab/GeoGround](https://github.com/VisionXLab/GeoGround). |
| **WHY** | Spec needs spatial evidence. GeoChat already in stack. MGVLF is a **specialist grounding** CNN-transformer if boxes from GeoChat are weak. |
| **COMPUTE REQUIREMENT** | GeoChat: as above. MGVLF: paper trained on **1× GTX 1080 Ti 11 GB**, 640×640, batch 8. |
| **DATASET** | GeoChat instruct + referring. Backup: **DIOR-RSVG**. Eval also **VRSBench** object refers. |
| **BENCHMARK** | VRSBench grounding Acc@IoU (their protocol); DIOR-RSVG if MGVLF trained. |
| **IMPLEMENTATION RISK** | **High:** GeoChat box parse must follow **their** coordinate convention. MGVLF **pretrained full checkpoint not verified**. GeoGround **weights unverified**. P1 if P0 caption already satisfies “one additional single-image capability”. |

---

## Capability: bi-temporal change detection (M4 map)

| | |
| --- | --- |
| **RECOMMENDED MODEL** | **BIT (Bi-temporal Image Transformer)** via **Open-CD** — weights `LEVIR-CD/bit_r18_256x256_40k_levircd.pth` on [likyoo/Open-CD_Model_Zoo](https://huggingface.co/likyoo/Open-CD_Model_Zoo) |
| **BACKUP MODEL** | Open-CD **ChangeFormer** MiT-B0/B1 LEVIR-CD zoo files. Avoid original [justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD) / [wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer) READMEs’ **non-commercial** clauses unless legal review says otherwise. |
| **WHY** | Pixel change map (spec). Open-CD **Apache-2.0** toolbox + zoo. BIT paper: Chen et al., arXiv:2103.00208. ResNet-18 is hackathon-sized. |
| **COMPUTE REQUIREMENT** | 256×256 RGB pairs (that zoo config). Inference: CPU or small GPU. Sliding-window for large GeoTIFFs (our tool). |
| **DATASET** | Weights: **LEVIR-CD**. Not SAR-native. |
| **BENCHMARK** | LEVIR-CD if we re-run Open-CD test; **not** a substitute for CDVQA. Domain gap on Sentinel/ISRO imagery is expected and must be disclosed. |
| **IMPLEMENTATION RISK** | Medium: OpenMMLab/Open-CD dependency stack; RGB-only vs multispectral (band selection tool). Original BIT repo license ≠ Open-CD zoo. |

---

## Capability: change VQA (M4 questions)

| | |
| --- | --- |
| **RECOMMENDED MODEL** | **CDVQA baseline** — [YZHJessica/CDVQA](https://github.com/YZHJessica/CDVQA), Yuan et al., IEEE TGRS 2022 doi:10.1109/TGRS.2022.3203314 |
| **BACKUP MODEL** | **VisTA** — [like413/VisTA](https://github.com/like413/VisTA), arXiv:2410.23828 (**CC BY-NC 4.0** README). |
| **WHY** | Matches evaluation dataset **CDVQA**. Apache-2.0 code. VisTA is stronger in its paper on CDVQA/CDQAG but NC + heavier train. |
| **COMPUTE REQUIREMENT** | CDVQA: older CNN fusion — likely **single GPU 8–12 GB** to train (confirm in their scripts). VisTA paper: **4× RTX 4090**, batch 64. |
| **DATASET** | CDVQA (~2,968 pairs, >122k QA — paper). VisTA also QAG-360K (test set contact-gated per README). |
| **BENCHMARK** | **CDVQA** official split accuracy. VisTA CDQAG only if data access granted. |
| **IMPLEMENTATION RISK** | **High:** CDVQA **pretrained weights not confirmed**. Must train. Until then, P0 uses **change-map statistics tool** for describe / increase-decrease **without** pretending it is CDVQA. |

---

## Capability: optical–SAR joint analysis (M5)

| | |
| --- | --- |
| **RECOMMENDED MODEL** | **(Learned)** BigEarthNet v2 ResNet-50 **S1+S2** (`resnet50-all-v0.2.0`) for scene-level fusion. **(Pixel P0 tools)** optical water index + SAR backscatter stats with explicit fusion rules. **(Pixel P1)** dual-encoder U-Net/DeepLab **trained by us** on **WHU-OPT-SAR** ([AmberHen/WHU-OPT-SAR-dataset](https://github.com/AmberHen/WHU-OPT-SAR-dataset); classes include city/village and water). |
| **BACKUP MODEL** | Train **unofficial MCANet reimplementation** [Ray010221/MCANet](https://github.com/Ray010221/MCANet) on WHU-OPT-SAR (not official paper code). Or [RanFeng2/PAD](https://github.com/RanFeng2/PAD) / [fy-sun/FDMF-Net](https://github.com/fy-sun/FDMF-Net) if licenses and **Baidu** weights can be verified. |
| **WHY** | Official **multimodal** BigEarthNet weights exist. Pixel SOTA checkpoints with clear OSI licenses and easy downloads were **not** verified. Tools + our trained U-Net stay honest. |
| **COMPUTE REQUIREMENT** | Scene CNN: small. P1 U-Net: single 8–12 GB GPU to train on 256/512 tiles. |
| **DATASET** | BigEarthNet S1+S2; WHU-OPT-SAR (GF-1 optical + GF-3 SAR, 5 m, land-use labels). |
| **BENCHMARK** | BigEarthNet multimodal split; WHU-OPT-SAR hold-out mIoU **if** we train. No fake mIoU. |
| **IMPLEMENTATION RISK** | **High** for pixel masks on arbitrary ISRO pairs (sensor mismatch). MCANet GitHub is **unofficial**. FDMF weights on Baidu. Co-registration policy can block fusion. |

---

## Shared checkpoint note (GeoChat)

GeoChat is **one weight file**, **three registry adapters** (`vqa`, `caption`, `grounding`). That is allowed: specialists remain separate **interfaces**. Perception for change maps and BigEarthNet land-cover remains **non-GeoChat**. This is **not** “generic VLM alone.”

---

## Rejected / deferred as primary (with reason)

| Model | Reason |
| --- | --- |
| Vanilla LLaVA-1.5 / GPT-4V as sole stack | Spec: not generic VLM alone; closed models also fail open-source eval needs |
| EarthVQA as default ship | Academic/commercial prohibition |
| Original BIT_CD / ChangeFormer GitHub as license basis | Non-commercial README; use Open-CD Apache zoo instead |
| Claiming GeoGround/VisTA/MGVLF “ready weights” | Weights not verified in this pass |

---

## Document control

| Version | Date | Notes |
| --- | --- | --- |
| 0.1 | 2026-09-04 | Initial matrix from public repos/cards. Re-verify licenses and VRAM at implementation. |
