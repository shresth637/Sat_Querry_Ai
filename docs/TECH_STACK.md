# SatQuery AI — Technical stack and architectural decisions

| Field | Value |
| --- | --- |
| Companion | `PROJECT_SPEC.md`, `docs/ARCHITECTURE.md`, `docs/MODEL_DECISION_MATRIX.md` |
| Status | Decisions only — no application code |
| Date | 2026-09-04 |

This document **resolves** the open decisions in `PROJECT_SPEC.md` §14 and `docs/ARCHITECTURE.md` §12 for a hackathon-quality, honest implementation. Model catalog details live in `docs/MODEL_DECISION_MATRIX.md`. Candidate research notes are in §6 here. **No invented model names, repos, papers, licenses, weights, or benchmark scores.** Where a license or weight file was not verified on the official card, that gap is stated.

---

## 1. Frontend — Streamlit

**Decision:** Streamlit for the initial hackathon GUI.

**Why it is acceptable**

- Covers spec GUI surfaces: upload (single and multi), query, preview, metadata, validation, results, confidence, evidence images, change maps, grounding overlays, execution trace, report download.
- Python-native: same process as the agent, rasters (`rasterio`/`rioxarray`), and PyTorch — no duplicate type layer for MVP.
- Fast iteration for ISRO/SAC demo.

**Constraints to design around (not blockers)**

- Large GeoTIFF uploads: configure Streamlit max upload size; stream to disk; never load entire cubes into browser.
- Long inference: `st.status` / spinner + job-style progress from the tracer; optional `st.fragment` later.
- Streamlit is not a multi-user production UI. Sufficient for hackathon and local evaluation.

**Not chosen now:** React/Vue SPA. Revisit only if a separate FastAPI service is introduced (P2).

---

## 2. Backend / API — no FastAPI for MVP

**Decision:** **In-process Python backend.** Streamlit calls `Agent.analyze(...)` directly. No FastAPI/uvicorn layer in P0/P1.

**Why FastAPI is not required yet**

- A second HTTP hop does not improve validation, registries, tracing, or specialist accuracy.
- Spec requires UI ↔ agent separation **as modules**, not necessarily as network services (`ARCHITECTURE.md` layers).
- FastAPI would duplicate session/artifact handling already needed in Streamlit.

**When FastAPI would materially help (P2)**

- Concurrent users, GPU worker pool, or a non-Python client.
- Then: Streamlit (or other UI) → FastAPI jobs → same agent package.

**API-shaped boundary (even without HTTP):** keep `satquery.api` as a thin façade (`create_session`, `analyze`, `build_report`) so a server can wrap it later without rewriting specialists.

---

## 3. Agent — custom Python router

**Comparison**

| Option | Satisfies spec §7? | Cost | Verdict |
| --- | --- | --- | --- |
| **Custom Python router** | Yes, if steps are explicit, typed, traced, and registry-driven | Low | **Recommended** |
| LangGraph | Yes, if graph nodes map 1:1 to spec steps | Extra dependency, harder unit tests | Defer |
| LLM-based structured routing | Partial: parse only; **must not** select specialists by vibes or replace models | Unreliable; GPU/API; fails “not generic LLM alone” if it owns perception | **Not** the controller |

**Decision:** Deterministic **custom Python controller** (`AgentController`) with injected ports from `ARCHITECTURE.md`. Observable steps are code paths + `TraceEvent`s, not a hidden LLM chain.

LangGraph is optional later if multi-step retries become complex. An LLM may be used **only** as an optional query parser (see §4), never as the sole VQA/change/optical–SAR model.

---

## 4. Query interpretation → `AgentPlan`

Natural language is converted to a **structured plan** before any specialist runs.

### 4.1 Method (simplest that is real)

**Primary:** rule + keyword/intent classifier over normalized query text and **slot inventory** (what the user actually uploaded).

**Optional (P1):** small structured-output parse (JSON schema) with a local or API LLM **constrained to fill `Intent` fields only**. If parse fails or conflicts with slots, **rules win**. The parser never invents raster metadata or confidence.

**Not used:** free-form “tool calling” where the LLM picks arbitrary Python.

### 4.2 `AgentPlan` (required fields)

| Field | Meaning |
| --- | --- |
| `task` | Canonical task id(s), e.g. `single_vqa`, `caption`, `grounding`, `change_detect`, `change_describe`, `change_vqa`, `optical_sar`, `mixed` |
| `required_inputs` | Slot ids that must be present (`optical`, `sar`, `t0`, `t1`) and input mode I1–I4 |
| `selected_models` | Registry IDs resolved from **config + capability + input mode**, not hardcoded in the UI |
| `selected_tools` | Registry IDs (inspect, preview, align-check, overlay, change-map color, report, optional indices) |
| `permitted_parameters` | Whitelist only: e.g. `max_side_px`, `change_prob_threshold` **from config**, `grounding_phrase` extracted from query, `answer_max_tokens`. Unknown keys ignored. |

Additional plan metadata (recommended): `intent_tags`, `warnings`, `blocked` (bool), `block_reason`.

### 4.3 Routing sketch (not canned answers)

| Query pattern | `task` | Required slots |
| --- | --- | --- |
| Land-cover / objects / “what is in this image” | `caption` + `single_vqa` (and land-cover specialist) | I1 or I2 |
| “Highlight / locate / water body referred” | `grounding` | I1 or I2 |
| “What changed / where” | `change_detect` + `change_describe` | I4 |
| Optical + SAR built-up / water | `optical_sar` | I3 |
| Built-up increased / decreased / unchanged | `change_detect` + `change_vqa` (and/or map-derived stats tool) | I4 |

If slots do not match `required_inputs`, validation **fails** and specialists are not run.

---

## 5. Remote-sensing adaptation (M1)

**Decision:** The production specialist that satisfies M1 is a **BigEarthNet (reBEN / v2) multi-label land-cover image classifier**, not a generic ImageNet/CLIP/VLM checkpoint.

**Recommended artifact:** official **ResNet-50 trained on BigEarthNet v2.0** using **Sentinel-1 + Sentinel-2** bands:

- Weights: Hugging Face `BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0`
- Loading path documented by the authors: `reben_publication.BigEarthNetv2_0_ImageClassifier.from_pretrained(...)` (requires their architecture package / `configilm` as stated on the model card)
- Dataset: [bigearth.net](https://bigearth.net/) — BigEarthNet licensed **Community Data License Agreement – Permissive, Version 1.0**
- Paper for v2/reBEN: Clasen et al., “reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis”, arXiv:2407.03653 / IGARSS 2025

**Sibling weights:** `BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0` is tagged **MIT** on Hugging Face. **Confirm the license tag on `resnet50-all-v0.2.0` at download time** (not assumed here).

**Why this slot (not “just use GeoChat”)**

- Spec forbids claiming a generic pretrained VLM as M1.
- This checkpoint **is** a visual model adapted on an approved open RS dataset and is used in **production paths**: land-cover evidence for single-image answers, and **optical–SAR complementary scene labels** (S1+S2 joint weights).
- Fine-tuning feasibility: ResNet-50 on BigEarthNet is realistic on a single 8–24 GB GPU; 7B VLM LoRA on BigEarthNet multi-label is not the minimum honest M1 path.

**Optional strengthening (P1):** continue training / calibrate on the official BigEarthNet split and record seed, config, and checkpoint hash. Using the official BigEarthNet-trained weights **already** meets “adapted using BigEarthNet”; additional fine-tuning is documentation-positive, not a substitute for wiring the model into `TaskPlan`.

**Must not:** ship M1 as an unused file while GeoChat answers land-cover from generic language priors only.

---

## 6. Model selection research (real candidates)

Full recommend/backup table: `docs/MODEL_DECISION_MATRIX.md`. Notes below are research facts and gaps.

### 6.1 Remote-sensing VQA

| Candidate | Repo | Paper | License (verified?) | Weights | Notes |
| --- | --- | --- | --- | --- | --- |
| **GeoChat** | [mbzuai-oryx/GeoChat](https://github.com/mbzuai-oryx/GeoChat) | [arXiv:2311.15826](https://arxiv.org/abs/2311.15826) (CVPR 2024) | Code: Apache 2.0 (`pyproject.toml`). Weights card: [MBZUAI/geochat-7B](https://huggingface.co/MBZUAI/geochat-7B) `apache-2.0`. **Additional LLaMA/Vicuna lineage terms may still apply** — review before redistribution. | Hugging Face `MBZUAI/geochat-7B` | RS-specific grounded LVLM: VQA, captioning, referring detection. Optical RS. ~7B; FP16 on the order of **~14 GB VRAM** reported by third-party listings; treat as **GPU-class**. Active project (~700+ GitHub stars). |
| **RS-LLaVA** | [BigData-KSU/RS-LLaVA](https://github.com/BigData-KSU/RS-LLaVA) | Bazi et al., *Remote Sensing* 16(9):1477, 2024, [doi:10.3390/rs16091477](https://www.mdpi.com/2072-4292/16/9/1477) | GitHub license **Other / NOASSERTION**. Weights: [BigData-KSU/RS-llava-v1.5-7b-LoRA](https://huggingface.co/BigData-KSU/RS-llava-v1.5-7b-LoRA). Base LLM: `Intel/neural-chat-7b-v3-3`. **Do not assume Apache.** | HF LoRA + Neural Chat base | Trained on RS-instructions including **RSVQA-LR** and caption sets (as stated by authors). |
| **RSVQA baseline** | [syvlo/RSVQA](https://github.com/syvlo/RSVQA) | Lobry et al., IEEE TGRS 2020, [doi:10.1109/TGRS.2020.2988782](https://doi.org/10.1109/tgrs.2020.2988782) | **GPL-3.0** | Training code; skip-thoughts dependency. Public **pretrained inference weights not confirmed** in README. | Dataset/task baseline for **RSVQA** eval. Fragile stack (skipthoughts). |
| **EarthVQA** | [Junjue-Wang/EarthVQA](https://github.com/Junjue-Wang/EarthVQA) | Wang et al., AAAI 2024 | Dataset: academic / commercial prohibited (project site). HF dataset card also **CC-BY-NC-ND-4.0** for related EarthVLSet. | [Kingdrone-Junjue/EarthVQA-pretrained](https://huggingface.co/Kingdrone-Junjue/EarthVQA-pretrained) | Strong specialist; **license too tight** as default ship. |

**Recommend:** GeoChat adapter for open RS VQA (capability `vqa`), with BigEarthNet classifier fused as land-cover evidence.  
**Backup:** RS-LLaVA after license review; RSVQA code for **eval protocol**, not as primary demo model.

### 6.2 Remote-sensing captioning

GeoChat and RS-LLaVA both state captioning as a first-class task. VRSBench evaluates captioning for GeoChat, LLaVA-1.5, MiniGPT-v2, Mini-Gemini ([lx709/VRSBench](https://github.com/lx709/VRSBench), [arXiv:2406.12384](https://arxiv.org/abs/2406.12384)); **those paper numbers are not SatQuery results**.

**Recommend:** GeoChat `caption` adapter. **Backup:** RS-LLaVA.

### 6.3 Text-guided grounding

| Candidate | Repo | Paper | License / weights |
| --- | --- | --- | --- |
| **GeoChat** | same as §6.1 | same | Referring object detection / grounded conversation (authors). |
| **GeoGround** | [VisionXLab/GeoGround](https://github.com/VisionXLab/GeoGround) / [zytx121/GeoGround](https://github.com/zytx121/GeoGround) | [arXiv:2411.11904](https://arxiv.org/abs/2411.11904) | **Public pretrained weights not verified** in this research pass. |
| **MGVLF / RSVG** | [ZhanYang-nwpu/RSVG-pytorch](https://github.com/ZhanYang-nwpu/RSVG-pytorch) | Yang et al., IEEE TGRS 2023, [doi:10.1109/TGRS.2023.3250471](https://doi.org/10.1109/TGRS.2023.3250471); arXiv:2210.12634 | DIOR-RSVG data on Google Drive. **End-to-end pretrained MGVLF weights not clearly published**; train from DETR+BERT inits. |
| **LPVA** | [like413/OPT-RSVG](https://github.com/like413/OPT-RSVG) | TGRS 2024 | Stronger reported DIOR-RSVG numbers in their README; **do not treat as our scores**. Weight availability not verified here. |

**Recommend:** GeoChat grounding adapter (boxes from referring queries). **Backup:** train MGVLF on DIOR-RSVG if GeoChat boxes are insufficient; GeoGround if weights appear.

### 6.4 Bi-temporal change detection

| Candidate | Repo | Paper | License | Weights |
| --- | --- | --- | --- | --- |
| **BIT via Open-CD** | Toolbox: [likyoo/open-cd](https://github.com/likyoo/open-cd) Apache-2.0. Method paper: Chen et al., “Remote Sensing Image Change Detection with Transformers”, [arXiv:2103.00208](https://arxiv.org/abs/2103.00208), IEEE TGRS. Official impl [justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD) **research/non-commercial** notice. | Open-CD Apache-2.0; **prefer Open-CD weights** [likyoo/Open-CD_Model_Zoo](https://huggingface.co/likyoo/Open-CD_Model_Zoo) Apache-2.0, including `LEVIR-CD/bit_r18_256x256_40k_levircd.pth` | RGB pairs, typically **256×256** crops in that config. Optical. GPU optional for ResNet-18. |
| **ChangeFormer via Open-CD** | Open-CD configs + Model Zoo `changeformer_mit-b0/b1_...levircd.pth`. Original [wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer) GitHub **MIT** metadata **and** README “non-commercial / research only” — **conflict; treat original as restricted**. | Prefer Open-CD Apache zoo. | Heavier than BIT. |

**Recommend:** Open-CD **BIT ResNet-18 LEVIR-CD** checkpoint. **Backup:** Open-CD ChangeFormer MiT-B0/B1 LEVIR weights.

### 6.5 Change VQA

| Candidate | Repo | Paper | License | Weights |
| --- | --- | --- | --- | --- |
| **CDVQA baseline** | [YZHJessica/CDVQA](https://github.com/YZHJessica/CDVQA) | Yuan et al., IEEE TGRS 2022, [doi:10.1109/TGRS.2022.3203314](https://doi.org/10.1109/TGRS.2022.3203314) | **Apache-2.0** | Dataset on GitHub. **Released pretrained demo weights not confirmed** in README — likely **train on CDVQA**. |
| **VisTA** | [like413/VisTA](https://github.com/like413/VisTA) | [arXiv:2410.23828](https://arxiv.org/abs/2410.23828) | Dataset/code notice **CC BY-NC 4.0** (README). | Train scripts; `MODEL.WEIGHTS` path — **public final checkpoint not verified**. Train: paper describes 4× RTX 4090, batch 64. |

**Recommend:** CDVQA baseline trained on the official CDVQA split (Apache). **Backup:** VisTA if NC license is acceptable. **P0 honesty path** if CDVQA training does not finish: **map-derived change description + closed-set increase/decrease** from change map ∩ optional built-up mask (tool), labeled as **not** CDVQA neural answers.

### 6.6 Optical–SAR analysis / fusion

| Candidate | Repo | Paper / data | License | Weights |
| --- | --- | --- | --- | --- |
| **BigEarthNet v2 ResNet-50 S1+S2** | HF `BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0` | reBEN / BigEarthNet v2 | Dataset CDLA-Permissive-1.0; **model license: verify card** (S2-only sibling MIT) | **Yes** (~95 MB class of checkpoint) | **Scene-level** joint land-cover, not pixel masks. |
| **WHU-OPT-SAR + unofficial MCANet** | Data: [AmberHen/WHU-OPT-SAR-dataset](https://github.com/AmberHen/WHU-OPT-SAR-dataset). Code: [Ray010221/MCANet](https://github.com/Ray010221/MCANet) (**author states unofficial reimplementation**). Paper: Li et al., *Int. J. Appl. Earth Obs. Geoinf.*, [doi via ScienceDirect S0303243421003457](https://www.sciencedirect.com/science/article/pii/S0303243421003457) | Unofficial code; **no official pretrained weights** | Pixel classes include city/village (built-up) and water — **only after we train**. |
| **PAD RGB-SAR seg** | [RanFeng2/PAD](https://github.com/RanFeng2/PAD) | Supports WHU-OPT-SAR in README | License not verified here | Training-oriented. |
| **FDMF-Net** | [fy-sun/FDMF-Net](https://github.com/fy-sun/FDMF-Net) | Chen et al., IEEE TGRS 2025, doi:10.1109/TGRS.2025.3622749 | Weights advertised on **Baidu Netdisk** — availability risk. |

**Recommend (P0):** S1+S2 BigEarthNet classifier for complementary scene labels **plus** documented **tools**: optical water index (e.g. NDWI when required bands exist) and SAR backscatter statistics, fused with **explicit rules** and provenance (index vs learned). **P1:** train a dual-encoder U-Net/DeepLab on WHU-OPT-SAR for pixel built-up/water — standard architecture, **our** checkpoint, no fake SOTA name.

---

## 7. Dataset mapping

| Dataset | Official entry | System component | Role |
| --- | --- | --- | --- |
| **BigEarthNet** | [bigearth.net](https://bigearth.net/), v2/reBEN | M1 land-cover classifier; optical–SAR **scene** fusion; `eval` suite | Train/adapt/eval multi-label land cover; S1+S2 joint input for M5 scene path |
| **VRSBench** | [vrsbench.github.io](https://vrsbench.github.io/), [lx709/VRSBench](https://github.com/lx709/VRSBench), HF `xiang709/VRSBench` | VQA, captioning, grounding eval (and optional LoRA) | Caption / grounding / VQA metrics on official splits; PNG/JPEG per spec §3.1 |
| **RSVQA** | [rsvqa.sylvainlobry.com](https://rsvqa.sylvainlobry.com/), [syvlo/RSVQA](https://github.com/syvlo/RSVQA) | VQA eval; RS-LLaVA training mix (authors) | Official VQA accuracy protocol |
| **CDVQA** | [YZHJessica/CDVQA](https://github.com/YZHJessica/CDVQA) | Change VQA model train/eval | Change questions (including increase/decrease style items as in that dataset) |
| **ISRO/SAC** | As provided by organizers | Full agent path | Official eval only; no placeholder scores |

Supporting (not in the four-name table but used by recommended weights): **LEVIR-CD** (Open-CD BIT), **WHU-OPT-SAR** (pixel optical–SAR P1), **DIOR-RSVG** (grounding backup).

---

## 8. Co-registration policy (never silent)

Alignment state is an enum on the pair: `user_declared_coregistered` | `grids_match` | `resampled_under_policy` | `unaligned` | `unknown`.

| Situation | Policy |
| --- | --- |
| User declares pair already co-registered | Record declaration. Still **inspect** CRS, transform, size, resolution. If grids already match → `grids_match`. If not → **warn**; do **not** silently treat as pixel-aligned. |
| Same CRS, same transform, same size | `grids_match`. Pixel specialists allowed. |
| Mismatched CRS | **Do not** run pixel change/fusion until resampled **or** user confirms geographic overlay only. Offer **explicit** `reproject_to_reference` tool (rasterio/GDAL). Status `resampled_under_policy` with residual/grid report. If user refuses → `unaligned`; **no change map / no pixel fusion**. |
| Same CRS, different dimensions or resolution | **Do not** assume alignment. Optional `resample_match` to reference slot with documented resampling (nearest for SAR labels, bilinear for optical reflectance — config). Always show before/after size and GSD. |
| Optional registration (feature matching / RPC) | **P2 / off by default.** If enabled, must report RMSE/inliers or **fail**. Never mark `grids_match` without a measured check. |
| Missing CRS on both, identical H×W | `unknown`. Allow **only** if user checks “treat as already pixel-aligned (no georeference)”. Trace records the assumption. Default is **block** geospatial map export. |

GUI shows this status next to validation. Change maps and fusion masks require `grids_match` or `resampled_under_policy` or explicit pixel-aligned override.

---

## 9. Confidence representation

`ConfidenceReport` is always present. **No default numeric fill.**

| Case | Representation |
| --- | --- |
| Model emits scores | `value` = documented aggregation of **those** scores (e.g. mean box score; change-map mean softmax of “change” class on predicted change pixels; land-cover sigmoid max). `method` = id e.g. `bit_change_mean_prob`. `signals` lists fields used. |
| No model scores (e.g. greedy VLM text) | `value = null`, `method = none`, `reason = no_model_native_score`. GUI shows **“Not available”**, not 0.xx. |
| Multiple outputs | `per_claim[]` each with own method. Optional `aggregate` **only if** every contributing `value` is non-null; formula in config (e.g. min, or validation-capped min). If any required claim has `null`, aggregate is `null` with `reason = incomplete_signals`. |
| Validation warning | `cap` applied to numeric values (config, e.g. max 0.7) and explained. Caps are **policy**, not fake model probabilities. |

Never display a hardcoded 0.85. Never average with invented 1.0 for tools that have no score.

---

## 10. Change description architecture

Pipeline (order fixed in plan):

1. **Change detection** → probability/logit map + binary map (threshold from config, recorded).
2. **Geometry tool** (not an LLM): connected components, area (px and m² **only if** transform exists), centroid, simple region labels (N/S/E/W thirds).
3. **Optional class evidence:** BigEarthNet scene labels on t0/t1 **or** P1 built-up mask; intersection with change map for “built-up change area”.
4. **Text:** template filled **only** with measured quantities (`ChangeDescriptionTool`). Example structure: changed fraction, component count, location bins, built-up area delta if masks exist.
5. **Change VQA** (when query is interrogative): CDVQA/VisTA on the pair; answers **must** attach the change map as evidence. If the neural model is not configured, answer **only** questions the stats tool can support (e.g. increase/decrease of changed/built-up area) and set `not_configured` for open-ended CDVQA.

LLM may **rephrase** the template for the HTML report **without adding facts**. If it adds facts, reject (schema: no new numbers).

---

## 11. Report format

**Decision:** **HTML first** (self-contained or HTML + sidecar images). **PDF optional (P2)** via print-to-PDF or a later WeasyPrint/Playwright step.

HTML includes: query, slot filenames, real `RasterMeta`, validation, plan IDs, answers, confidence block, embedded/linked evidence, change map, grounding, full trace.

---

## 12. Development environment

| Item | Recommendation |
| --- | --- |
| Python | **3.11** (3.10+ OK; avoid 3.13 until all wheels exist) |
| PyTorch | **2.4.x or 2.5.x** matching local CUDA; CPU wheel for laptop-only CNN path |
| CUDA | Official PyTorch CUDA 12.1/12.4 wheels **if** NVIDIA GPU present; do not vendor CUDA toolkit unless needed |
| CPU fallback | BIT, BigEarthNet ResNet-50, raster tools: **yes**. GeoChat 7B: **GPU required** or `not_configured` |
| Virtualenv | `venv` or **uv** project; `PYTHONPATH` = repo root |
| Requirements | `requirements.txt` (API/UI/raster/tests); `requirements-ml.txt` (torch, timm, open-cd extras); lock optional `uv.lock` / `requirements.lock` later |
| Raster | `rasterio`, GDAL wheels as needed, `pillow` for previews |
| Config | YAML (`config/default.yaml`) — model ids, paths, thresholds |
| Tests | `pytest` |

---

## 13. Deployment (hackathon)

**Practical:** one Windows/Linux workstation, `streamlit run`, models on local disk/SSD, `.env` for HF token if gated.

- Document `SATQUERY_DEVICE=cuda|cpu`.
- Git LFS or manual checkpoint download script (P1) — **do not** commit 7B weights.
- Optional: Gradio/Streamlit Cloud **not** assumed (GPU/size).
- Eval: CLI `python -m satquery.eval` on the same machine overnight.

---

## 14. Hardware (minimum practical)

| Mode | Estimate | Notes |
| --- | --- | --- |
| **Development** (UI, agent, validation, raster tests) | CPU, **16 GB RAM**, SSD | No GPU required |
| **Inference (CNN specialists only)** | **8 GB GPU** or strong CPU | BIT + ResNet-50 |
| **Inference (full demo with GeoChat)** | **16 GB VRAM** class (24 GB more comfortable); **32 GB system RAM** | 7B FP16; 4-bit may reduce VRAM but must be tested — do not claim it works until measured |
| **Fine-tune M1 ResNet-50 BigEarthNet** | **8–12 GB GPU**, many hours/days depending on split and augmentation | Feasible hackathon |
| **Train CDVQA baseline** | Single **8–12 GB** GPU typical for older CNN-RNN; confirm when implementing | |
| **LoRA on GeoChat / VRSBench** | **24 GB+** GPU recommended | P1/P2, not required for M1 |

---

## 15. MVP prioritization

| Component | Priority |
| --- | --- |
| Streamlit GUI (spec surfaces, honest empty states) | **P0** |
| Agent router, `AgentPlan`, trace, logging, YAML config | **P0** |
| Input validation + preprocess (GeoTIFF/TIFF, PNG/JPEG policy) | **P0** |
| Model + tool registries | **P0** |
| BigEarthNet ResNet-50 in production path (M1) | **P0** |
| Open-CD BIT change detection + change map + stats description | **P0** |
| Optical–SAR: S1+S2 classifier + index tools + co-reg policy | **P0** |
| GeoChat VQA + caption (if GPU) | **P0** if GPU; else **P1** with `not_configured` and CNN land-cover text only |
| GeoChat grounding | **P1** (M3: caption may satisfy minimum if grounding delayed) |
| HTML report download | **P0** (simple HTML) |
| Confidence module (null-safe) | **P0** |
| CDVQA trained adapter | **P1** |
| WHU-OPT-SAR pixel fusion | **P1** |
| Eval harness (BigEarthNet / RSVQA / VRSBench / CDVQA) | **P1** |
| FastAPI, PDF, LangGraph, automatic registration | **P2** |
| VisTA / GeoGround / ChangeFormer | **P2** backups |

---

## 16. Final recommended stack (summary)

```
Streamlit UI
  → Python façade (no FastAPI)
    → Custom AgentController (typed AgentPlan, tracer, confidence)
      → Tools: rasterio inspect/preview/resample-check, overlays, HTML report, NDWI/SAR stats
      → Models (config registry):
          M1 + optical-SAR scene: BIFOLD BigEarthNet v2 ResNet-50 (S1+S2)
          VQA + caption + grounding: GeoChat-7B (RS LVLM)
          Change detect: Open-CD BIT r18 LEVIR-CD
          Change VQA: CDVQA baseline (train) [P1]
          Pixel optical-SAR: our U-Net on WHU-OPT-SAR [P1]
```

Python 3.11, PyTorch 2.4/2.5, CUDA optional, venv/uv, YAML config, HTML reports, local Streamlit deploy.

---

## Document control

| Version | Date | Notes |
| --- | --- | --- |
| 0.1 | 2026-09-04 | Stack and policies. Implementation not started. |
