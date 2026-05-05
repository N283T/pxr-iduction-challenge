# Track 2: フラグメント結合姿勢予測 — 発見指向サーベイ (rev2)

> **プロベナンス**: ChatGPT Deep Research レポート (2026-04-26 取得、prompt rev2:
> discovery-focused)。 rev1 (`docs/deepresearch/track2_fragment_pose_rev1.md`, PR #131) が
> AlphaFold3 / Boltz / DiffDock / Vina / gnina の比較に偏っていた反省から、
> 「AI/ML 研究者が見落としがちな手法」 を炙り出すよう FBDD コミュニティ・
> 結晶学・QM/MM・水分子ネットワーク・multi-state refinement 等を明示的に
> 掘らせるプロンプトで再実行。
>
> ## 一番の収穫: パラダイムシフトのフレーミング
>
> > "expert workflows don't SOLVE the pose problem. They CONSTRAIN it with
> > one or more of: density, prior bound analogues, hydration thermodynamics,
> > pose stability / occupancy heterogeneity."
>
> 我々はこれまで「Boltz をどう良くするか」 (sampling 増やす / model 選択 /
> refine) で考えていた。 これは **「単一 docker を改善」** のフレーム。
> 専門家コミュニティの発想は **「Boltz を信用しない、 制約を増やす」** で、
> **4 種の orthogonal 情報源** で pose 候補を絞り込む:
>
> 1. **密度** (PanDDA / qFit / XGen / GLR / in situ QMR) — 結晶 reference 由来。
>    我々の自前予測には適用不可 (density なし) だが、 reference 側の
>    multi-state 性は意識すべき。
> 2. **既知 analogues** (GLR-style template transfer / SEED2XR / MCS overlay)
>    — fragment が小さく scoring failure に sensitive なので、 同一 pocket の
>    既存リガンド情報が大きく効く。
> 3. **水ネットワーク** (FTMap / WaterKit / HydraMap / MixMD / WScore /
>    SZMAP / SILCS) — 大きく hydrophobic な pocket では水仮説が pose
>    決定に大きい。
> 4. **pose 安定性** (DUck / dynamic undocking / quasi-bound state) —
>    平衡 score ではなく「key 接触を破る仕事」 で安定 pose を検出。
>    docking score と orthogonal。
>
> ## このサーベイから導いた我々の優先順位
>
> | 手法 | 我々のセットでの想定効果 | 着手予定 |
> |---|---|---|
> | **GLR-style template transfer** | 我々は `structures/pxr_lbd/` に PXR holo 結晶 72 個を持っており、 既存リガンドからの pose 転写は最も即効性高い | 明日朝の cooldown スロット |
> | **FTMap hotspot rerank** | 1 target に 1 回計算で 184 全部に流用、 安価 | cooldown 待ち時間中に web server で実行 |
> | **DUck on ambiguous subset** | 5-model spread > 2Å の ~30 化合物にだけ stability 検証 | 後 (重いが効きそう) |
> | **MixMD** | 1 target deep dive で重いが、 fragment subset 限定なら検討余地 | queue 入り |
> | **multi-ligand Boltz** (1:1 ではなく n>=2 で予測) | PXR pocket は 2 fragment 同時収容可能、 2 サイトを自然に試させる | queue 入り (ユーザー発案) |
> | qFit-ligand / PanDDA / XGen / QMR | 密度依存、 我々の自前予測には不適用 | スキップ |
> | QM rescoring (PM7 / GFN2-xTB) | レポート自身が "fragment-specific の証拠は弱い" | 当面後回し |
>
> ## 関連リンク
>
> - rev1: `docs/deepresearch/track2_fragment_pose_rev1.md` (PR #131)
> - 提出仕様: `docs/track2/submission_spec.md` (PR #125)
> - Track 2 進捗: issue #129
> - 現在の LB position: rank 7 / LDDT-PLI 0.4655 (id=33, 2026-04-26 10:10 JST)

---

## What looks genuinely missing from an AI/ML-centered survey

For your specific regime—large malleable hydrophobic pocket, fragment-heavy ligands, apo-soak crystallographic references, and a contact-recovery metric—the biggest blind spots are not “better docking engines.” They are workflows that treat **electron density, conserved or displaceable waters, analogue transfer, and pose ambiguity** as first-class objects. In the fragment-screening community, it is routine to ask whether a pose is **supported by an event map, a water-network hypothesis, a hotspot map, or a stable quasi-bound interaction**, rather than whether it merely scores well in a single static docking model. That difference matters especially for apo-soak fragments, because the data often encode **partial occupancy, alternate subsites, and apo-like pocket geometry** that a one-pose/one-structure predictor tends to flatten away. citeturn3search0turn15search0turn17search1turn23search0turn32search0

A second recurring theme is that the strongest non-ML methods are often **series-aware** rather than de novo. If one fragment or analogue has ever been seen in the pocket, medicinal-chemistry and crystallography groups frequently transfer that local geometry forward using template alignment, guided replacement, or hotspot overlays before they trust a fresh de novo prediction. Fragments are small enough that these priors often dominate the outcome. citeturn14search0turn25search5turn24search2

A third theme is that the literature is notably thinner than many people assume for **fragment-specific quantum rescoring wins**. What I found with the highest confidence is that semiempirical QM, force-field-plus-density refinement, and in situ QM restraints help most as **geometry cleanup, local validation, and restraint generation**, not as magic first-pass pose generators for wet, ambiguous apo-soak fragments. citeturn10search1turn33search0turn36search1turn38search3turn37search0

## Surprise table

| Surprise | Method or workflow | Originating community | Primary citation | Open-source status | Why it is specifically useful for **fragment apo-soaks** | Why you should care alongside a Boltz-style baseline |
|---|---|---|---|---|---|---|
| 1 | **Sparse-density ligand identification** | Macromolecular crystallography | *Automated identification of crystallographic ligands using sparse-density representations*, DOI **10.1107/S1399004714008578** | I did not locate a maintained public package during this search | It was explicitly framed as suitable for **fragment screening** and for identifying what small ligand best explains a weak density cluster; that is unusually relevant when multiple tiny chemotypes could fit an event. citeturn16search0turn16search3 | Adds a **density-to-ligand identity** check that a structure-only pose generator does not provide. |
| 1 | **SEED / SEED2XR** | In silico FBDD and fragment crystallography | *Exhaustive docking of molecular fragments with electrostatic solvation*, DOI **10.1002/(SICI)1097-0134(19991001)37:1<88::AID-PROT9>3.0.CO;2-O**; *In silico fragment-based drug design with SEED*, DOI **10.1016/j.ejmech.2018.07.042** | **Open-source** | SEED was built for **fragment docking**, uses continuum desolvation, and later SEED2XR workflows reported X-ray hit rates in the **10–40%** range; a later assessment reported docking times of about **1–10 s per fragment** and target-dependent true-positive rates up to **27%** at the experimental hit-rate cutoff. citeturn27search5turn25search0turn25search1turn25search2turn25search3 | Gives you a very fast **fragment-first exhaustive prior** over apo pocket subsites, which is often better calibrated for small rigid ligands than general ligand methods. |
| 1 | **MixMD for displaceable versus conserved waters** | Physical-organic medicinal chemistry and MD mapping | *Predicting Displaceable Water Sites Using Mixed-Solvent Molecular Dynamics*, DOI **10.1021/acs.jcim.7b00268** | Academic method; not a polished turnkey product | It explicitly classifies water sites as **conserved, selectively displaced, or freely displaced**, and predicts **which probe functional groups** displace each site. The paper emphasizes apo structures and notes GPU-adapted runs can finish in about a day for one system. citeturn23search0 | Adds **functional-group-specific water competition** information instead of only protein–ligand geometry. |
| 1 | **DUck dynamic undocking** | Industrial SBDD / kinetic triage | *Dynamic undocking and the quasi-bound state as tools for drug discovery*, DOI **10.1038/nchem.2660** | Method public; original tooling was not presented as an easy OSS package | DUck ranks poses by the work needed to break a key native contact and reach a **quasi-bound state**. In the original prospective Hsp90 fragment screen the docking-plus-DUck combination reported a hit rate approaching **40%**; follow-on fragment work used dynamic undocking to discover a novel kinase hinge-binding fragment. citeturn32search0turn32search1turn31search0 | Gives you a **pose-stability** filter orthogonal to static geometric prediction. |
| 2 | **Guided Ligand Replacement** | Iterative crystallographic SBDD | *Ligand placement based on prior structures: the guided ligand-replacement method*, DOI **10.1107/S1399004713030071** | Available in PHENIX; free to academics, not permissive OSS | GLR uses one prior protein–ligand structure to transfer a related ligand into a new map. That is exactly how many fragment-to-analogue series are actually handled when density is weak or multiple copies exist. citeturn14search0turn14search1 | If you already have even one soaked fragment in the pocket, GLR-style transfer often beats treating each new ligand as de novo. |
| 2 | **WaterKit** | Explicit-solvent thermodynamics | *WaterKit: Thermodynamic Profiling of Protein Hydration Sites*, DOI **10.1021/acs.jctc.2c01087** | **Open-source** | It aims to reproduce crystallographic waters and GIST-like thermodynamics at far lower cost than full-site MD, and was designed to be compatible with HTVS-style pipelines. citeturn17search0turn17search1turn17search2 | Gives a **one-time apo hydration map** that can rerank many fragment poses cheaply. |
| 2 | **HydraMap v.2** | Medicinal-chemistry hydration mapping | *HydraMap v.2: Prediction of Hydration Sites and Desolvation Energy with Refined Statistical Potentials*, DOI **10.1021/acs.jcim.3c00408** | Executables and test data were released, but the licensing model is not as clear as typical OSS | Designed to compare hydration sites **before and after ligand binding**, identify bridging versus replaceable waters, and estimate desolvation energy with a much lighter footprint than explicit-solvent MD. citeturn21search0turn21search1turn21search2 | Pragmatic way to add **water-aware scoring** to a fragment-heavy campaign without full MD. |
| 2 | **SILCS FragMaps / SILCS-MC pose refinement** | Probe-mapping and MD-derived free-energy grids | *Reproducing Crystal Binding Modes of Ligand Functional Groups Using SILCS Simulations*, DOI **10.1021/ci100462t** | Original academic method yes; current production docking software is mostly commercial | SILCS converts competitive fragment-and-water sampling into **FragMaps/Grid Free Energies**. The original validation showed overlap with known ligand functional-group positions and useful scoring of crystallographic-like poses. citeturn28search0turn28search1turn28search3turn28search4 | Adds **functional-group probability voxels** tied to protein and solvent environment, not just one best pose. |
| 2 | **In situ Quantum Mechanical Restraints** | Crystallographic refinement with QM | *In situ ligand restraints from quantum-mechanical methods*, DOI **10.1107/S2059798323000025** | PHENIX feature; academic/free rather than true OSS | QMR optimizes ligand geometry **in situ** in the pocket before generating restraints, which is exactly the right regime when weak fragment density plus apo geometry makes generic ligand dictionaries misleading. It was benchmarked on **>2330 ligand instances** in **>1700** protein–ligand models. citeturn33search0turn33search4turn33search9 | Best viewed as a **local truthing tool** for density-supported poses, especially for low-occupancy or strained fragments. |
| 2 | **XGen** | Industrial crystallographic model building | *XGen: Real-Space Fitting of Complex Ligand Conformational Ensembles to X-ray Electron Density Maps*, DOI **10.1021/acs.jmedchem.0c01373** | I did not find a public maintained OSS release | XGen explicitly fits **occupancy-weighted ligand ensembles** and showed better density fit than deposited single/alternate conformers in tested structures. citeturn16search1turn16search2turn16search4turn16search5 | Useful when your fragment should be treated as **multi-modal** rather than “choose one orientation and commit.” |
| 3 | **qFit-ligand / qFit 3** | Ensemble crystallography | *qFit 3: Protein and ligand multiconformer modeling for X-ray crystallographic and single-particle cryo-EM density maps*, DOI **10.1002/pro.4001** | **Open-source (MIT)** | qFit is one of the clearest answers to the fragment problem “several orientations fit almost equally well.” It automates parsimonious multiconformer models for ligands and occupancies. citeturn15search0turn15search1 | Strongest open solution I found for **ambiguity-aware ligand modeling** from density. |
| 3 | **FTMap plus user-selected probes / pharmacophore expansion** | Hot-spot mapping | *FTMAP: extended protein mapping with user-selected probe molecules*; *Expanding FTMap for Fragment-Based Identification of Pharmacophore Regions in Ligand Binding Sites*, DOI **10.1021/acs.jcim.3c01969** | Free web server; not standard OSS | FTMap identifies consensus hot spots where many probes cluster, and its newer pharmacophore expansion gives more chemically specific subregions within those hot spots. citeturn24search2turn24search7turn24search4 | Excellent for turning a floppy apo pocket into a **small set of anchor hypotheses** that fragment poses can be forced to respect. |
| 3 | **NMR CSP-guided fragment docking** | NMR FBDD | *Fragment docking supported by NMR shift perturbations*, DOI **10.1186/1758-2946-6-S1-P18**; *An NMR-based scoring function improves the accuracy of binding pose predictions by docking by two orders of magnitude* | Workflow-level, not one standard maintained package | CSPs are especially useful for fragments because docking scores are often nearly degenerate. NMR-derived restraints can physically rule out wrong orientations even when RMSD-like differences are small. citeturn29search0turn29search2turn29search6 | Supplies **experimental restraints** in exactly the low-specificity regime where fragments are hardest to rank computationally. |
| 3 | **GemSpot / GOLEM** | Cryo-EM ligand model building | *GemSpot: A Pipeline for Robust Modeling of Ligands into Cryo-EM Maps*, DOI **10.1016/j.str.2020.04.018**; *GOLEM: Automated and Robust Cryo-EM-Guided Ligand Docking with Explicit Water Molecules* | GemSpot is mixed/proprietary; GOLEM is a **free** VMD plugin | Both are adjacent-field methods, but they matter because they combine **map agreement, local QM-style cleanup, and explicit waters** instead of asking docking alone to do everything. GOLEM’s explicit-water handling is particularly relevant to weak small ligands in fuzzy maps. citeturn30search1turn30search2turn30search7 | These are examples of **experiment-guided pose building** that AI/ML pose papers often ignore. |
| 4 | **AFITT in PHENIX** | Industrial crystallographic refinement | *Improved ligand geometries in crystallographic refinement using AFITT in PHENIX*, DOI **10.1107/S2059798316012225**; original AFITT paper DOI **10.1107/S0907444906016076** | Mixed/commercial | AFITT uses an all-atom MM force field during refinement and improved ligand geometry without harming density fit; it handles alternate conformations and covalent ligands. citeturn10search0turn10search1turn11search1turn11search6 | Not glamorous, but it is a practical way to kill **bad fragment strain** after initial placement. |
| 5 | **PanDDA + DIMPLE + XChemExplorer** | High-throughput fragment crystallography around entity["organization","Structural Genomics Consortium","oxford, uk"] and entity["organization","University of Oxford","oxford, uk"] | *A multi-crystal method for extracting obscured crystallographic states…*, DOI **10.1038/ncomms15123**; *The XChemExplorer graphical workflow tool…*, DOI **10.1107/S2059798316020234** | Open academic pipeline components, though installation details vary | For apo-soaked fragments this is the reference workflow: DIMPLE for rapid difference maps, PanDDA for event maps and background correction, XCE for project-scale triage. PanDDA was built precisely for **weak, low-occupancy fragment events** and showed substantially more hits than manual inspection. citeturn3search0turn3search2turn5search0turn5search1turn6search6 | This is the most important reminder that, if density exists, **pose prediction should become density interpretation**. |
| 5 | **JAWS / WaterMap / WScore / SZMAP** | Water-aware industrial SBDD | JAWS DOI **10.1021/jp9047456**; WScore DOI **10.1021/acs.jmedchem.6b00131**; SZMAP DOI **10.1021/ci500746d**; WaterMap application review DOI **10.2174/1568026617666170414141452** | Mostly commercial / mixed | These methods explicitly model whether waters should be kept, displaced, or bridged, and WScore directly incorporates explicit waters into docking. SZMAP specifically correlated with water conservation or displacement in crystal structures. citeturn18search5turn20search0turn22search0turn19search0turn20search1 | In a big hydrophobic pocket, **water hypotheses often explain fragment orientation better than generic scoring does**. |

## What the table says when you step back

The recurring pattern is simple: the “obvious” expert workflows are those that **constrain** the pose problem rather than “solve” it from scratch. They constrain it with one or more of four information sources: **density**, **prior bound analogues**, **hydration thermodynamics**, and **pose stability / occupancy heterogeneity**. That is why the most useful ideas here are not replacements for a Boltz-style baseline but **orthogonal add-ons**. citeturn14search0turn17search1turn23search0turn32search0turn15search0

It is also striking how much of the strong evidence comes from practitioner ecosystems rather than benchmark papers. The clearest examples are the crystallographic fragment-screening pipelines around PanDDA and XCE, the fragment-docking-to-X-ray workflows around SEED, the quasi-bound stability filter from the DUck collaboration that included entity["company","Vernalis","cambridge, uk"], and company-authored density/hydration methods such as AFITT, WScore, SZMAP, and GemSpot. Those are the places where people were trying to make decisions on weak fragment data, not to top a leader board. citeturn25search0turn25search3turn32search0turn10search1turn20search0turn22search0turn30search2

## Three concrete add-ons to try this week

### Apo hydration and hotspot reranking

Run a **one-time apo pocket characterization** with FTMap plus either WaterKit or HydraMap v.2, then rerank each Boltz-derived pose by three simple quantities: hotspot overlap, clash with predicted conserved waters, and plausible displacement of unstable waters. FTMap gives you consensus hot spots and sub-pharmacophore regions; WaterKit and HydraMap add the solvent-thermodynamic view of which hydration sites are expensive or cheap to displace. MixMD is the slower, higher-physics variant if you want a single target-level deep dive rather than a quick sweep. citeturn24search2turn24search7turn17search1turn21search0turn23search0

What this adds that a standard structure-only pose generator does not add by itself is **explicit pocket prior information**: which subregions are truly anchoring hot spots, which waters should probably remain, and which tiny hydrophobic or hydrogen-bond vectors are worth occupying in an **apo-shaped** site. That is unusually important for your problem because fragment apo-soaks often preserve apo geometry and unusual water networks rather than collapsing into a single holo-like optimum. citeturn3search0turn17search1turn22search0turn23search0

This is the most deployment-friendly add-on under your budget because the mapping is done **once per target**, not once per ligand. WaterKit was explicitly positioned as suitable for integration into high-throughput pipelines, and HydraMap is much cheaper than full explicit-solvent thermodynamics. citeturn17search1turn21search0

### Template transfer before de novo ranking

If you have *any* prior soaked fragment structures for the same pocket, use them as **template priors**. In practice, that means generating analogue-aligned poses by GLR-style graph matching, MCS overlay, or a shape/field overlay workflow, then taking the union of those poses with Boltz poses and reranking. The fragment-docking literature is quite direct that fragments are especially sensitive to scoring failure, and template/binding-mode information improves fragment pose selection. citeturn14search0turn25search5

What this adds is information that de novo predictors usually do not exploit well: **local experimental geometry from related ligands in the same pocket**. GLR was built exactly for this use case in iterative SBDD, where the question is often not “where can this ligand bind?” but “which of the already-observed local binding modes does this analogue most plausibly inherit?” citeturn14search0turn14search1

If I had to choose a single add-on for a fragment-heavy, promiscuous, apo-soak set with existing fragment structures, this would probably be the first one I would try. It is cheap, series-aware, and most aligned with how medicinal chemists actually propagate fragment poses. citeturn14search0turn25search5

### Stability and strain triage on the ambiguous subset

Do **not** spend extra physics on all 200 ligands. Spend it only on the ambiguous ones. A practical stack is: Boltz top poses, then DUck-style quasi-bound pulling on a key native contact for the top pose family per ligand, plus a fast semiempirical or force-field cleanup step such as PM7-based local optimization or fragment-based semiempirical interaction-energy validation. If density exists for a subset, finish those with QMR or qFit/XGen rather than pretending the pose is single-state. citeturn32search0turn38search3turn36search1turn33search0turn15search0turn16search2

What this adds is **orthogonal pose information**: quasi-bound resilience and local strain. DUck was developed exactly because equilibrium affinity surrogates and docking scores often do not tell you whether the contact pattern is structurally robust. The newer semiempirical papers I found are promising for local cleanup and reduced-model validation, but not yet persuasive as universal fragment-pose winners. citeturn32search0turn36search1turn38search3turn37search0

Operationally, this means triaging perhaps the 20–30% most ambiguous ligands rather than the full library. That is much more realistic inside your 24 h budget than trying to run a heavy rescoring stack on every candidate. citeturn32search0turn23search0

## Honest dead ends and weak spots

The literature on **QM and semiempirical rescoring** is real, but the fragment-specific evidence is much weaker than the marketing aura around it. The strongest recent papers I found either focus on **general docking sets** with PM7 geometry cleanup, **binding-affinity scoring** rather than pose discrimination, or **QM-informed reduced models** for validating interaction energies. None of these gives a convincing public answer to “PM7/GFN2-xTB/ANI/OrbNet reliably wins fragment pose prediction in flexible apo-like nuclear-receptor pockets.” For this problem class, QM currently looks more like a **cleanup/validation layer** than a primary locator. citeturn38search3turn36search1turn37search0

A second dead end is very old standalone density fitters such as **X-LIGAND**. They remain historically important, and some of their ideas still survive, but I did not find evidence in this search that they remain active, fragment-oriented practitioner defaults compared with the later ecosystem of LigandFit, AFITT, rhofit, XGen, PanDDA, and qFit-style multistate fitting. citeturn11search8turn13search0

A third weak spot is **3D-RISM-style hydration analysis** for this application. It is scientifically respectable and there are modern apo-site studies, but compared with WaterMap/SZMAP/GCMC/WaterKit/HydraMap/MixMD I found much less fragment-pose-specific practitioner evidence and much less sign of routine adoption in medicinal-chemistry workflows. citeturn21search3turn21search2turn17search1turn23search0

The sparse-density ligand-identification paper is clever and directly relevant to fragment screening, but during this search I did **not** find signs of a maintained, widely reused software ecosystem around it. I would treat it as an idea worth borrowing from rather than a mature workflow you can just drop in. citeturn16search0turn16search3

## Data sources and practitioner communities that are actually useful

If you want reusable public data most aligned with your problem, the strongest source I found remains the high-throughput crystallographic fragment-screening ecosystem built around PanDDA, XCE, and later large-scale screens. These datasets are valuable not just because they contain hits, but because they encode **weak events, alternative sites, and pocket plasticity under soaking conditions**. That is exactly what standard docking benchmarks usually wash out. citeturn3search0turn5search0turn2search1

For fragment-specific docking outside density, the best public “industrial-ish” evidence I found came from the SEED ecosystem, especially bromodomain case studies and the later multi-target assessment with collaborators at entity["organization","University of Zurich","zurich, ch"] and industry. Those papers are unusually frank about when fragment docking helps and where its limits are. citeturn25search3turn25search2turn27search5

For higher-quality physics scoring, the new PL-REX dataset from the SQM2.20 work is relevant, but it is mainly an **affinity/scoring** resource, not a fragment apo-soak pose benchmark. I would mine it for rescoring experiments, not as a direct answer to your benchmark design. citeturn37search0

The frustrating part is that many industrial workflows are still described more clearly in **methods papers and vendor-authored validation articles** than in fully reproducible open benchmark suites. The best-documented examples I found were from Bristol-Myers Squibb on GLR, Merck on XGen, AstraZeneca on SZMAP, Schrödinger on WScore/WaterMap/GemSpot, OpenEye-affiliated AFITT work, and the DUck collaboration with Vernalis. citeturn14search0turn16search4turn22search0turn20search0turn10search1turn32search0turn30search2

## Research questions still open

The literature still does **not** have a clean answer to how fragment pose prediction should be scored when the biologically relevant unit is an **occupancy-weighted pose family** rather than a single rigid body. PanDDA, qFit, and XGen all point toward multi-state modeling, while newer evaluation work argues that interaction recovery can matter more than RMSD. But there is no broadly accepted benchmark designed around exactly that combination. citeturn3search0turn15search0turn16search2turn31search5

It is also still unclear how to disentangle the three main causes of error in your regime: **pocket malleability, water-network uncertainty, and fragment pose ambiguity**. Water-aware methods can explain some failures, hotspot maps explain some others, and template transfer explains still others, but there are few head-to-head public studies on large flexible pockets with apo-soak references. citeturn17search1turn21search0turn23search0turn24search2turn14search0

Another open problem is **when analogue transfer beats de novo prediction**. GLR and template-informed fragment docking clearly help, and medicinal chemists rely on them heavily, but I did not find a modern public benchmark that cleanly measures the crossover point as scaffold similarity decreases in a flexible pocket. citeturn14search0turn25search5

The published literature also has not settled which **cheap physics add-on** is the best buy under modest compute for fragments: hotspot mapping, explicit-water mapping, quasi-bound stability, semiempirical cleanup, or some committee of all of them. These methods are almost always validated separately, which is not how a practitioner would really use them. citeturn24search2turn17search1turn32search0turn36search1

Finally, there is still no widely used public benchmark curated specifically for **apo-soak fragment pose prediction in promiscuous pockets** with a contact-centric success criterion. The raw ingredients exist in crystallographic fragment-screening datasets, but the benchmark packaging has not caught up to the problem. citeturn3search0turn5search0turn2search1turn31search5
