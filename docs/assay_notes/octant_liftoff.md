---
title: "Predicting PXR Induction - We have liftoff"
source: "https://openadmet.ghost.io/predicting-pxr-induction-we-have-liftoff/"
author:
  - "[[OpenADMET]]"
published: 2026-04-01
created: 2026-04-18
description: "Today is launch day for the PXR induction challenge!Below you will find more information about the challenge, following up on our announcement blog post. We will attempt to answer the questions we have received so far and provide more general information about challenge operations, rules, and infrastructure.Activity Dataset"
tags:
  - "clippings"
---
### Today is launch day for the PXR induction challenge!

Below you will find more information about the challenge, following up on our [announcement blog post](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/). We will attempt to answer the questions we have received so far and provide more general information about challenge operations, rules, and infrastructure.

## Activity Dataset - How the assays work

Building on the scientific background in our [announcement blog post](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/), we want to clarify how the PXR primary assay, counter assay, and single-concentration screening work at a high level.

### Primary assay

The assay used to generate this activity data is a cell-based reporter assay. Very simply, a cell line with the luciferase gene downstream from PXR’s DNA binding site is dosed with compound at a specific concentration. If the compound interacts with PXR, PXR will then bind to its cognate DNA binding site and induce expression of luciferase. After the incubation period, the cells are lysed, and the luciferase substrate is added. When the substrate and luciferase interact, the reaction luminesces, and the amount of light produced is directly proportional to the degree of PXR activity. This assay is used both for single concentration screening and for 8-point dose-response curves (DRC), which are fitted to the Hill equation, yielding summary statistics (EC50, Emax, and Hill slope).

### Counter assay

The corresponding counter screen assay that checks for false positives is essentially identical to the activity assay, but does not contain the PXR DNA binding site. Instead, the counter assay cell line has a steady baseline of luciferase expression from a different promoter, ensuring that any observed signal is independent of PXR binding. This means that when the cell line is incubated with a variable concentration of the compound, as in the primary assay, any light signal measured after the luciferase substrate is added is *not* due to PXR activity and is more likely a function of off-target transcriptional modification (or another kind of assay interference). **A compound that has both a high signal in the primary assay *and* the counter screen assay is likely to be a false positive.** A data analysis and fitting procedure similar to that used for the primary screen is used for the counter assay. As an additional reminder, while you are not evaluated on the counter screen data, we strongly recommend using it to inform your models for the prediction of the primary assay pEC50s.

### Single concentration screening

Single-dose screening was used to triage compounds from the larger Enamine library. The larger Enamine library contained roughly 11,000 compounds: 10,000 from the [Discovery Diversity 10 set](https://enamine.net/compound-libraries/diversity-libraries/dds-10240?ref=openadmet.ghost.io) and ~1,000 from the [FDA Approved Drugs set](https://enamine.net/compound-libraries/bioactive-libraries/fda-approved-drugs-collection?ref=openadmet.ghost.io). Each compound was run at single discrete concentrations: ~10, 30, and 100 µM. This screening is referred to as single-dose/single-concentration screening because the three separate concentrations tested are not used to construct a full dose-response curve and are mainly used to determine an appropriate working concentration range, along with corresponding hit rates and triage potential actives. Full characterization of actives is then conducted with DRCs.

## Activity Dataset - Structure of the dataset

We are very excited to release the datasets to you on [HuggingFace](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test?ref=openadmet.ghost.io)! However before you run off to make your first model, **please keep reading. Context is very important.**

When constructing the dataset for this challenge, we wanted to give participants as much data as possible for **training** while maintaining a rigorous **testing** setup. To this end, we have combined **as much PXR data as we have available** (all collected using the same assay, same workflow, same lab, etc.) in our training set. As this assay came from a multi-staged assay flow run across a variety of chemical matter, the resulting dataset has some complexities. We have prepared the Sankey diagram below, based on actual data as staged on Hugging Face following quality control (QC), to help visualize the assay flow.

![](https://storage.ghost.io/c/3f/51/3f515c22-43d0-4696-a664-031fddd40688/content/images/size/w2400/2026/04/pxr_dataset_sankey-2.png)

Figure 1: Sankey diagram showing flow between screening and assay steps used to construct the challenge dataset. Note that the counter-assay to test set flow indicates chemisimilar relationship rather than subsetting.

Initial direct-to-DRC experiments were conducted early in the program's development with other chemical matter, while the single-concentration-to-DRC promotion flow was followed for the larger-scale screen on the two Enamine decks (DDS10 and FAD). Hopefully, this helps contextualize data availability for different sections of the dataset.

Of the 11,000 compounds from the single-dose pilot screen, a few thousand compounds yielded enough activity to be promoted to a full 8-point concentration dose response curve (DRC). By fitting these DRCs, the EC50s for these few thousand compounds were estimated. As you might expect in a typical drug discovery program, we further down-selected 63 highly potent compounds (with EC50 ≤ 1 µM) with confirmed on-target activity and ordered at least 10 commercially available chemically similar compounds to each highly potent compound to begin studying structural activity relationships (SAR) in detail. All of these chemisimilar compounds were run with full DRCs and constitute the **test set.**

In addition, any other compounds that were run in the PXR assay with full DRC from previous experiments were added to the train set for this challenge. Whilst exploring the training dataset on HuggingFace, you may notice that the number of highly potent compounds varies slightly from the 63 reported above. Subsequent repeat measurements allowed for the refinement of EC50 values, resulting in only 46 highly potent compounds in the final dataset at the same threshold. Importantly, this does not change the chemisimilar compounds used to create the test set.

## Activity Dataset - The two stages

The challenge is split into two stages with distinct phases designed to simulate the iterative nature of drug discovery. In Phase 1, participants will submit predictions for the full test set, while the live leaderboard reflects performance on a subset known as "Analog Set 1." At the start of Phase 2, the ground-truth pEC50 values for Analog Set 1 will be unblinded and released, allowing you to incorporate this new data into your training pipeline. Your final performance will then be evaluated on "Analog Set 2," the remaining blinded portion of the test set, to determine the final standings.

## Structure Dataset

Crystal structures were obtained by the methodology detailed in our [announcement blog post](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/) and consist of fragments from the DSI-poised, Enamine Essential and UCSF libraries solved at the NSLS-II beamline. Following a rigorous quality review, we have **cut down the fragment set** from our first announcement to **78 structures** exhibiting the highest-quality density.

**IMPORTANT: We are currently finalizing additional crystal structures for molecules from the Enamine decks (in both train and test sets).** We want to get these in people’s hands as a “late-breaking” release so you have more awesome data on real potent compounds to test you on. **Do not be surprised if we extend the structure prediction track test set with these additional crystal structures early in the challenge.** Similarly, we wanted to get the challenge launched without delay, so decided not to wait for these structures. We will let you know as soon as we have a final decision. In the meantime, please go ahead and **submit as usual** to the structure prediction track. **If these come in we will have to reset the structure track scoring.** This will be early in the challenge, meaning any potential disruption should be relatively minimal.

## What we have changed from our previous challenges

Thank you to everyone who submitted feedback on our previous blind challenge, either on the Discord or through our feedback form. We’ve been working hard to address the issues raised and improve things for our upcoming PXR challenge. We will continue this [Kaizen](https://en.wikipedia.org/wiki/Kaizen?ref=openadmet.ghost.io) process across all our challenges to create an engaging and scientifically rigorous exercise that benefits the community.

### Challenge design

On the challenge design front, we have introduced a 2-part staged element for the activity prediction component. After Phase 1 ends, half of the test set will be unblinded, allowing participants to train their models on a larger training set. This is designed to make the blind challenge not only about measuring performance on a static reference dataset, but also test the ability of teams to **respond to new information** and incorporate that into their predictions, the quintessential problem of drug discovery as applied in practice. We are excited to see how this format goes and welcome feedback from participants.

We also noted feedback around the reporting of the use of proprietary data. Using proprietary data has proven to be a winning strategy across multiple challenges, reinforcing the OpenADMET (and many others) thesis that a lack of data is the key bottleneck in building performant ML models. However, to better track who is playing on which playing field, we have added a tick box to the submission form to indicate the use of proprietary data. Using proprietary data is still encouraged as it gives us a picture of **what is possible** rather than testing ML methods in isolation, but we ask that participants disclose this honestly. You can now also filter the leaderboard by proprietary data use.

### Infrastructure and scoring

We’ve made some big changes behind the scenes to our submission and scoring pipeline, which should significantly improve stability and performance. In short, we have re-engineered the backend on AWS to create something reproducible for use across our upcoming challenges. As with any new system, there may be some *teething issues*, and we appreciate your patience if we need to work through them live. We will open-source this infrastructure in due course for the community to use.

We have also taken on board your feedback regarding submission “feedback”. Once we’ve received your submission and completed scoring, an automated update will be posted to the *#pxr-challenge-submissions* channel on our [OpenADMET Challenges Discord](https://discord.gg/5hEzv2ME?ref=openadmet.ghost.io). No more repeatedly refreshing the leaderboard hoping your entry will show up! You’ll now get an update for every entry, and feedback if your submission failed for any reason. Similar to the leaderboard, the feedback will use your alias if you’ve entered an anonymous submission. We are hoping this significantly closes the loop on feedback.

To reduce the number of invalid or failed submissions, we’re also releasing validation scripts so you can test your entries before submitting. These can be found in our [PXR Challenge Tutorial](https://github.com/OpenADMET/PXR-Challenge-Tutorial?ref=openadmet.ghost.io) GitHub repo, and will catch common mistakes that can cause your entry to fail. Please make sure you check every entry with these scripts before submitting. We will dynamically update these with any additional common mistakes that arise throughout the challenge.

You’ll be able to submit one entry per day for each track. It’s possible that we’ve overlooked some mistakes that can cause the scoring to fail, so for the first week, we’ll be reducing this limit to one submission every four hours, so that any unexpected errors don’t prevent you from getting your entries in.

For the structure submission, we are learning from the experience of the Polaris-ASAP Antiviral challenge and encourage participants to submit whole PDB structures that we then evaluate primarily using a superposition-free method in our backend. This will reduce the burden on participants to get the pre-alignment of structures correct and reduce the associated noise in scoring submissions.

**Never hesitate to reach out to us on Discord with questions about infrastructure or scoring**. We will try to answer them as best we can.

## How to get started

Head over to our [Hugging Face space](https://huggingface.co/spaces/openadmet/pxr-challenge?ref=openadmet.ghost.io)! Submissions are now open: follow the instructions [here](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test?ref=openadmet.ghost.io) to download the training set.

We have also provided [dedicated tutorials](https://github.com/OpenADMET/PXR-Challenge-Tutorial/tree/main?ref=openadmet.ghost.io) for each track to guide you through downloading the data, training your models, and formatting your final submissions:

- **Activity prediction track:** A step-by-step guide to importing and analyzing pEC50 activity data and training an example LGBM model. This tutorial also explores auxiliary challenge data, such as single-concentration counter-screen results, that can be integrated into your training workflows.
- **Structure** **prediction track:** Instructions for importing and analyzing the test set compounds, along with a walkthrough on how to validate your submission using a set of pre-generated Boltz-2 protein-ligand structures.

Join us on the **#pxr-challenge** channel on the OpenADMET Discord to discuss your strategies or ask questions.

## How are submissions scored:

### Activity track

For the activity prediction track, the primary metric is RAE (Relative Absolute Error) of the pEC50 predictions. Other metrics (RAE, R2, Spearman’s ρ, and Kendall’s τ) will also be calculated. During Phase 1 (from April 1st to 25th May), there will be a live, interim leaderboard, showing scores on roughly half of the test data (Analog Set 1). During Phase 2 (from 26th May to 1st July), the pEC50 values for Analog Set 1 will be unblinded, and there will not be a live leaderboard. After the conclusion of Phase 2 (11:59 PM UTC on 1st July), the final leaderboard will be released.

### Structure track

For the structure prediction track, the primary metric is the lDDT-PLI (local Distance Difference Test for Protein-Ligand Interactions), developed for the CASP15 competition in 2022. lDDT-PLI is “superposition-free”; it uses local differences in atomic distances, so it does not require aligning predictions with a reference structure. This focus on local, ligand-protein interactions makes this test more robust to conformational changes. We also calculate the BiSyRMSD (Binding-Site Superposed, Symmetry-Corrected Pose Root-Mean-Square Deviation), also developed for CASP15. This superimposes the binding site and computes the symmetry-corrected ligand RMSD. This again focuses on the ligand's local environment. Throughout both Phase 1 and Phase 2, there will be an live, interim leaderboard, showing scores on roughly half of the test data. After Phase 2 concludes on July 1st, the final leaderboard will be released.

As scoring for structure prediction is nuanced, we will try to outline some of the complexities here. Scoring is performed using an [OpenStructure](https://openstructure.org/?ref=openadmet.ghost.io) pipeline (reference implementation from CASP). The pipeline computes scores **pairwise** for each ligand in the predicted and reference structures. This leads to an *NxM* matrix of scores where N is the number of ligands in the reference structure, and *M* is the number of ligands in the submitted structure **for which a chemical match can be obtained** (by graph isomorphism). As our PXR crystal system is a homodimer and many small fragment molecules can bind to the large, flexible PXR binding site multiple times, many of the reference complexes contain 2 or more ligands (\*N-\*dimensional). Of the available pairwise comparisons, the backend then selects your **best-scoring** submission by lDDT-PLI to be your score. This inherently captures some of the uncertainty in the binding fragment by allowing matching against any pose in the reference structure. For your submissions, **you should submit a PXR monomer** (not a homodimer) to make the scoring as seamless as possible.

While we have attempted to make the scoring as robust as possible, this pipeline leads to a common possible failure mode:

**No match to the reference ligand for a predicted structure ligand by graph isomorphism**: Essentially, OpenStructure cannot find the reference ligand in your structure. In this case, your submission will still be scored, but each missing structure will receive a penalty LDDT-PLI of 0 and a BiSyRMSD of 20. We have tested our pipeline with common structure prediction tools, and it should be easy to get a full match to the reference structures with off-the-shelf tools. We also provide a script to validate that your chemical structures match those in the backend. You will also get live feedback in the Discord channel about how many structures in your submission were matched.

### Both tracks

Entries should appear on the leaderboard within 2 hours, though there may be delays if we’re experiencing a high volume of entries. The leaderboards will include an option to split entries into those that use proprietary data and those that use only publicly available data. Consistent with the ExpansionRx Challenge, all metrics are reported as the mean and standard deviation, derived from 1000 bootstrap iterations (with replacement). This approach allows for the calculation of confidence intervals, enabling more robust statistical comparisons between submissions.

The submission scoring code is available in the [tutorial repo](https://github.com/OpenADMET/PXR-Challenge-Tutorial/tree/main?ref=openadmet.ghost.io)**.**

### What are the rules?

**Methodology and data**  
We welcome submissions of all kinds, including machine learning and physics-based approaches.

- **Additional assays:** We strongly encourage participants to leverage the full depth of the provided dataset. This includes not only the pEC50 values but also single-concentration primary screens, Emax data, and counter-assay results. **In our internal testing, this has proven beneficial for model performance.**
- **Sharing data across tracks:** Participants can also choose to use structural data and insights from the Structure Prediction track to inform their work on the Activity Prediction track, and vice versa.
- **External sources:** You are free to incorporate data from external sources (*e.g.,* public repositories) and employ pre-training approaches as you see fit.
- **Baseline and common public-model** **approaches:** We encourage participants to submit baselines and common public-model approaches, within reason.

**Submissions**

- **Submission frequency:** You may submit to the Hugging Face portal multiple times throughout the challenge, with the initial cap set at **once every four hours** for initial debugging and testing. We will likely reduce this to a **once-per-day rate limit**, at the OpenADMET team's final discretion.
- Only your **latest submission** will count toward the intermediate and final leaderboards.
- **Anonymous participation:** We welcome teams from industry and academia alike. We understand that some participants may have constraints on participating in public-facing challenges; therefore, we will allow teams to compete anonymously as a group using a pseudonym or *alias*. While a Hugging Face account is required to track submissions, only your *alias* will be visible to other participants until you choose to reveal your identity after the challenge concludes.
- **Methodology report:** In the spirit of open science, we strongly encourage participants to share the code used for their submissions. If this is not possible due to IP or other constraints, we require, **at a minimum**, a short written report describing the methodology used (template available [here](https://docs.google.com/document/d/1bttGiBQcLiSXFngmzUdEqVchzPhj-hcYLtYMszaOqP8/edit?usp=sharing&ref=openadmet.ghost.io)). Please note that this report must be provided by the challenge close date: **submissions without a corresponding methodology report or code will not be included in the final leaderboard.**

**Evaluation and leaderboard**

- **Scoring metrics:** Submissions will be evaluated using the metrics described in the section above.
- **Statistical rigor:** We will estimate errors for these metrics using bootstrapping and apply the criteria outlined in the [following paper](https://pubs.acs.org/doi/10.1021/acs.jcim.5c01609?ref=openadmet.ghost.io) to determine whether model performance is statistically distinct.
- **Live vs Final leaderboard:** During the challenge, the live leaderboard will reflect results based on a partial fraction of the test set (**validation split)**. The Final leaderboard will be released after the challenge closes, representing the evaluation against the full, blinded test set.
	- For the **activity prediction task**, the *Analogue Set 1* will be used for the live leaderboard, while the *Analogue Set 2* will be reserved for the final leaderboard.
		- For the **structure prediction task**, half of the test set will be used for the live leaderboard, with the remaining half held back for the final evaluation.

## Other FAQ

1. **Where can I find the validation scripts?**  
	These can be found in our [PXR Challenge Tutorial](https://github.com/OpenADMET/PXR-Challenge-Tutorial?ref=openadmet.ghost.io) GitHub repo.
2. **Where can I find my submission status?**  
	Check out the `#pxr-challenge-submissions` channel on our [OpenADMET Challenges Discord](https://discord.gg/5hEzv2ME?ref=openadmet.ghost.io).
3. **My entry hasn’t appeared on the leaderboard. What should I do?**  
	Please reconfirm your entry passes validation scripts, then check the `#pxr-challenge-submissions` Discord channel for feedback on your submission. If you don’t get any feedback within two hours, then contact us through the [OpenADMET Challenges Discord](https://discord.gg/5hEzv2ME?ref=openadmet.ghost.io).
4. **When do final entries have to be submitted by?**  
	All entries must be submitted before **11:59 PM UTC on July 1st**.
5. **My submission to the structure track was only partially scored. What should I do?**  
	Check the submission against the validation script carefully. The most common issue is incorrect bond orders on your ligand. Ensure that they match the SMILES in the challenge set.
6. **Will there be a summary paper like last time?**  
	We plan on writing a summary paper; this may be combined with the Expansion challenge paper to streamline the number of papers we need to submit over the coming year.
7. **I have a question for the experimentalists, where can I ask it?**  
	You can ask on [Discord](https://discord.gg/5hEzv2ME?ref=openadmet.ghost.io)!
8. **Can I compete in one or both tracks?**  
	Yes, you can compete in just a single track or both at your discretion.
9. **I want to participate with my colleagues, should we submit together or apart?**  
	We encourage teams to submit under one account or alias. This makes tracking performance and attributing it correctly easier.
10. **What should be in my submitted PDB for the structure track?**  
	Submit a PDB file containing PXR as a monomer along with your predicted ligand pose. Ensure the ligand’s connectivity and bond orders match the provided SMILES exactly; the scoring backend uses graph isomorphism, so any mismatch will result in a failure to score. You can check this using the validation script in the tutorial repo.
11. **What is the protein sequence used for the PXR construct?**  
	A FASTA file containing the sequence is available in the [tutorial repo](https://github.com/OpenADMET/PXR-Challenge-Tutorial?ref=openadmet.ghost.io).
12. **You have said elsewhere that you made changes to the training set, what were they?** See this document [here](https://docs.google.com/document/d/14-2EL4Zk8g3NNO7bi33gA3fHz_ef98bBigunJMH3Ui4/edit?usp=sharing&ref=openadmet.ghost.io)