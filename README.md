# 🧠 PhD Scripts Repository

<div align="center">

![Made with Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![ImageJ](https://img.shields.io/badge/ImageJ-Macro-00BFFF?style=flat-square)
![MATLAB](https://img.shields.io/badge/MATLAB-R2022b-orange?style=flat-square&logo=mathworks&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)

</div>

---

## Repository Structure

```
PhD-Scripts/
│
├──  Histo/                 ← Histology image analysis
├──  InVivo_Multi/          ← Multi-channel electrophysiology
├──  InVivo_Single/         ← Unitary spike analysis
└──  MOCAP/                 ← Motion capture & behavior
```

---

## Histo — Histological Image Analysis

> Neuroanatomical mapping of mouse central nervous system sections

**Tools:** `ImageJ` · `Fiji` · `Python`

Contains scripts for histological image processing:

- **Image handling & preprocessing** with ImageJ macros — batch processing, channel splitting, rotation, format handling
- **Cell counting & fluorescence quantification** across defined anatomical structures
- **Anatomical mapping** — region-of-interest (ROI) delineation, heatmaps

---

## In Vivo Multi — Multi-Channel Electrophysiology

> Extracellular recordings from 16-channel arrays implanted in mouse motor cortex

**Tools:** `Python` · `MATLAB` · `Kilosort` / `Phy`

Pipeline for handling multi-unit activity data from chronic implants:

- **Raw data import & preprocessing** — format handling, filtering, artefact rejection, optotagging
- **Spike sorting** using spikeinterface and kilosort on 16-channel silicon probes
- **Manual curation** workflows compatible with Phy/Kilosort outputs
- **Single-unit isolation** — waveform alignment, ISI analysis, cluster quality metrics
- **Population-level analyses** — PSTHs, cross-correlograms, firing rate dynamics

```
System  : RHD
Channels : 16
Region   : Mouse Primary Motor Cortex (M1) and Primary Sensory Cortex (S1)
Paradigm : In vivo, freely moving
```

---

## In Vivo Single — Unitary Spike Analysis

> Spike train analysis on isolated single units using glass pipettes

**Tools:** `Python` 

Analysis for characterizing the electrophysiological properties of individual neurons:

- **Spike train statistics** — mean firing rate, inter-spike interval (ISI) distributions
- **Peri-stimulus time histograms (PSTHs)** aligned to stimulation events
- **Raster plot generation**

```
System  : Spike2
Region   : Mouse Primary Motor Cortex (M1) and Primary Sensory Cortex (S1)
```

---

## MOCAP — Motion Capture & Behavioral Analysis

> Kinematic analysis of mouse locomotion using Vicon Nexus

**Tools:** `Python` · `Vicon Nexus SDK` 

Scripts for processing 3D motion capture data of rodent behavior:

- **Data import** from Vicon Nexus (.c3d / .csv exports)
- **Marker trajectory processing** — gap filling, filtering, coordinate transformations
- **Limb kinematics** — joint angles, step cycles, stride parameters
- **Gait analysis** — stance/swing phase detection, inter-limb coordination, symmetry indices
- **Visualization** of 3D trajectories and kinematic waveforms

```
System  : Vicon Nexus
Subject : Mouse (quadruped locomotion)
Output  : Joint angles, gait metrics, behavioral epochs
```
