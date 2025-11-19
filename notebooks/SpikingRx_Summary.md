# 1. SpikingRx Summary and Introduction

## 1.1 Research Background and Motivation

In the 5G NR downlink receiver (UE Receiver), the traditional physical-layer signal processing chain includes:

- FFT  
- Channel Estimation (LS/LMMSE)  
- Equalization (ZF/LMMSE Equalizer)  
- Soft Demodulation (Soft Demapper → LLR)  
- LDPC Decoding  

Within this chain, **“Equalizer + Soft Demodulation”** is the most complex part and the most sensitive to channel impairment, requiring highly specialized signal-processing algorithms.

**SpikingRx** is a new **SNN-based receiver** proposed in 2025.  
Its main idea is to use a Spiking Neural Network (SNN) to replace three blocks in the chain:

- Channel Estimation  
- Equalization  
- Soft Demapper (LLR generation)  

It uses a **SEW-ResNet spiking architecture**, utilizing **LIF neurons** and **surrogate gradients** to directly learn the mapping from the OFDM resource grid to bit-wise LLRs (which are compatible with LDPC decoders).

Therefore, the key features of SpikingRx include:

✔ No explicit channel estimation  
✔ No traditional equalizer  
✔ Directly learns soft bits (LLR)  
✔ Much lower computational energy than ANN receivers  
✔ Fully compatible with standard 5G NR LDPC decoders  


## 1.2 Position of SpikingRx in This Project

The purpose of this project is **not** to implement the entire 5G receiver.

Instead, the goal is to:

- Implement the SpikingRx SNN receiver described in the paper  
- Integrate it with real OFDM downlink data from **OAI (OpenAirInterface)**  
- Validate the inference capability of SpikingRx under realistic 5G conditions  

The **complete RX chain** in this project is:

1. OAI gNB/UE generates real PDSCH I/Q (after FFT → resource grid)  
2. Export full-grid dump to complex16 binary file  
3. Convert dump into **32×32×T** spiking tensor  
4. Feed into SpikingRx SNN Receiver (faithful implementation from the paper)  
5. Output LLR (bit0, bit1)  
6. Visualize spike activity (before vs after training)  

And importantly:

This system **does NOT perform LDPC decoding**, because SpikingRx’s task is **soft demodulation (LLR output)**.  
LDPC decoding belongs to the next module in the PHY chain.


## 1.3 Which Parts of the Traditional Receiver Does SpikingRx Replace?

Traditional 5G NR receiver:

```yaml
FFT
→ Channel Estimation
→ Equalizer
→ Soft Demapper (LLR)
→ LDPC Decoder
```


SpikingRx receiver:

```yaml
FFT
→ SpikingRx (direct LLR output)
→ LDPC Decoder
```

**Replaced blocks:**

- Channel Estimation (DMRS-based LS/LMMSE)  
- Equalization (ZF/LMMSE)  
- Soft Demodulation (LLR computation)  

Meaning:

**SpikingRx replaces the entire “Equalizer + Soft Demapper” block with an SNN.**  


## 1.4 Main Tasks of This Project

This project achieves SpikingRx–OAI integration through the following steps:

- Modify OAI (`nr-softmodem`) to export FFT-processed full-grid dumps (complex16)  
- Build Python loader to convert 13×1272 SC grid into 32×32 spiking tensor (1×T×2×32×32)  
- Fully implement the SpikingRx SNN architecture  
  (StemConv + 6×SEW blocks + ANN Readout)  
- Use synthetic QPSK+OFDM dataset for training  
  (because OAI cannot provide bit-level labels)  
- Run inference on OAI dumps → produce LLR + spike rate  
- Visualize spike activity (before vs after training)  
- Build complete workflow, documentation, and directory structure  


## 1.5 Contributions of This Project

This project successfully achieves something the original paper **did not**:

✔ **SpikingRx + OAI integration (real-world data)**  
  - The paper only used simulation data.  
  - This project feeds real OAI FFT output into the SNN receiver.  

✔ **Standardized 32×32 full-grid mapping**  
  - A universal compression/mapping to convert any OAI grid into a fixed 32×32 format.  

✔ **Complete spike-activity visualization**  
  including:  
  - Spike rate per SEW block  
  - Timestep GIF  
  - Before/after training comparison  

✔ **Reproduction of the full SpikingRx architecture**  
  - Triangular surrogate gradient  
  - SEW-ResNet blocks  
  - LIF neurons  
  - Rate pooling  
  - ANN LLR Readout  

# 2. System Architecture and File Mapping

The purpose of this chapter is:

- To present the entire system (OAI → Loader → SpikingRx → LLR → Visualization) in a block-diagram style  
- To explain the role of each block and how it corresponds to the original SpikingRx paper  
- To describe how each module in this project maps to the components described in the paper  


## 2.1 Overall System Architecture

Below is the technical flow diagram of the entire project system:

(Place your diagram here — same location as your notes)

[Full architecture diagram: OAI → Python Loader → SpikingRx SNN → LLR → Visualization]


## 2.2 Explanation of Each Block (Purpose and Relation to the Paper)

Below is the detailed explanation of each system block.


### (1) OAI gNB/UE (Full-grid FFT Dump)

**Function**

- Executes 5G NR downlink OFDM receive chain  
- After FFT, obtains the entire resource grid  
- You modified OAI to export this grid as binary data  

**Input**  
- Downlink PDSCH from rfsim mode (simulated or real RF)

**Output**  

```bash
/tmp/spx_fullgrid_f{frame}_s{slot}.bin
```

Format: `complex16`, containing **13 OFDM symbols × 1272 subcarriers**

**Relation to the Paper**  
Corresponds to **Figure 1: “OFDM Receiver Front-End”** in the SpikingRx paper.  
The paper uses simulation-generated grids; this project uses real OAI outputs.


---

### (2) Python Loader (`load_oai_fullgrid`)

**Function**

Converts OAI dump format into the spiking feature grid required by SpikingRx.

**Input**

- `complex16` full-grid dump from OAI

**Processing Steps**

- Parse binary file  
- Reconstruct resource grid  
- Normalize  
- Compress/mapping into **32×32**  
- Expand into **T timesteps** (required for LIF temporal dynamics)

**Output**

```yaml
(1, T, 2, 32, 32) Tensor
```

**Relation to the Paper**

Corresponds to:

**Input preprocessing module** in SpikingRx.  
In the paper, the input grid is simulation-based; in this project, OAI’s real grid is converted into the same consistent format.

---

### (3) SpikingRx SNN Receiver (Core)

**Input**

```csharp
(1, T, 2, 32, 32)
```

Then the forward flow is:

```yaml
↓ StemConv (2 → 16)
(1, T, 16, 32, 32)

↓ SEW1
(1, T, 16, 32, 32)

↓ SEW2
(1, T, 32, 32, 32)

↓ SEW3
(1, T, 32, 32, 32)

↓ SEW4
(1, T, 64, 32, 32)

↓ SEW5
(1, T, 64, 32, 32)

↓ SEW6
(1, T, 64, 32, 32)

↓ Temporal Mean (across T)
(1, 64, 32, 32)

↓ ANN Readout (1×1 Conv → ReLU → 1×1 Conv)
(1, 32, 32, 2) = LLR
```

---

#### 3.1 StemConv (First spiking feature extractor)

**Modules**

- Conv (2 → base_ch)  
- SpikeNorm  
- LIF (unrolling across timesteps)

**Purpose**  
Transforms raw I/Q features into spiking feature maps — the entry point to the SNN backbone.

**Relation to the Paper**  
Corresponds to the **“Stem” block** in the SpikingRx architecture.


#### 3.2 SEW Blocks × 6 (Spiking SNN Backbone)

Each SEW block contains:

- Conv → Norm → LIF  
- Conv → Norm → LIF  
- Shortcut branch  
- Spiking element-wise merge (SEW merge)

**Purpose**

Learn complex:

- Channel remnants  
- Constellation deformation  
- Symbol distribution patterns  

→ Produces a high-dimensional spiking representation.

**Relation to the Paper**

Corresponds to **Figure 3: “SEW-ResNet”**.


#### 3.3 Temporal Rate Pooling

Average across T timesteps:

```yaml
(1, T, C, 32, 32) → (1, C, 32, 32)
```

**Purpose**

Convert multi-timestep spiking activity into a stable feature map, used for soft demodulation.

**Relation to the Paper**

Corresponds to **time-averaged spiking representation.**


#### 3.4 ANN Readout (Soft Demodulation → LLR Output)

- 1×1 Conv  
- ReLU  
- 1×1 Conv  

Each RE outputs 2 values: **LLR for bit0 and bit1**

**Purpose**

Translate spike features into bitwise likelihood ratios.

**Relation to the Paper**

Corresponds to **“LLR-compatible Readout Module”** in the original architecture.


---

### (4) Visualization Modules

Includes:

- **LLR heatmap**  
  Shows soft bit decisions from SpikingRx.

- **Spike rate per stage**  
  Visualizes sparsity and spiking dynamics per SEW layer.

- **Spike GIF**  
  Shows timestep progression for each SEW block.

- **Training before/after comparison**  
  Demonstrates SNN learning dynamics (paper contains similar analyses; this project expands beyond).


**Relation to the Paper**  
Matches the **Experiments / Ablation Section**, with more detailed spike-activity visualization.


---

## 2.3 Project Folder Structure and Code Mapping

```yaml
SpikingRx-on-OAI/
│
├── src/
│ ├── data/
│ │ └── oai_to_spikingrx_tensor.py ← Paper: Input preprocessing
│ │
│ ├── models/
│ │ ├── conv_block.py ← Conv + Norm
│ │ ├── norm_layer.py ← SpikeNorm
│ │ ├── lif_neuron.py ← LIF + surrogate gradient
│ │ ├── sew_block.py ← SEW-ResNet block
│ │ └── spikingrx_model.py ← Full SpikingRx Receiver
│ │
│ ├── train/
│ │ ├── dataset_simple_ofdm.py ← Synthetic QPSK OFDM dataset
│ │ └── train_spikingrx.py ← Surrogate-gradient training
│ │
│ ├── inference/
│ │ └── run_spikingrx_on_oai_dump.py ← OAI dump → LLR inference
│ │
│ └── visualize/
│ └── visualize_spiking_activity.py ← Spike activity, before/after comparison
│
└── docs/
├── images/ ← Architecture diagrams, model diagrams
├── results/ ← LLR, spike GIF, comparison graphs
└── notes/ ← Research notes, design documentation
```


---

## 2.4 Chapter Summary

This chapter explained:

- How SpikingRx integrates with OAI  
- The technical role of each system block (input, processing, output)  
- How each module corresponds to specific sections of the SpikingRx paper  
- How the project directory structure cleanly reflects the architecture described in the paper 


# 3. Full System Flow of the SpikingRx × OAI Pipeline

The focus of this chapter is:

When you run the full SpikingRx system once,  
**what steps does it go through from beginning to end**,  
what each step does conceptually,  
and how to execute each step through the terminal.


## 3.1 Step 1: OAI Connection → Full-grid FFT Dump

### What this step does (concise and technical)

- Start gNB and UE in rfsim mode  
- Establish full RRC / PDCP / PHY connection  
- gNB begins transmitting PDSCH  
- UE receives OFDM data  

Inside the UE's PHY RX path, you added custom C code that:

**Dumps the entire 13 × 1272 FFT-processed Resource Grid into a binary file**

This means you have successfully “opened up” OAI’s internal receiver  
and extracted the exact data needed by SpikingRx.


### Files involved inside OAI

```vbnet
openairinterface5g/
└─ nr-ue
└─ PHY → nr_dlsch_demodulation.c (your modified location)
```

Your custom function is:

```powershell
spx_fullgrid_dump()
```


### Terminal Commands

Start gNB:

```powershell
sudo ./nr-softmodem --config-file gnb.conf --rfsim
```

Start UE:

```vbnet
sudo ./nr-uesoftmodem --config-file ue.conf --rfsim
```


Once the connection is established, OAI automatically outputs binary dumps:

```yaml
/tmp/spx_fullgrid_f{frame}_s{slot}.bin
```

This `.bin` file is the **true data source** for the SpikingRx receiver.


---

## 3.2 Step 2: Python Loader → OAI Dump → Convert to 32×32 SNN Input

### What this step does (concise)

- Load OAI's raw complex16 dump  
- Normalize  
- Map/reshape into fixed-size 32×32 grid  
- Add T timesteps (required for SNN temporal unrolling)  
- Return the tensor ready for SpikingRx  

### Python file used

```csharp
src/data/oai_to_spikingrx_tensor.py
```


### Terminal

You **never need to run the loader manually**.  
It is automatically called by the inference script.

Meaning, when you run:

```yaml
python src/inference/run_spikingrx_on_oai_dump.py
```

The loader will automatically load the latest dump.


---

## 3.3 Step 3: Training the SpikingRx SNN

### What this step does

Uses a **synthetic QPSK + OFDM dataset** generated by yourself to train the model.

The SNN learns:

- How to map OFDM symbols → LLR (soft bits)  
- Using surrogate gradients (Triangular surrogate)  
- Learns the soft decision boundaries of QPSK  

Important:  
Training **does not** use OAI dumps because OAI does not provide ground-truth bit labels.

### Files used

src/train/train_spikingrx.py

shell
複製程式碼

### Terminal commands

```yaml
cd ~/SpikingRx-on-OAI
source .venv/bin/activate
python src/train/train_spikingrx.py
```

After training, it produces:
```yaml
checkpoints/spikingrx_checkpoint.pth
```

This file is the **trained SNN receiver model**.


---

## 3.4 Step 4: Inference → Obtain LLR from OAI Data

### What this step does (concise)

- Automatically finds the latest OAI dump  
- Loads your trained checkpoint  
- Runs a full SpikingRx forward pass:
- 
```markdown
32×32 SNN input
→ spike propagation through SEW blocks
→ rate pooling
→ ANN readout
→ LLR output
```

It also saves:

- `llr.npy`  
- LLR heatmap  
- Spike-rate summary  


### Files used

```shell
src/inference/run_spikingrx_on_oai_dump.py
```


### Terminal command

```yaml
python src/inference/run_spikingrx_on_oai_dump.py
--ckpt checkpoints/spikingrx_checkpoint.pth
```

If you omit `--ckpt`, the script uses a **random (untrained) model**.


---

## 3.5 Step 5: Spike Visualization (Before vs After Training)

### What this step does (concise)

Uses the same OAI dump and runs **two** forward passes:

1. Random model (before training)  
2. Trained model (after training)  

Compares:

- Spike rate for each SEW block  
- Per-timestep GIF animation  
- LLR heatmap  
- SNN learning dynamics

This step is one of the **key highlights** of the entire project.


### Files used

```shell
src/visualize/visualize_spiking_activity.py
```




### Terminal command

```yaml
python src/visualize/visualize_spiking_activity.py
--ckpt checkpoints/spikingrx_checkpoint.pth
```

This outputs:

```yaml
docs/results/visualize/
spike_stage1.gif
spike_stage2.gif
spike_stage3.gif
spike_stage4.gif
spike_stage5.gif
spike_stage6.gif
spike_rate_before_after.png
llr_heatmap_trained.png
```


---

## 3.6 Step 6: Full SNN Receiver Pipeline Complete

By this point you have completed:

- OAI → FFT dump  
- Loader → tensor conversion  
- SNN training  
- OAI inference  
- Spike activity analysis  
- LLR heatmap visualization  

This is the **complete end-to-end SpikingRx receiver system**.


---

## Summary of Chapter 3 (Concise but Fully Technical)

| Step | Purpose | Python/OAI Files | Terminal |
|------|---------|------------------|----------|
| 1. OAI Dump | Extract FFT grid | `nr_dlsch_demodulation.c` | Start gNB/UE |
| 2. Loader | Dump → 32×32 Tensor | `oai_to_spikingrx_tensor.py` | Auto-run |
| 3. Training | Learn QPSK → checkpoint | `train_spikingrx.py` | `python train_spikingrx.py` |
| 4. Inference | Dump → SNN → LLR | `run_spikingrx_on_oai_dump.py` | `python ... --ckpt` |
| 5. Visualization | Before/after comparison | `visualize_spiking_activity.py` | `python visualize_spiking_activity.py` |
