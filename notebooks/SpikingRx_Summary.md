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

FFT
→ Channel Estimation
→ Equalizer
→ Soft Demapper (LLR)
→ LDPC Decoder


```yaml
SpikingRx receiver:
```
FFT
→ SpikingRx (direct LLR output)
→ LDPC Decoder


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
