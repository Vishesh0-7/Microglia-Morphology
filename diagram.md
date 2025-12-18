# Deep Learning Architectures for Microglia Morphology Analysis

## 1. Supervised Classifier Architecture

```mermaid
flowchart TD
    subgraph INPUT1["Input Modalities"]
        I1["Cell Body Image<br/>128x128x1"]
        I2["Skeleton Image<br/>128x128x1"]
        I3["Skeleton Features<br/>9 dimensions"]
    end
    
    subgraph CELL_ENC["Cell Image Encoder - CNN"]
        C1["Conv2d: 1→32<br/>kernel=3, padding=1"]
        C2["BatchNorm + ReLU<br/>MaxPool 2x2"]
        C3["Conv2d: 32→64<br/>kernel=3, padding=1"]
        C4["BatchNorm + ReLU<br/>MaxPool 2x2"]
        C5["Conv2d: 64→128<br/>kernel=3, padding=1"]
        C6["BatchNorm + ReLU<br/>MaxPool 2x2"]
        C7["Conv2d: 128→256<br/>kernel=3, padding=1"]
        C8["BatchNorm + ReLU<br/>MaxPool 2x2"]
        C9["Flatten<br/>256×8×8=16384"]
        C10["Linear: 16384→512<br/>Dropout 0.3"]
        C11["Linear: 512→128"]
        
        C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7 --> C8 --> C9 --> C10 --> C11
    end
    
    subgraph SKEL_ENC["Skeleton Image Encoder - CNN"]
        S1["Conv2d: 1→32<br/>kernel=3, padding=1"]
        S2["BatchNorm + ReLU<br/>MaxPool 2x2"]
        S3["Conv2d: 32→64<br/>kernel=3, padding=1"]
        S4["BatchNorm + ReLU<br/>MaxPool 2x2"]
        S5["Conv2d: 64→128<br/>kernel=3, padding=1"]
        S6["BatchNorm + ReLU<br/>MaxPool 2x2"]
        S7["Conv2d: 128→256<br/>kernel=3, padding=1"]
        S8["BatchNorm + ReLU<br/>MaxPool 2x2"]
        S9["Flatten<br/>256×8×8=16384"]
        S10["Linear: 16384→512<br/>Dropout 0.3"]
        S11["Linear: 512→128"]
        
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9 --> S10 --> S11
    end
    
    subgraph FEAT_ENC["Feature Encoder - MLP"]
        F1["Linear: 9→64<br/>ReLU + Dropout 0.2"]
        F2["Linear: 64→64<br/>ReLU + Dropout 0.2"]
        
        F1 --> F2
    end
    
    subgraph CLASSIFIER["Fusion & Classification"]
        CL1["Concatenate<br/>128+128+64=320"]
        CL2["Linear: 320→256<br/>ReLU + Dropout 0.3"]
        CL3["Linear: 256→128<br/>ReLU + Dropout 0.3"]
        CL4["Linear: 128→5<br/>Softmax"]
        CL5["Output Classes"]
        
        CL1 --> CL2 --> CL3 --> CL4 --> CL5
    end
    
    I1 --> C1
    I2 --> S1
    I3 --> F1
    C11 --> CL1
    S11 --> CL1
    F2 --> CL1
    
    style INPUT1 fill:#e1f5ff
    style CELL_ENC fill:#fff4e1
    style SKEL_ENC fill:#ffe1f5
    style FEAT_ENC fill:#e1ffe1
    style CLASSIFIER fill:#ffe1e1
```

---

## 2. Unsupervised Autoencoder Architecture

```mermaid
flowchart TD
    subgraph INPUT2["Input Modalities"]
        AI1["Cell Body Image<br/>128x128x1"]
        AI2["Skeleton Image<br/>128x128x1"]
        AI3["Skeleton Features<br/>9 dimensions"]
    end
    
    subgraph ENCODERS["Encoders"]
        subgraph CE["Cell Image Encoder"]
            CE1["Conv2d: 1→32<br/>stride=2, kernel=4"]
            CE2["Conv2d: 32→64<br/>stride=2, kernel=4"]
            CE3["Conv2d: 64→128<br/>stride=2, kernel=4"]
            CE4["Conv2d: 128→256<br/>stride=2, kernel=4"]
            CE5["FC: 256×8×8→512"]
            CE6["FC: 512→32"]
            CE1 --> CE2 --> CE3 --> CE4 --> CE5 --> CE6
        end
        
        subgraph SE["Skeleton Image Encoder"]
            SE1["Conv2d: 1→32<br/>stride=2, kernel=4"]
            SE2["Conv2d: 32→64<br/>stride=2, kernel=4"]
            SE3["Conv2d: 64→128<br/>stride=2, kernel=4"]
            SE4["Conv2d: 128→256<br/>stride=2, kernel=4"]
            SE5["FC: 256×8×8→512"]
            SE6["FC: 512→32"]
            SE1 --> SE2 --> SE3 --> SE4 --> SE5 --> SE6
        end
        
        subgraph FE["Feature Encoder"]
            FE1["Linear: 9→32<br/>ReLU + Dropout"]
            FE2["Linear: 32→16<br/>ReLU + Dropout"]
            FE1 --> FE2
        end
    end
    
    subgraph LATENT["Latent Space"]
        LAT["Concatenate<br/>32+32+16=80<br/><b>Compressed Representation</b>"]
    end
    
    subgraph DECODERS["Decoders"]
        subgraph CD["Cell Image Decoder"]
            CD1["FC: 80→512"]
            CD2["FC: 512→256×8×8"]
            CD3["Reshape: 256×8×8"]
            CD4["ConvTranspose2d<br/>256→128"]
            CD5["ConvTranspose2d<br/>128→64"]
            CD6["ConvTranspose2d<br/>64→32"]
            CD7["ConvTranspose2d<br/>32→1"]
            CD1 --> CD2 --> CD3 --> CD4 --> CD5 --> CD6 --> CD7
        end
        
        subgraph SD["Skeleton Image Decoder"]
            SD1["FC: 80→512"]
            SD2["FC: 512→256×8×8"]
            SD3["Reshape: 256×8×8"]
            SD4["ConvTranspose2d<br/>256→128"]
            SD5["ConvTranspose2d<br/>128→64"]
            SD6["ConvTranspose2d<br/>64→32"]
            SD7["ConvTranspose2d<br/>32→1"]
            SD1 --> SD2 --> SD3 --> SD4 --> SD5 --> SD6 --> SD7
        end
        
        subgraph FD["Feature Decoder"]
            FD1["Linear: 80→64<br/>ReLU + Dropout"]
            FD2["Linear: 64→32<br/>ReLU + Dropout"]
            FD3["Linear: 32→9"]
            FD1 --> FD2 --> FD3
        end
    end
    
    subgraph OUTPUT2["Reconstructed Outputs"]
        AO1["Reconstructed<br/>Cell Image"]
        AO2["Reconstructed<br/>Skeleton Image"]
        AO3["Reconstructed<br/>Features"]
    end
    
    AI1 --> CE1
    AI2 --> SE1
    AI3 --> FE1
    
    CE6 --> LAT
    SE6 --> LAT
    FE2 --> LAT
    
    LAT --> CD1
    LAT --> SD1
    LAT --> FD1
    
    CD7 --> AO1
    SD7 --> AO2
    FD3 --> AO3
    
    style INPUT2 fill:#e1f5ff
    style ENCODERS fill:#fff4e1
    style LATENT fill:#ff6b6b,color:#fff
    style DECODERS fill:#ffe1f5
    style OUTPUT2 fill:#e1ffe1
```

---

## 3. Denoising Autoencoder Architecture

```mermaid
flowchart TD
    subgraph INPUT3["Input Modalities"]
        DI1["Cell Body Image<br/>128x128x1"]
        DI2["Skeleton Image<br/>128x128x1"]
        DI3["Skeleton Features<br/>9 dimensions"]
    end
    
    subgraph NOISE["Noise Addition Layer"]
        N1["Gaussian Noise<br/>σ=0.1"]
        N2["Salt & Pepper<br/>amount=0.05"]
        N3["Feature Dropout<br/>p=0.1"]
    end
    
    subgraph NOISY["Noisy Inputs"]
        NI1["Noisy Cell Image"]
        NI2["Noisy Skeleton"]
        NI3["Noisy Features"]
    end
    
    subgraph DENCODERS["Encoders"]
        subgraph DCE["Cell Image Encoder"]
            DCE1["Conv2d: 1→32<br/>stride=2, kernel=4"]
            DCE2["Conv2d: 32→64<br/>stride=2, kernel=4"]
            DCE3["Conv2d: 64→128<br/>stride=2, kernel=4"]
            DCE4["Conv2d: 128→256<br/>stride=2, kernel=4"]
            DCE5["FC: 256×8×8→512"]
            DCE6["FC: 512→32"]
            DCE1 --> DCE2 --> DCE3 --> DCE4 --> DCE5 --> DCE6
        end
        
        subgraph DSE["Skeleton Image Encoder"]
            DSE1["Conv2d: 1→32<br/>stride=2, kernel=4"]
            DSE2["Conv2d: 32→64<br/>stride=2, kernel=4"]
            DSE3["Conv2d: 64→128<br/>stride=2, kernel=4"]
            DSE4["Conv2d: 128→256<br/>stride=2, kernel=4"]
            DSE5["FC: 256×8×8→512"]
            DSE6["FC: 512→32"]
            DSE1 --> DSE2 --> DSE3 --> DSE4 --> DSE5 --> DSE6
        end
        
        subgraph DFE["Feature Encoder"]
            DFE1["Linear: 9→32<br/>ReLU + Dropout"]
            DFE2["Linear: 32→16<br/>ReLU + Dropout"]
            DFE1 --> DFE2
        end
    end
    
    subgraph DLATENT["Latent Space"]
        DLAT["Concatenate<br/>32+32+16=80<br/><b>Robust Features</b>"]
    end
    
    subgraph DDECODERS["Decoders"]
        subgraph DCD["Cell Image Decoder"]
            DCD1["FC: 80→512"]
            DCD2["FC: 512→256×8×8"]
            DCD3["Reshape: 256×8×8"]
            DCD4["ConvTranspose2d<br/>256→128"]
            DCD5["ConvTranspose2d<br/>128→64"]
            DCD6["ConvTranspose2d<br/>64→32"]
            DCD7["ConvTranspose2d<br/>32→1"]
            DCD1 --> DCD2 --> DCD3 --> DCD4 --> DCD5 --> DCD6 --> DCD7
        end
        
        subgraph DSD["Skeleton Image Decoder"]
            DSD1["FC: 80→512"]
            DSD2["FC: 512→256×8×8"]
            DSD3["Reshape: 256×8×8"]
            DSD4["ConvTranspose2d<br/>256→128"]
            DSD5["ConvTranspose2d<br/>128→64"]
            DSD6["ConvTranspose2d<br/>64→32"]
            DSD7["ConvTranspose2d<br/>32→1"]
            DSD1 --> DSD2 --> DSD3 --> DSD4 --> DSD5 --> DSD6 --> DSD7
        end
        
        subgraph DFD["Feature Decoder"]
            DFD1["Linear: 80→64<br/>ReLU + Dropout"]
            DFD2["Linear: 64→32<br/>ReLU + Dropout"]
            DFD3["Linear: 32→9"]
            DFD1 --> DFD2 --> DFD3
        end
    end
    
    subgraph OUTPUT3["Clean Reconstructed Outputs"]
        DO1["Clean Cell Image<br/><i>(Denoised)</i>"]
        DO2["Clean Skeleton<br/><i>(Denoised)</i>"]
        DO3["Clean Features<br/><i>(Denoised)</i>"]
    end
    
    DI1 --> N1
    DI2 --> N2
    DI3 --> N3
    
    N1 --> NI1
    N2 --> NI2
    N3 --> NI3
    
    NI1 --> DCE1
    NI2 --> DSE1
    NI3 --> DFE1
    
    DCE6 --> DLAT
    DSE6 --> DLAT
    DFE2 --> DLAT
    
    DLAT --> DCD1
    DLAT --> DSD1
    DLAT --> DFD1
    
    DCD7 --> DO1
    DSD7 --> DO2
    DFD3 --> DO3
    
    style INPUT3 fill:#e1f5ff
    style NOISE fill:#ffcc00
    style NOISY fill:#ff9999
    style DENCODERS fill:#fff4e1
    style DLATENT fill:#9b59b6,color:#fff
    style DDECODERS fill:#ffe1f5
    style OUTPUT3 fill:#90ee90
```

---

## Architecture Comparison Summary

| Architecture | Purpose | Latent Dimension | Key Feature |
|-------------|---------|------------------|-------------|
| **Supervised Classifier** | Cell classification | 320 (fusion) | Supervised learning with labeled data |
| **Unsupervised Autoencoder** | Feature learning & clustering | 80 | Self-supervised reconstruction |
| **Denoising Autoencoder** | Robust feature extraction | 80 | Learns robust features from noisy inputs |

