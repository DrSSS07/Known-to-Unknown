# Known-to-Unknown  
**Generative Background Replacement: Integrating Semantic Segmentation with Text-to-Image Models**  

Author: **Dr. Sivabalan Settu**  
Professor of Computer Science and Engineering, Andhra, India  
📧 Email: sivabalan1990s@gmail.com  

---

## 📖 Overview  
This repository contains the source code, datasets, and LaTeX files for the paper:  
*Generative Background Replacement: Integrating Semantic Segmentation with Text-to-Image Models*  

The project explores a unified framework that integrates **semantic segmentation** and **text-to-image diffusion models** for background replacement. It demonstrates how segmentation preserves object boundaries while generative AI produces realistic, creative, and controllable background replacements.  

Applications include:  
- 🎥 Film and multimedia production  
- 🕶️ AR/VR immersive environments  
- 🖼️ Digital media design  
- 📹 Virtual conferencing  

---

## ⚙️ Features  
- Foreground extraction via **Mask2Former/DeepLab**  
- Background replacement with **Stable Diffusion + ControlNet**  
- Evaluation on **COCO-Stuff** and **ADE20K** datasets  
- Metrics: **FID, SSIM, IoU, Human preference scores**  

---

## 📂 Repository Structure  
```bash
.
├── data/                 # Dataset links or preprocessing scripts
├── src/                  # Source code (segmentation + diffusion integration)
├── paper/                # LaTeX source of the research paper
├── figures/              # TikZ/PNG diagrams used in the paper
├── README.md             # Project documentation
└── LICENSE               # MIT License
