# Uneshance_2023
[Unsehance_2023.pdf](https://github.com/Smruti0603/Uneshance_2023/files/12482798/Unsehance_2023.pdf)

😌 Abstract
Medical imaging plays a crucial role in a number of clinical applications, allowing for pre- cise analysis, diagnosis, and treatment planning. However, the quality of medical images is frequently constrained by hardware, imaging protocols, and patient motion.Reduced spatial details in low-resolution medical images result in the loss of fine structures and crucial di- agnostic information. This restriction can impede the precise identification and segmentation of anatomical structures, lesions, and abnormalities. Low-resolution images may also contain blurring, noise, and artefacts, which further complicates image interpretation and analysis.
This proposal presents a novel method for enhancing Single Image Super Resolution for Med- ical Images. The proposed method ESRGAN(Enhanced Super-Resolution Generative Ad- versarial Network ) along with the attention mechanism employs deep learning-based ap- proaches, specifically General adversarial Networks (GANs), to discover the mapping between low- and high-resolution medical images.



😃 Outline of procedure and methodology
ESRGAN is the enhanced version of the SRGAN(Super-Resolution Generative Adversarial Network), Author primarily modify the structure of generator G in two ways: 1) Eliminate all BN layers, as shown in Figure 2; 2) Replace the original basic block with the proposed Residual-in-Residual Dense Block (RRDB), which incorporates multi-level residual network and dense connections.

![2](https://github.com/Smruti0603/Uneshance_2023/assets/121166411/72064feb-2521-438e-b3f3-f6267739b364)



Figure 1: Network Architecture with SE Block
