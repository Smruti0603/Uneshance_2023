# Uneshance_2023
PDF ➜ [Unsehance_2023.pdf](https://github.com/Smruti0603/Uneshance_2023/files/12482798/Unsehance_2023.pdf)

😌 Abstract
Medical imaging plays a crucial role in a number of clinical applications, allowing for pre- cise analysis, diagnosis, and treatment planning. However, the quality of medical images is frequently constrained by hardware, imaging protocols, and patient motion.Reduced spatial details in low-resolution medical images result in the loss of fine structures and crucial di- agnostic information. This restriction can impede the precise identification and segmentation of anatomical structures, lesions, and abnormalities. Low-resolution images may also contain blurring, noise, and artefacts, which further complicates image interpretation and analysis.
This proposal presents a novel method for enhancing Single Image Super Resolution for Med- ical Images. The proposed method ESRGAN(Enhanced Super-Resolution Generative Ad- versarial Network ) along with the attention mechanism employs deep learning-based ap- proaches, specifically General adversarial Networks (GANs), to discover the mapping between low- and high-resolution medical images.



😃 Outline of procedure and methodology
ESRGAN is the enhanced version of the SRGAN(Super-Resolution Generative Adversarial Network), Author primarily modify the structure of generator G in two ways: 1) Eliminate all BN layers, as shown in Figure 2; 2) Replace the original basic block with the proposed Residual-in-Residual Dense Block (RRDB), which incorporates multi-level residual network and dense connections.



![2](https://github.com/Smruti0603/Uneshance_2023/assets/121166411/8c1ac266-76bf-45a4-adb4-6da84407fe3b)


Figure 1: Network Architecture with SE Block


![3](https://github.com/Smruti0603/Uneshance_2023/assets/121166411/f0dd4c66-11c4-4550-b789-3d6c474547cb)


Figure 2: Residual Block and Residual Block w/o BN)



Besides using standard discriminator ESRGAN uses the relativistic GAN, The high-level ar- chitecture of the GAN contains two main networks namely the generator network and the dis- criminator network. The generator network tries to generate the fake data and the discriminator network tries to distinguish between real and fake data, hence helping the generator to generate more realistic data eventually predicting the probability that the real image is relatively more realistic than a fake image , you can check out the paper [2].
Our enhanced ESRGAN model with the attention mechanism can be referred from the Figure 3

![1](https://github.com/Smruti0603/Uneshance_2023/assets/121166411/e382277b-2b70-45d6-b63d-b4073397acde)

👀Attention mechanism :Attention can be interpreted as a means of biasing the allocation of available computational resources towards the most informative components of a signal. Atten- tion mechanisms have demonstrated their utility across many tasks including sequence learn- ing, localisation and understanding in images , image captioning and lip reading. In these applications, it can be incorporated as an operator following one or more layers representing higher-level abstractions for adaptation between modalities. In this paper we proposed apply- ing attention mechanism using SE block, which is an architectural unit designed to improve the representational power of a network by enabling it to perform dynamic channel-wise fea- ture recalibration. SE block [1] comprises a lightweight gating mechanism which focuses on enhancing the representational power of the network by modelling channel-wise relationships in a computationally efficient manner.


We have integrated the SE Block in the network architecure of ESRGAN by replacing with a convolution layer after the up-sampling layer.SE block’s flexibility allows it to be applied directly on transformations beyond standard convolutions. The SE Block is shown in Figure 4
1. The function is given an input convolutional block and the current number of channels it has.
2. It squeezes each channel to a single numeric value using average pooling.
3. A fully connected layer followed by a ReLU function adds the necessary non linearity. It’s output channel complexity is also reduced by a certain ratio.
4. A second fully connected layer followed by a Sigmoid activation gives each channel a smooth gating function.
5. At last, we weight each feature map of the convolutional block based on the result of our side network.


![4](https://github.com/Smruti0603/Uneshance_2023/assets/121166411/9f127940-8eb7-48bf-a416-380a26e19b10)


💪🏽Training
To evaluate the influence of SE block , we performed model traning on the IVUS image dataset provided USenhance 2023 which comprises 1045 IVUS images of various parts of our body including breast , thyroid etc.We trained the network on the training set and report the PSNR , MSE and SSIM on the testing set.The model has been trained using 300 epochs.

#➣Results

PSNR --> 34.38 
MSE --> 25.31 
SSIM --> 0.83

References

[1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, and Enhua Wu. Squeeze-and-excitation networks. september 2017. URL https://arxiv. org/abs/1709.01507 v4, 2018.
[2] Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, and Xiaoou Tang. Esrgan: Enhanced super-resolution generative adversarial networks (2018). arXiv preprint arXiv:1809.00219, 1809.




