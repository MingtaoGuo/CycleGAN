# CycleGAN
A simple code of CycleGAN which is easy to read is implemented by TensorFlow

Method
------

Please see the detail about the paper CycleGAN from here [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf).The method of original pix2pix needs lots of paired datasets. For example, sketch-to-cat translation, we should have the datasets about many cats and its corresponding sketches in pixelwise. Sometimes, paired datasets are very rare. Just like man-to-woman translation for pix2pix it's impossible, but the method of this paper(CycleGAN) it's become possible. The most innovation point of this paper is about that the kinds of datasets don't need paired image, instead it just need two domains of different kinds of samples. For example, one domain is about man, another is about woman.

The main method of this paper, please see the image below(from the paper). Inspiration of "Cycle" is from language to language translation[Dual Learning for Machine Translation](http://papers.nips.cc/paper/6469-dual-learning-for-machine-translation.pdf). For example, French-to-English and English-to-French, they called it dual learning. In CycleGAN, datasets domain X and datasets domain Y translate each other, like this X-to-Y and Y-to-X.

<div align=center><img src=https://github.com/MingtaoGuo/CycleGAN/raw/master/method/cycleGAN_method.jpg></div>

In this code, we don't use the loss function of [LSGAN](http://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf), instead we use [WGAN](https://arxiv.org/abs/1701.07875) which has been proved that it can yield high quality results and faster convergence rate, the problem of mode collapse of GAN also be solved by WGAN. But WGAN has shotcoming that it must satisfy the 1-Lipschitz condition. [WGAN-GP](http://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf) solve the problem of WGAN about 1-Lipschitz condition by gradient penalty, but it has a problem that the computation of gradient penalty cost too much time in training phase. In this code, we don't use gradient penalty, instead we use [spectral Normalization](https://arxiv.org/abs/1802.05957) which is published in ICLR2018, it not only can yield high quality results like WGAN-GP, but also cost less time in training phase.

Results from the paper
----------------------

<div align=center><img src=https://github.com/MingtaoGuo/CycleGAN/raw/master/result/resultOfPaper.jpg></div>

How to use the code 
---------------------
Firstly, you should install python and install some packages of python like tensorflow, numpy, scipy, pillow etc.

Please use these commands to install:
1. pip install tensorflow
2. pip install numpy
3. pip install scipy
4. pip install pillow

Secondly, putting two domains of datasets into the folder X and Y. For example, put the datasets about man into X, and put the datasets about woman into Y. I will provide one datasets about man2woman, which is selected from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). There are about 8000 images for man, and 10000 images for woman and the images i had croped and resized it to 256x256. Please download the datasets from here [BaiDuYun](https://pan.baidu.com/s/1PNBklyMbn7wESkW5UIbMVw)

Results of this code 
------------------------

Because of my poor device, i just train the CycleGAN for 13 epochs, the result of man2woman as shown below, the result of woman2man looks not really well, so i don't show it.

<div align=center><img src=https://github.com/MingtaoGuo/CycleGAN/raw/master/result/resultofmine.jpg></div>



