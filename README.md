# Polar Transform Network (PTN)
Polar Transform Network for Prostate Ultrasound Segmentation with Uncertainty Estimation

This is a python (PyTorch) implementation of **Polar Transform Network (PTN)** method for prostate ultrasound segmentation proposed in our ***Medical Image Analysis*** journal paper [**"Polar Transform Network for Prostate Ultrasound Segmentation with Uncertainty Estimation"**](https://doi.org/10.1016/j.media.2022.102418).

## Citation
  *X. Xu et al., "Polar transform network for prostate ultrasound segmentation with uncertainty estimation," Med. Image Anal., vol. 78, May 2022, Art. no. 102418.*

    @article{Xu2022PTN,
      title={Polar transform network for prostate ultrasound segmentation with uncertainty estimation}, 
      author={Xu, Xuanang and Sanford, Thomas and Turkbey, Baris and Xu, Sheng and Wood, Bradford J. and Yan, Pingkun},
      journal={Medical Image Analysis}, 
      volume = {78},
      pages={102418},
      year={2022},
      publisher={Elsevier},
      doi={10.1016/j.media.2022.102418}
    }

## Update
  - **Mar 31, 2022**: You can access the final version of our article on *ScienceDirect* through the [Personalized Share Link](https://authors.elsevier.com/c/1eo-W_UzlO11E5) provided by *Elsevier* (available before *May 15, 2022*).

## Abstract
Automatic and accurate prostate ultrasound segmentation is a long-standing and challenging problem due to the severe noise and ambiguous/missing prostate boundaries. In this work, we propose a novel polar transform network (PTN) to handle this problem from a fundamentally new perspective, where the prostate is represented and segmented in the polar coordinate space rather than the original image grid space. This new representation gives a prostate volume, especially the most challenging apex and base sub-areas, much denser samples than the background and thus facilitate the learning of discriminative features for accurate prostate segmentation. Moreover, in the polar representation, the prostate surface can be efficiently parameterized using a 2D surface radius map with respect to a centroid coordinate, which allows the proposed PTN to obtain superior accuracy compared with its counterparts using convolutional neural networks while having significantly fewer (18%~41%) trainable parameters. We also equip our PTN with a novel strategy of centroid perturbed test-time augmentation (CPTTA), which is designed to further improve the segmentation accuracy and quantitatively assess the model uncertainty at the same time. The uncertainty estimation function provides valuable feedback to clinicians when manual modifications or approvals are required for the segmentation, substantially improving the clinical significance of our work. We conduct a three-fold cross validation on a clinical dataset consisting of 315 TRUS images to comprehensively evaluate the performance of the proposed method. The experimental results show that our proposed PTN with CPTTA outperforms the state-of-the-art methods with statistical significance on most of the metrics while exhibiting a much smaller model size. Source code of the proposed PTN is released at [https://github.com/DIAL-RPI/PTN](https://github.com/DIAL-RPI/PTN).

## Method
### Scheme of Polar Transformation for Prostate Ultrasound Segmentation
<img src="./fig2.png"/>

### Architecture of Polar Transform Network
<img src="./fig1.png"/>

## Contact
You are welcome to contact us:  
  - [xux12@rpi.edu](mailto:xux12@rpi.edu)(Dr. Xuanang Xu)
