# DID-MDN
## Density-aware Single Image De-raining using a Multi-stream Dense Network
He Zhang, Vishal M. Patel

[[Paper Link](https://arxiv.org/abs/1701.05957)] (CVPR'18)

We present a novel density-aware multi-stream densely connected convolutional neural
network-based algorithm, called DID-MDN, for joint rain density estimation and de-raining. The proposed method
enables the network itself to automatically determine the rain-density information and then efficiently remove the
corresponding rain-streaks guided by the estimated rain-density label. To better characterize rain-streaks with dif-
ferent scales and shapes, a multi-stream densely connected de-raining network is proposed which efficiently leverages
features from different scales. Furthermore, a new dataset containing images with rain-density labels is created and
used to train the proposed density-aware network. 

	@inproceedings{derain_zhang_2018,		
	  title={Density-aware Single Image De-raining using a Multi-stream Dense Network},
	  author={Zhang, He and Patel, Vishal M},
	  booktitle={CVPR},
	  year={2018}
	} 

<img src="sample_results/121_input.jpg" width="400px" height="200px"/><img src="sample_results/121_input.jpg" width="400px" height="200px"/>



## Prepare
Instal torch7

Install nngraph

Install hdf5
 
Download the dataset from (https://drive.google.com/open?id=0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s) 
and put the dataset folder into the "IDCGAN" folder

## Training

	DATA_ROOT=./datasets/rain name=rain which_direction=BtoA th train.lua

## Testing

	DATA_ROOT=./datasets/rain name=rain which_direction=BtoA phase=test_nature th test.lua


##  Testing using ID-CGAN model
The trained ID-CGAN model  and our training and testing datasets can be found at 
(https://drive.google.com/open?id=0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s)

*Make sure you download the vgg model that used for perceotual loss and put it in the ./IDCGAN/per_loss/models



##Acknowledgments##

Code borrows heavily from [[pix2pix](https://github.com/phillipi/pix2pix)]
 and [[Perceptual Loss](https://github.com/jcjohnson/fast-neural-style)]. Thanks for the sharing.
