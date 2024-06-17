### This repository contains the steps and code to reproduce the work for "Exploring the Capabilities and Limitations of Video Stream Fingerprinting" (to be presented at the SecWeb 2024 workshop) and my dissertation.

This is a work-in-progress through at least mid-2025, and intended only for academic research purposes. The sister repository with the code (and lists of our URLs) for our dataset collection effort is https://github.com/timwalsh300/tor-browser-crawler-video

For access to the raw dataset, please contact us through this form: https://docs.google.com/forms/d/1ldfKdUBMA2DJNh1_sGth-puYQfadybvNb3CPiihY3cE

Due to the considerable size of the raw dataset, to make it available while minimizing the cost of storage, we are using the Amazon S3 Glacier Deep Archive. That requires some coordination to grant access, including a restore request that can take up to 48 hours on Amazon's end. The raw dataset is organized in four parts:

1. monitored_https (369 GB as .tar.7z, 730 GB uncompressed)

2. monitored_tor (319 GB as .tar.7z, 961 GB uncompressed)

3. unmonitored_https (96 GB as .tar.7z, 204 GB uncompressed)

4. unmonitored_tor (110 GB as .tar.7z, 377 GB uncompressed)

### Explanation of the parts of this repository (and my to-do list):

0_raw_to_csv: Launch separate batch jobs by modifying the monitored_raw_to_csv.sh script for every combination of representation, protocol, and platform to turn the raw dataset into initial .csv files.

- [x] sirinam_wf, sirinam_vf, rahman, hayden, schuster2, schuster4, schuster8, dschuster8, schuster16, dschuster16

1_csv_to_pkl: From an interactive session with 128 GB of memory, run the csv_to_pkl.py script with no arguments to turn all the initial .csv files into intermediate .csv files and .pkl files with the closed-world train/val/test splits.

- [x] sirinam_wf, sirinam_vf, rahman, hayden, schuster2, schuster4, schuster8, dschuster8, schuster16, dschuster16

2_closed_world: Launch separate batch jobs by modifying the search.sh script for every combination of representation, protocol, and platform to find good hyperparameters with Ray Tune. Look at the tail of the .out files to see the best hyperparameters found and copy them into the dictionary in the evaluation.py script. Launch a batch job with evaluation.sh and look at the .out file to see the number of epochs, training time, and test set accuracy for each model.

- [x] Ray Tune search for dschuster16, schuster16, rschuster8, dschuster8, schuster8, schuster4, schuster2, hayden, rahman, sirinam_vf

- [x] put all best found hyperparameters into evaluation.py

- [x] evaluation batch job for all

3_open_world_baseline: 61-way classification task with the best representation(s) only. Modify and run the 0_raw_to_csv/unmonitored_raw_to_csv.sh script for best closed-world data representations. Create train/val/test splits with csv_to_pkl_open.py for both HTTPS and Tor. Modify and run search_open.sh to do hyperparameter searches with Ray Tune. Copy best found hyperparameters into train_open.py and run that to train and save the models. Output P-R curve figures and results for P, R, F1, FP, FPR on test sets 1k to 64k after choosing thresholds using the validation set with evaluation_open.py.

- [x] parsing unmonitored set, creating train/val/test sets, for schuster8 (Tor) and dschuster16 (HTTPS)

- [x] hyperparameter search, training for dschuster16 (HTTPS), schuster8 (Tor)

- [x] produce P-R curves over validation and test sets

- [x] select thresholds based on max F1, zero FP, and 0.5 R over the validation set

4_open_world_enhancements: Experiment with a number of more sophisticated methods attempting to address the theoretical shortcomings of the baseline, existing approach to the open-world / open set recognition task. 

- [x] Temperature scaling, threshold for calibrated max softmax probability

- [x] Monte Carlo Dropout, Spike-and-Slab Concrete Dropout, threshold for Bayesian model averaged max softmax probability, total uncertainty, and epistemic uncertainty, with Standard Model / Background Class

- [x] mixup for training data augmentation...

- [ ] NOTA defensive padding for training data augmentation...

- [x] GAN-trained discriminator / augmentation of training data with generated fakes, threshold for discriminator prediction, also with mixup...

- [ ] Class-specific autoencoders, threshold for reconstruction error...

5_across_vantage_points

- [ ] new training and validation sets from just two vantage points with distinct box plots (see end of Chapter 3)

- [ ] new training and validation sets, same size, but sampled from all vantage points except us-west-2

- [ ] all data from all vantage points except us-west-2

- [ ] test against residential ISP and Wi-Fi
