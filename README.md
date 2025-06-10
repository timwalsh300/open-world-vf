### This repository contains the steps and code to reproduce the work for "Improved Open-World Fingerprinting Increases Threat to Streaming Video Privacy but Realistic Scenarios Remain Difficult" (to be presented at the 2025 Privacy Enhancing Technologies Symposium) and "Exploring the Capabilities and Limitations of Video Stream Fingerprinting" (presented at the 2024 IEEE S&P Workshop on Designing Security for the Web).

This code is intended only for academic research purposes.

The sister repository with the code (and lists of our URLs) for our associated dataset collection effort is https://github.com/timwalsh300/tor-browser-crawler-video

For access to the raw dataset, please contact us through this form: https://docs.google.com/forms/d/1ldfKdUBMA2DJNh1_sGth-puYQfadybvNb3CPiihY3cE

Due to the considerable size of the raw dataset, to make it available while minimizing the cost of storage, we are using the Amazon S3 Glacier Deep Archive. That requires some coordination to grant access, including a restore request that can take up to 48 hours on Amazon's end. The raw dataset is organized in four parts:

1. monitored_https (369 GB as .tar.7z, 730 GB uncompressed)

2. monitored_tor (319 GB as .tar.7z, 961 GB uncompressed)

3. unmonitored_https (96 GB as .tar.7z, 204 GB uncompressed)

4. unmonitored_tor (110 GB as .tar.7z, 377 GB uncompressed)

### Explanation of the parts of this repository:

0_raw_to_csv: This is the code for parsing the .pcap files from the crawler(s) and turning them into .csv files for each per packet or per time step data representation that we tried. Launch separate Slurm batch jobs by modifying the monitored_raw_to_csv.sh script for every combination of representation, protocol, and platform.

1_csv_to_pkl: This is the code that converts the .csv files into closed-world train/val/test splits in the form of NumPy arrays, and then saves those arrays to disk using pickle. From an interactive session with 128 GB of memory, run the csv_to_pkl.py script with no arguments. The process also creates some intermediate .csv files.

2_closed_world: This is the code for tuning, training, and testing our closed-world models. Launch separate Slurm batch jobs by modifying the search.sh script for every combination of representation, protocol, and platform to find good hyperparameters with Ray Tune. Look at the tail of the .out files to see the best hyperparameters found and copy them into the dictionary in the evaluation.py script. Launch a Slurm batch job with evaluation.sh and look at the .out file to see the results of training and testing each model.

3_open_world_baseline: This is the code for our baseline open-world experiments. Modify and run the 0_raw_to_csv/unmonitored_raw_to_csv.sh script for best data representations found for Vimeo through our closed-world experiments. Create open-world train/val/test splits with csv_to_pkl_open.py for both protocols. Modify and run search_open.sh to do hyperparameter searches again with Ray Tune. Copy the best found hyperparameters into train_open.py and run that to train and save the models. Run evaluation_open.py to get precision-recall curve figures and results (at various thresholds) for precision, recall, F1, false positives, and false positive rate on the 1k to 64k test sets.

4_open_world_enhancements: This is the code for our adaptation and experimentation with a number of more recent or advanced techniques and approaches that attempt to address the theoretical shortcomings of the baseline, existing approach to the open-world / open set recognition task: temperature scaling for calibrated MSP and further separation of in- and out-of-distribution instances with Standard Model; 

- [x] Spike-and-Slab, Concrete Dropout for Bayesian model average MSP and total uncertainty with Standard Model

- [x] mixup for training data augmentation, all pairs with Standard Model, both deterministic and Bayesian models

- [x] NOTA defensive padding for training data augmentation, untargeted and targeted adversarial examples with PGD, mean and uniform (weighted average) padding between original and adversarial examples, with Standard Model, also combined with mixup and both deterministic and Bayesian models

- [x] GAN-trained discriminator, augmentation of training data with generated fakes, threshold for discriminator prediction (i.e. OpenGAN), with Standard Model, also combined with mixup

- [x] Class-specific autoencoders, threshold for reconstruction error (i.e. CSSR)

5_across_vantage_points: This is the code for studying the closed-world vs. open-world effectiveness of training and testing models across different geographic vantage points, i.e. where the training set was streamed from one vantage point and the test set was streamed from another.

6_video_lengths: This is the code for studying the closed-world vs. open-world performance of models as a function of video lengths or traffic flow lengths. The code for training and testing truncates all traffic flows in the us-west-2 Vimeo open-world splits to 0:20, 0:30, 0:40... 4:00 lengths.
