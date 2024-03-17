This repository contains the steps and code to reproduce the work for "Exploring the Capabilities and Limitations of Video Stream Fingerprinting" (to be presented at the SecWeb 2024 workshop) and my dissertation. It is a work-in-progress through at least mid-2025, and intended only for academic research purposes. For access to the raw dataset, please contact me at timothy.walsh@nps.edu. Due to the considerable size of the raw dataset, to make it available while minimizing the cost of storage, we are using the Amazon S3 Glacier Deep Archive. That requires some coordination to grant access, including a restore request that can take up to 48 hours on Amazon's end. The sister repository with the code (and lists of our URLs) for our dataset collection effort is https://github.com/timwalsh300/tor-browser-crawler-video 

0_raw_to_csv: Launch separate batch jobs by modifying the monitored_raw_to_csv.sh script for every combination of representation, protocol, and platform to turn the raw dataset into initial .csv files.

    * sirinam_wf, sirinam_vf, rahman, hayden, schuster2, schuster4, schuster8, dschuster8, schuster16, dschuster16 done

1_csv_to_pkl: From an interactive session with 128 GB of memory, run the csv_to_pkl.py script with no arguments to turn all the initial .csv files into intermediate .csv files and .pkl files with the closed-world train/val/test splits.

    * sirinam_wf, sirinam_vf, rahman, hayden, schuster2, schuster4, schuster8, dschuster8, schuster16, dschuster16 done

2_closed_world: Launch separate batch jobs by modifying the search.sh script for every combination of representation, protocol, and platform to find good hyperparameters with Ray Tune. Look at the tail of the .out files to see the best hyperparameters found and copy them into the dictionary in the evaluation.py script. Launch a batch job with evaluation.sh and look at the .out file to see the number of epochs, training time, and test set accuracy for each model.

    * Ray Tune search for dschuster16, schuster16, rschuster8, dschuster8, schuster8, schuster4, schuster2, hayden, rahman, sirinam_vf done

    * put all best found hyperparameters into evaluation.py, done

    * evaluation batch job for all done

3_open_world_baseline: 61-way classification task with the best representation(s) only. Modify and run the 0_raw_to_csv/unmonitored_raw_to_csv.sh script for best closed-world data representations. Create train/val/test splits with csv_to_pkl_open.py for both HTTPS and Tor. Modify and run search_open.sh to do hyperparameter searches with Ray Tune. Copy best found hyperparameters into train_open.py and run that to train and save the models. Output P-R curve figures and results for P, R, F1, FP, FPR on test sets 1k to 64k after choosing thresholds using the validation set with evaluation_open.py.

    * parsing unmonitored set, creating train/val/test sets, for schuster8 (Tor) and dschuster16 (HTTPS) done

    * hyperparameter search, training for dschuster16 (HTTPS), schuster8 (Tor) done

    * produce P-R curves over validation and test sets done

    * select thresholds based on zero FP and 0.5 R over the validation set done

4_open_world_enhancements: Experiment with a number of more sophisticated methods attempting to address the theoretical shortcomings of the baseline, existing approach to the open-world / open set recognition task. 

    * Temperature scaling, threshold for calibrated max softmax probability done

    * Monte Carlo Dropout, Spike-and-Slab Concrete Dropout, threshold for Bayesian model averaged max softmax probability, total uncertainty, and epistemic uncertainty, with Standard Model / Background Class training data augmentation done

    - NOTA / Mixup for monitored training data augmentation...

    - Class-specific autoencoders, threshold for reconstruction error...

    - GAN-trained discriminator, threshold for discriminator prediction...
