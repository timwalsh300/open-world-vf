0_raw_to_csv: Launch separate batch jobs by modifying the monitored_raw_to_csv.sh script for every combination of representation, protocol, and platform to turn the raw dataset into initial .csv files.

    * sirinam_wf, sirinam_vf, rahman, hayden, schuster2, schuster4, schuster8, dschuster8, schuster16, dschuster16 done

1_csv_to_pkl: From an interactive session with 128 GB of memory, run the csv_to_pkl.py script with no arguments to turn all the initial .csv files into intermediate .csv files and .pkl files with the closed-world train/val/test splits.

    * sirinam_wf, sirinam_vf, rahman, hayden, schuster2, schuster4, schuster8, dschuster8, schuster16, dschuster16 done

2_closed_world: Launch separate batch jobs by modifying the search.sh script for every combination of representation, protocol, and platform to find good hyperparameters with Ray Tune. Look at the tail of the .out files to see the best hyperparameters found and copy them into the dictionary in the evaluation.py script. Launch a batch job with evaluation.sh and look at the .out file to see the number of epochs, training time, and test set accuracy for each model.

    * Ray Tune search for dschuster16, schuster16, rschuster8, dschuster8, schuster8, schuster4, schuster2, hayden, rahman, sirinam_vf done

    * put all best found hyperparameters into evaluation.py, done

    * evaluation batch job for all done

3_open_world_baseline: 

    * 61-way classification task, best representation(s) only...

    * modify and run 0_raw_to_csv/unmonitored_raw_to_csv.sh script for best closed-world representations, done

    * set aside 64k unmonitored instances for testing... remaining 12k (Tor) to 14k (HTTPS) are split 85/15 for training and validation...

    * create train/val/test splits with csv_to_pkl_open.py for both HTTPS and Tor... 1k, 2k, 4k, 8k, 16k, 32k, 64k test sets... done

    * modify and run search_open.sh to do hyperparameter searches with Ray Tune

        * schuster8, dschuster8 done

    * evaluation_open.py outputs results for P, R, FP, FPR after choosing a threshold using the validation set

        - retrain with early stopping based on validation accuracy or F1

        - produce P-R and ROC curves over validation and test sets

        - select threshold based on minimum acceptable recall on validation set

    * evaluation_open_trials.sh independently samples train/val/test sets, trains a model, and evaluates the model five times to get results that we can average

4_open_world_enhancements:

    - temperature scaling, threshold for calibrated softmax probability tuned on the validation set, five independent trials

    - monte carlo dropout, threshold for epistemic uncertainty...

    - class-specific autoencoders, threshold for reconstruction error...

    - GAN-trained discriminator, threshold for discriminator prediction...

    - NOTA...
