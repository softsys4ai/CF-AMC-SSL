#!/bin/bash

> test_results_EMP_comparing_cifar100_cifar10_v3.txt


# Define arrays of parameter values
test_patches_values=("1" "32" "64")

model_path_values=("/home/Fatemeh/one-epoch-AT/logs/EMP-SSL-Training/patchsim200_numpatch16_bs50_lr0.3_NONE/save_models_adv_wo_Normalization_8_cifar100/29.pt" 
"/home/Fatemeh/one-epoch-AT/logs/EMP-SSL-Training/patchsim200_numpatch4_bs100_lr0.3_NONE/save_models_adv_wo_Normalization_8_cifar100_v2/29.pt")
                 
scale_min=0.25
scale_max=0.25
ratio_min=1.0
ratio_max=1.0

type="patch"


# Loop through the parameter combinations and run the code
for test_patches in "${test_patches_values[@]}"; do
    iteration=0
    for model_path in "${model_path_values[@]}"; do
            if [ "$iteration" -lt 2 ]; then
                data="cifar100"
            else
                data="cifar100"
            fi
            echo "Running with test_patches = $test_patches, model_path = $model_path, data = $tdata, scale_min = $scale_min, scale_max = $scale_max, ratio_min = $ratio_min, ratio_max = $ratio_max"
            python evaluate_EMPSSL.py --data "$data" --test_patches "$test_patches" --model_path "$model_path" --type "$type" --scale_min "$scale_min" --scale_max "$scale_max" --ratio_min "$ratio_min" --ratio_max "$ratio_max" >> test_results_EMP_comparing_cifar100_cifar10_v3.txt
            ((iteration++))

    done
done
echo test_results_EMP_comparing_cifar100_cifar10_v3.txt
