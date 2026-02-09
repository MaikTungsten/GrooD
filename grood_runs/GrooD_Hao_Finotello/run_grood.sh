#### No target during sim

echo "---------------------------"
echo 'Standard GrooD'
echo 'Hao no target pseudobulks in train and Finotello in test - CPM normalization'
echo "---------------------------"

# Print start date and time
echo 'Start date: '
date '+%B %V %T'
echo "---------------------------"
echo ""

# Parameters - fixed for all models
depth=4
n_estimators=500
lr=0.01
split=50
loss="squared_error"

# Fixed input params
scdata="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/data/references/Hao_2021/Hao_2_train.h5ad"
bulkprops="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/data/bulk_data/Finotello_2019/final_data/Finotello_FACS_proportions_no_neutrophils_others.csv"
bulk="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/data/bulk_data/Finotello_2019/final_data/Finotello_TPM_data.csv"

# Variable paths for test
output_dir="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/grood_runs/GrooD_Hao_Finotello/intersect_CPM/"

# Norm & feature curation
norm="CPM"
feature_curation="intersect"

## Training on pseudobulks simulated from sc data, direct inference on bulks
python grood.py --grood_mode grood --sc $scdata \
    --mode all --output $output_dir \
    --depth $depth --n_estimators $n_estimators --learning_rate $lr --min_samples_split $split --loss_function $loss \
    --threads 8 \
    --norm $norm --feature_curation $feature_curation \
    --bulk $bulk --props $bulkprops

echo "---------------------------"
echo 'Standard GrooD'
echo 'Hao no target pseudobulks in train and test - log normalization'
echo "---------------------------"

# Variable paths for test
output_dir="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/grood_runs/GrooD_Hao_Finotello/intersect_log/"

# Norm & feature curation
norm="log"
feature_curation="intersect"

## Training on pseudobulks simulated from sc data, direct inference on bulks
python grood.py --grood_mode grood --sc $scdata \
    --mode all --output $output_dir \
    --depth $depth --n_estimators $n_estimators --learning_rate $lr --min_samples_split $split --loss_function $loss \
    --threads 8 \
    --norm $norm --feature_curation $feature_curation \
    --bulk $bulk --props $bulkprops


echo "---------------------------"
echo 'Standard GrooD'
echo 'Hao no target pseudobulks in train and test - rank normalization'
echo "---------------------------"

# Variable paths for test
output_dir="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/grood_runs/GrooD_Hao_Finotello/intersect_rank/"

# Norm & feature curation
norm="rank"
feature_curation="intersect"

## Training on pseudobulks simulated from sc data, direct inference on bulks
python grood.py --grood_mode grood --sc $scdata \
    --mode all --output $output_dir \
    --depth $depth --n_estimators $n_estimators --learning_rate $lr --min_samples_split $split --loss_function $loss \
    --threads 8 \
    --norm $norm --feature_curation $feature_curation \
    --bulk $bulk --props $bulkprops


echo "---------------------------"
echo 'Standard GrooD'
echo 'Hao no target pseudobulks in train and test - CPM normalization, mRNA_intersect'
echo "---------------------------"

# Variable paths for test
output_dir="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/grood_runs/GrooD_Hao_Finotello/mRNA_intersect_CPM/"

# Norm & feature curation
norm="CPM"
feature_curation="mRNA_intersect"

## Training on pseudobulks simulated from sc data, direct inference on bulks
python grood.py --grood_mode grood --sc $scdata \
    --mode all --output $output_dir \
    --depth $depth --n_estimators $n_estimators --learning_rate $lr --min_samples_split $split --loss_function $loss \
    --threads 8 \
    --norm $norm --feature_curation $feature_curation \
    --bulk $bulk --props $bulkprops


echo "---------------------------"
echo 'Standard GrooD'
echo 'Hao no target pseudobulks in train and test - CPM normalization, non_zero_intersect'
echo "---------------------------"

# Variable paths for test
output_dir="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/grood_runs/GrooD_Hao_Finotello/non_zero_intersect_CPM/"

# Norm & feature curation
norm="CPM"
feature_curation="non_zero_intersect"

## Training on pseudobulks simulated from sc data, direct inference on bulks
python grood.py --grood_mode grood --sc $scdata \
    --mode all --output $output_dir \
    --depth $depth --n_estimators $n_estimators --learning_rate $lr --min_samples_split $split --loss_function $loss \
    --threads 8 \
    --norm $norm --feature_curation $feature_curation \
    --bulk $bulk --props $bulkprops


echo "---------------------------"
echo 'Standard GrooD'
echo 'Hao no target pseudobulks in train and test - CPM normalization, mRNA'
echo "---------------------------"

# Variable paths for test
output_dir="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/grood_runs/GrooD_Hao_Finotello/mRNA_CPM/"

# Norm & feature curation
norm="CPM"
feature_curation="mRNA"

## Training on pseudobulks simulated from sc data, direct inference on bulks
python grood.py --grood_mode grood --sc $scdata \
    --mode all --output $output_dir \
    --depth $depth --n_estimators $n_estimators --learning_rate $lr --min_samples_split $split --loss_function $loss \
    --threads 8 \
    --norm $norm --feature_curation $feature_curation \
    --bulk $bulk --props $bulkprops


echo "---------------------------"
echo 'Standard GrooD'
echo 'Hao no target pseudobulks in train and test - CPM normalization, all'
echo "---------------------------"

# Variable paths for test
output_dir="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/grood_runs/GrooD_Hao_Finotello/all_CPM/"

# Norm & feature curation
norm="CPM"
feature_curation="all"

## Training on pseudobulks simulated from sc data, direct inference on bulks
python grood.py --grood_mode grood --sc $scdata \
    --mode all --output $output_dir \
    --depth $depth --n_estimators $n_estimators --learning_rate $lr --min_samples_split $split --loss_function $loss \
    --threads 8 \
    --norm $norm --feature_curation $feature_curation \
    --bulk $bulk --props $bulkprops


echo "---------------------------"
echo 'Standard GrooD'
echo 'Hao no target pseudobulks in train and test - CPM normalization, non_zero'
echo "---------------------------"

# Variable paths for test
output_dir="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/grood_runs/GrooD_Hao_Finotello/non_zero_CPM/"

# Norm & feature curation
norm="CPM"
feature_curation="non_zero"

## Training on pseudobulks simulated from sc data, direct inference on bulks
python grood.py --grood_mode grood --sc $scdata \
    --mode all --output $output_dir \
    --depth $depth --n_estimators $n_estimators --learning_rate $lr --min_samples_split $split --loss_function $loss \
    --threads 8 \
    --norm $norm --feature_curation $feature_curation \
    --bulk $bulk --props $bulkprops


# Print start date and time
echo "---------------------------"
echo "GrooD tests done"
date '+%B %V %T'
echo "---------------------------"
echo ""
echo ""

