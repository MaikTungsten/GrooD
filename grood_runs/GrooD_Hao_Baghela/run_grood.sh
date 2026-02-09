#### No target during sim

echo "---------------------------"
echo 'Standard GrooD'
echo 'Hao no target pseudobulks in train and test - CPM normalization'
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
bulk="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/data/bulk_data/Baghela_2020/TPM_data_Baghela.csv"

# Variable paths for test
output_dir="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/grood_runs/GrooD_Hao_Baghela/intersect_mRNA/"

# Norm & feature curation
norm="CPM"
feature_curation="mRNA_intersect"

## Training on pseudobulks simulated from sc data, direct inference on bulks
python grood.py --grood_mode grood --sc $scdata \
    --mode all --output $output_dir \
    --depth $depth --n_estimators $n_estimators --learning_rate $lr --min_samples_split $split --loss_function $loss \
    --threads 8 \
    --norm $norm --feature_curation $feature_curation \
    --bulk $bulk

# Variable paths for test
output_dir="/ceph/ibmi/it/projects/ML_BI/16_GrooD/GrooD/grood_runs/GrooD_Hao_Baghela/intersect/"

# Norm & feature curation
norm="CPM"
feature_curation="intersect"

## Training on pseudobulks simulated from sc data, direct inference on bulks
python grood.py --grood_mode grood --sc $scdata \
    --mode all --output $output_dir \
    --depth $depth --n_estimators $n_estimators --learning_rate $lr --min_samples_split $split --loss_function $loss \
    --threads 8 \
    --norm $norm --feature_curation $feature_curation \
    --bulk $bulk

