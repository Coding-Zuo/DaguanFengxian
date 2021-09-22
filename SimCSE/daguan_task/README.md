


### SimCSE framework

#### requirements

需要特别注意：

transformers==4.2.1



#### 训练


```bash



# 准备NLI数据

python src/SimCSE/daguan_task/prepare_nli_datasets.py --dataset_path datasets/phase_1/splits/fold_0 --nli_dataset_path datasets/phase_1/splits/fold_0_nli --sampling_times 100000

# 训练
cd src/SimCSE
CUDA_VISIBLE_DEVICES="1" nohup ./daguan_task/run_sup_example_0905.sh > ../../experiments/logs/simcse_framwork_0905.log &
CUDA_VISIBLE_DEVICES="1" nohup ./daguan_task/run_sup_example_0905.sh > ./logs/simcse_framwork_1.log &



```