######## Multi-GPU training example #######
python train.py --name colon2depth_512p --batchSize 8 --gpu_ids 1,2 --label_nc 0 --no_instance --tf_log --no_vgg_loss --continue_train