
cfg = {

    ### Global Set
    'GPU_ID': '0',
    'random_seed':42,

    ### Model Setting
    "model_name": "timm:efficientnet_b0",  
    'pretrained':'', #local path or '' or 'default'
    
    "num_classes": 10,
    "class_names": [], #str in list or [] for default DIR name

    "head_type": "cls",  #cls, reg, det, seg

    "num_workers":4,


    ### Train Setting
    'train_dir':"./data/train",
    'val_dir':"./data/val", #if '' mean use k_flod
    "k_flod":5,
    'val_fold':0,


    "save_dir": "output/",
    "loss": 'CE', 

    ### Train Hyperparameters
    "img_size": [224, 224], # [w, h]
    'learning_rate':0.001,
    'batch_size':64,
    'epochs':100,
    'optimizer':'AdamW',  #Adam  SGD 
    'scheduler':'default-0.1-3', #default  SGDR-5-2    step-4-0.8

    'warmup_epoch':0, 
    'weight_decay' : 0,
    
    'early_stop_patient':7,
    'class_weight': [],

    'clip_gradient': 0,#1,     


    ### Test
    'model_path':'output/exp41/best.pt',

    'test_path':"./data/test",#test without label, just show img result

}
