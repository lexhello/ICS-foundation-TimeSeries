import numpy as np
import torch
import timesfm
from utils import gen_utils, ctown_utils
from data_loader import load_train_data, get_loaders, TimeSeriesDataset, load_test_data
from nixtla import NixtlaClient
import eval_utils
# import model_utils
import json
import pandas as pd
from tqdm import tqdm
import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline
from eval_utils import run_times_series_test

#TODO: delete
import importlib
importlib.reload(eval_utils)

def load_model(model_name: str, cfg):
    
    # Minimal example: one id, 5 time points
    df = pd.DataFrame({
        "unique_id": ["series_1"] * 5,
        "id_col": ["series_1"] * 5,
        "time_col": pd.date_range("2023-01-01", periods=5, freq="D"),
        "ds": pd.date_range("2023-01-01", periods=5, freq="D"),
        "target_col": [10, 12, 14, 13, 15]
    })
    
    if model_name == "timesfm" or model_name == "TimesFM":
        torch.set_float32_matmul_precision("high")

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

        # point_forecast, quantile_forecast = model.forecast(
        #     horizon=3,
        #     inputs=[
        #         np.linspace(0, 1, 100),
        #         np.sin(np.linspace(0, 20, 67)),
        #     ],  # Two dummy inputs
        # )
        # print("POINT FORECAST")
        # print(point_forecast.shape)
        # print(point_forecast)
        
        return model
        
    if model_name == "Chronos" or model_name == "chronos":
        
        #TODO: change device here
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map="mps",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            dtype=torch.bfloat16,
        )
        return pipeline
        
        
    if model_name == "TimeGPT" or model_name == "timegpt":
        nixtla_client = NixtlaClient(
            api_key='nixak-9k81qSZlbVbia1yFOBZJzXdhjRMvbYCcAJUOOjFRJYqyQiVH0qEErndMQv1RggFn4jETetzc2AVeaO4d'
        )
        nixtla_client.validate_api_key()
        
        return nixtla_client
   
def load_test(model_name, dataset):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f'On device: {device}')

    # CHANGE THESE VALUES
    model_type = model_name
    run_number = 1

    do_train = False
    do_test = True
    evasion_type = "cons"
    batch_size = 1
    
    # training params
    train_params_epochs = 1
    no_transform = True

    print(f'{model_type} {dataset}: run #{run_number}')

    cfg = {
        'model_name': model_name,
        'batch_size': batch_size,
        'train_epochs': train_params_epochs
    }
    
    Xfull, Ytest, sensor_cols = load_test_data(dataset, no_transform=no_transform)
    
    # print("SENSOR COLS: ")
    # print(sensor_cols)
    node_num = len(sensor_cols)
    cfg['node_num'] = node_num

    # use this once it's set up
    # model_utils.load_model_params_from_args(args, cfg)
    # context length
    model_params_history = 32
    run_number = 1
    cfg.update({
        'history': model_params_history,
        'run_number': run_number,
        # 'gnn_topk': gnn_model_params_topk,
        # 'gcn_degree': args.gcn_model_params_degree,
        # 'gcn_weight_strategy': args.gcn_model_params_weight_strategy,
        
        # 'gcn_graph_spec': args.gcn_graph_spec,

        # Kept from AAAI '21 implementation
        # 'embedding_dim': 64,
        # 'out_layer_num': 1,
        # 'out_layer_inter_dim': 128,
        'slide_stride': 1,

    })
    
    model = load_model(model_name, cfg)
    
    # NOTE: Added a custom function, since the RICSS CTOWN dataset is stored across 52 training files
    # model_dir = 'best_models_50epoch'
    
    # NOTE: Added a custom function, since the RICSS CTOWN dataset is stored across 52 training files
    if dataset == 'CTOWN':
        model_dir = 'best_models_new_ctown'
        train_dataset = ctown_utils.CTownMultiWeekDataset(config=cfg)
        #train_dataset = TimeSeriesDataset(Xfull.T, np.zeros(len(Xfull)), mode='train', config=cfg)      
    else:
        model_dir = 'best_models_50epoch'
        train_dataset = TimeSeriesDataset(Xfull.T, np.zeros(len(Xfull)), mode='train', config=cfg)    
    # test_dataset = TimeSeriesDataset(Xfull.T, Ytest, mode='test', config=cfg)
    
    # TimeSeriesDataset(Xfull.T, np.zeros(len(Xfull)), mode='train', config=cfg)    
    
    # _, val_dataloader = get_loaders(test_dataset, cfg['batch_size'], val_ratio = 0.2)
    train_dataloader, val_dataloader = get_loaders(train_dataset, cfg['batch_size'], val_ratio = 0.2)

    print(f'Validation samples: {len(val_dataloader.dataset)}')
    print(f'Validation batches: {len(val_dataloader)}')
 

    if do_test:
        avg_loss, val_result = run_times_series_test(model_name, model, val_dataloader, cfg)
        print("average loss of ", model_name, "is: ")
        print(avg_loss)
        
        Xval_pred, Xval_true, Y_true = val_result
        # print("shape of Xval_pred:")
        # shape is 
        # print(Xval_pred.shape)
        
        # gets weird here
        all_saved_metrics, _ = eval_utils.run_detection_eval(device, model, cfg, dataset=dataset, Xval_true=Xval_true, Xval_pred=Xval_pred, detection_threshold=None, evasion_type=evasion_type)
        return
        # all_saved_metrics['true-timing-avgrank'] = final_first_timing_avgrank # Keep for legacy plots
        final_roc_auc = all_saved_metrics['mse-roc-auc']
        final_cusum_roc_auc = all_saved_metrics['cusum-roc-auc']
        
        # eval_utils.run_attribution_eval_detections(device, model, cfg, all_saved_metrics, dataset=dataset, evasion_type=evasion_type)
 
        print('==' * 20)
        
        if evasion_type == 'cons':
            print(f'Final stats for {model_name}. ROC AUC {final_roc_auc:.3f} CUSUM ROC AUC {final_cusum_roc_auc:.3f} AvgRank: {final_first_timing_avgrank:.3f}')
            # json_filename = f'{model_name}_ta_metrics.json'
        
        """
        
        # else:
        #     print(f'Final stats for {evasion_type}_{model_name}. ROC AUC {final_roc_auc:.3f} CUSUM ROC AUC {final_cusum_roc_auc:.3f} AvgRank: {final_first_timing_avgrank:.3f}')
        #     json_filename = f'{evasion_type}_{model_name}_ta_metrics.json'

        print('==' * 20)

        with open(json_filename, 'w') as file:
            print(f'Writing {json_filename}')
            json.dump(all_saved_metrics, file, indent=4)
            
        """
    
    print('Done')

if __name__ == "__main__":
    # load_model("TimeGPT")
    # load_model("timesfm")

    # model_names = ["timesfm", "chronos"]
    model_names = ["timesfm"]
    dataset = "SWAT"
    
    for model_name in model_names:
        load_test(model_name=model_name, dataset=dataset)
    
    