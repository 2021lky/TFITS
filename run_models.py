import argparse
import math
import warnings
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
import h5py
import torch.nn as nn

from modeling.TFITS import TFITS
from baselines import iTransformer, TimesNet, DLinear, TFDNet

from unified_dataloader import UnifiedDataLoader
from utils import *

RANDOM_SEED = 26
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
warnings.filterwarnings('ignore')

MODEL_DICT = {
    'TFITS': TFITS,
    'iTransformer': iTransformer.Model,
    'TimesNet': TimesNet.Model,
    'DLinear': DLinear.Model,
    'TFDNet': TFDNet.Model,
}

OPTIMIZER = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}


# 合并参数
def read_arguments(arg_parser, cfg_parser):
    # file path
    arg_parser.dataset_base_dir = cfg_parser.get('file_path', 'dataset_base_dir')   # 数据集目录
    arg_parser.result_saving_base_dir = cfg_parser.get('file_path', 'result_saving_base_dir')  # 结果存储目录
    # dataset info
    arg_parser.seq_len = cfg_parser.getint('dataset', 'seq_len')  # 窗口大小
    arg_parser.batch_size = cfg_parser.getint('dataset', 'batch_size')
    arg_parser.num_workers = cfg_parser.getint('dataset', 'num_workers')
    arg_parser.feature_num = cfg_parser.getint('dataset', 'feature_num')  # 特征维度大小
    arg_parser.dataset_name = cfg_parser.get('dataset', 'dataset_name')  # 数据集名称
    # 获得数据集路径，数据集就在该目录下面的datasets.h5目录下
    arg_parser.dataset_path = os.path.join(arg_parser.dataset_base_dir, arg_parser.dataset_name)
    arg_parser.eval_every_n_steps = cfg_parser.getint('dataset', 'eval_every_n_steps')

    # training settings
    arg_parser.artificial_missing_rate = cfg_parser.getfloat('training', 'artificial_missing_rate')  # 人工设置缺失率
    arg_parser.lr = cfg_parser.getfloat('training', 'lr')  # 学习率
    arg_parser.optimizer_type = cfg_parser.get('training', 'optimizer_type')  # 优化器
    arg_parser.weight_decay = cfg_parser.getfloat('training', 'weight_decay')
    arg_parser.device = cfg_parser.get('training', 'device')
    arg_parser.epochs = cfg_parser.getint('training', 'epochs')
    arg_parser.early_stop_patience = cfg_parser.getint('training', 'early_stop_patience')  # 耐心
    arg_parser.norm = cfg_parser.getboolean('training', 'norm')  # 是否标准化输入
    arg_parser.step = cfg_parser.getint('training', 'step')
    arg_parser.gamma = cfg_parser.getfloat('training', 'gamma')


    # model settings
    arg_parser.model_type = cfg_parser.get('model', 'model_type')
    arg_parser.input_with_mask = cfg_parser.getboolean('model', 'input_with_mask')  # 输入是否合并mask
    arg_parser.n_layer = cfg_parser.getint('model', 'n_layer')
    arg_parser.d_model = cfg_parser.getint('model', 'd_model')
    arg_parser.d_inner = cfg_parser.getint('model', 'd_inner')
    arg_parser.n_head = cfg_parser.getint('model', 'n_head')
    arg_parser.d_k = cfg_parser.getint('model', 'd_k')
    arg_parser.d_v = cfg_parser.getint('model', 'd_v')
    arg_parser.dropout = cfg_parser.getfloat('model', 'dropout')
    arg_parser.diagonal_attention_mask = cfg_parser.getboolean('model', 'diagonal_attention_mask')  # 是否使用对角线掩码

    if arg_parser.model_type in {'TFITS'}:
        arg_parser.d_feature_model = cfg_parser.getint('model', 'd_feature_model')
        arg_parser.d_feature_inner = cfg_parser.getint('model', 'd_feature_inner')
        arg_parser.d_feature_k = cfg_parser.getint('model', 'd_feature_k')
        arg_parser.n_feature_head = cfg_parser.getint('model', 'n_feature_head')
        arg_parser.d_feature_v = cfg_parser.getint('model', 'd_feature_v')
        arg_parser.n_feature_layer = cfg_parser.getint('model', 'n_feature_layer')

    if arg_parser.model_type == 'TimesNet':
        arg_parser.top_k = cfg_parser.getint('model', 'top_k')
        arg_parser.num_kernels = cfg_parser.getint('model', 'num_kernels')
    if arg_parser.model_type == 'DLinear':
        arg_parser.moving_avg = cfg_parser.getint('model', 'moving_avg')  # 趋势季节分解的移动平均窗口大小
        arg_parser.channel_independence = cfg_parser.getboolean('model', 'channel_independence')  # 是否通道独立
    if arg_parser.model_type == 'PatchTST':
        arg_parser.patch_len = cfg_parser.getint('model', 'patch_len')  # 补丁长度
        arg_parser.stride = cfg_parser.getint('model', 'stride')  # 步长
    
    if arg_parser.model_type == 'TimeMixer':
        arg_parser.down_sampling_method = cfg_parser.get('model', 'down_sampling_method')  # avg
        arg_parser.down_sampling_layers = cfg_parser.getint('model', 'down_sampling_layers')  # 3
        arg_parser.down_sampling_window = cfg_parser.getint('model', 'down_sampling_window')  # 2
        arg_parser.channel_independence = cfg_parser.getboolean('model', 'channel_independence')  # 是否通道独立
        arg_parser.moving_avg = cfg_parser.getint('model', 'moving_avg')  # 趋势季节分解的移动平均窗口大小

    if arg_parser.model_type == 'TFDNet':
        arg_parser.n_fft = [int(x.strip()) for x in cfg_parser.get('model', 'n_fft').split(',')]
        arg_parser.kernel_num = cfg_parser.getint('model', 'kernel_num')
        arg_parser.kernel_size = cfg_parser.getint('model', 'kernel_size')
        # 在 IK（Individual Kernel）模式下，定义每个通道或特征维度的独立卷积核数量
        arg_parser.individual_factor = cfg_parser.getint('model', 'individual_factor')
        arg_parser.mode = cfg_parser.get('model', 'mode')


    # test
    arg_parser.model_path = cfg_parser.get('test', 'model_path')
    arg_parser.result_saving_path = cfg_parser.get('test', 'result_saving_path')
    arg_parser.save_imputations = cfg_parser.getboolean('test', 'save_imputations')

    return arg_parser


def result_processing(results):
    """process results and losses for each training step"""
    results['total_loss'] = results['reconstruction_loss'] + results['imputation_loss']  # 总损失
    return results


# 处理训练过程中的后续操作，判断是否需要早停
def process_each_training_step(results, optimizer, scheduler, val_dataloader, training_controller, logger):
    """process each training step and return whether to early stop"""
    state_dict = training_controller(stage='train')
    # 是否应用梯度裁剪 if args.max_norm != 0
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=True)
    results['total_loss'].backward()

    optimizer.step()
    scheduler.step()

    if state_dict['train_step'] % args.eval_every_n_steps == 0:
        state_dict_from_val = validate(model, val_dataloader, training_controller, logger)
        if state_dict_from_val['should_stop']:
            logger.info(f'Early stopping worked, stop now...')
            return True
    return False


def model_processing(data, model, stage,
                     optimizer=None, scheduler=None, val_dataloader=None, training_controller=None, logger=None):
    if stage == 'train':
        optimizer.zero_grad()

        indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
        inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask,
                  'X_holdout': X_holdout, 'indicating_mask': indicating_mask}
        results = result_processing(model(inputs, stage))
        early_stopping = process_each_training_step(results, optimizer, scheduler, val_dataloader,
                                                        training_controller, logger)
        return early_stopping
    else:  # in val/test stage
        indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
        inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask,
                  'X_holdout': X_holdout, 'indicating_mask': indicating_mask}
        results = model(inputs, stage)
        results = result_processing(results)
        return inputs, results


def train(model, optimizer, scheduler, train_dataloader, test_dataloader, training_controller, logger):
    for epoch in range(args.epochs):
        early_stopping = False
        args.final_epoch = True if epoch == args.epochs - 1 else False
        for idx, data in enumerate(train_dataloader):
            model.train()
            early_stopping = model_processing(data, model, 'train', optimizer, scheduler, test_dataloader,
                                              training_controller, logger)
            if early_stopping:
                break
        if early_stopping:
            break
        training_controller.epoch_num_plus_1()
    logger.info('Finished all epochs. Stop training now.')


def validate(model, val_iter, training_controller, logger):
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    total_loss_collector, imputation_loss_collector, reconstruction_loss_collector, reconstruction_MAE_collector = [], [], [], []

    with torch.no_grad():
        for idx, data in enumerate(val_iter):
            inputs, results = model_processing(data, model, 'val')
            evalX_collector.append(inputs['X_holdout'])
            evalMask_collector.append(inputs['indicating_mask'])
            imputations_collector.append(results['imputed_data'])

            total_loss_collector.append(results['total_loss'].data.cpu().numpy())
            reconstruction_MAE_collector.append(results['reconstruction_MAE'].data.cpu().numpy())
            reconstruction_loss_collector.append(results['reconstruction_loss'].data.cpu().numpy())
            imputation_loss_collector.append(results['imputation_loss'].data.cpu().numpy())

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(imputations_collector, evalX_collector, evalMask_collector)
    info_dict = {'total_loss': np.asarray(total_loss_collector).mean(),
                 'reconstruction_loss': np.asarray(reconstruction_loss_collector).mean(),
                 'imputation_loss': np.asarray(imputation_loss_collector).mean(),
                 'reconstruction_MAE': np.asarray(reconstruction_MAE_collector).mean(),
                 'imputation_MAE': imputation_MAE.cpu().numpy().mean()}
    state_dict = training_controller('val', info_dict, logger)

    if state_dict['save_model']:
        saving_path = os.path.join(args.model_saving, 'checkpoints.ckpt')
        save_model(model, optimizer, state_dict, args, saving_path)
        logger.info(f'Saved model -> {saving_path}')
    return state_dict


def test(model, test_dataloader):
    logger.info(f'Start evaluating on whole test set...')
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            inputs, results = model_processing(data, model, 'test')
            # collect X_holdout, indicating_mask and imputed data
            evalX_collector.append(inputs['X_holdout'])
            evalMask_collector.append(inputs['indicating_mask'])
            imputations_collector.append(results['imputed_data'])

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_RMSE = masked_rmse_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_MRE = masked_mre_cal(imputations_collector, evalX_collector, evalMask_collector)

    assessment_metrics = {'imputation_MAE on the test set': imputation_MAE,
                          'imputation_RMSE on the test set': imputation_RMSE,
                          'imputation_MRE on the test set': imputation_MRE,
                          'trainable parameter num': args.total_params}
    with open(os.path.join(args.result_saving_path, 'overall_performance_metrics.out'), 'w') as f:
        logger.info('Overall performance metrics are listed as follows:')
        for k, v in assessment_metrics.items():
            logger.info(f'{k}: {v}')
            f.write(k + ':' + str(v))
            f.write('\n')


def impute_all_missing_data(model, test_data):
    logger.info(f'Start imputing all missing data in test sets...')
    model.eval()
    with torch.no_grad():

        indices_collector, imputations_collector = [], []  # 收集索引，以及填补后的值

        for idx, data in enumerate(test_data):
            indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
            inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask,
                      'X_holdout': X_holdout, 'indicating_mask': indicating_mask}
            imputed_data, _ = model.impute(inputs)
            indices_collector.append(indices)
            imputations_collector.append(imputed_data)

        indices_collector = torch.cat(indices_collector)
        indices = indices_collector.cpu().numpy().reshape(-1)
        imputations_collector = torch.cat(imputations_collector)
        imputations = imputations_collector.data.cpu().numpy()
        ordered = imputations[np.argsort(indices)]  # to ensure the order of samples


    imputation_saving_path = os.path.join(args.result_saving_path, 'imputations.h5')
    with h5py.File(imputation_saving_path, 'w') as hf:
        hf.create_dataset('imputed_test_set', data=ordered)
    logger.info(f'Done saving all imputed data into {imputation_saving_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path of config file', default='configs/test.ini')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true', help='test mode to test saved model')

    args = parser.parse_args()

    assert os.path.exists(args.config_path), f'Given config file "{args.config_path}" does not exists'
    # 加载配置文件
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config_path)
    args = read_arguments(args, cfg)  # 合并配置参数

    args.model_name = "{}_{}".format(args.model_type, args.dataset_name)

    assert args.optimizer_type in OPTIMIZER.keys(), \
        f'optimizer type should be in {OPTIMIZER.keys()}, but get{args.optimizer_type}'

    time_now = datetime.now().__format__("%Y-%m-%d_T%H_%M_%S")
    if not torch.cuda.is_available():
        args.device = 'cpu'

    args.model_saving = check_saving_dir_for_model(args, time_now)  # 检查点保存位置
    logger = setup_logger('run')
    logger.info(f'args: {args}')
    logger.info(f'Config file path: {args.config_path}')
    logger.info(f'Config model name: {args.model_name}')

    unified_dataloader = UnifiedDataLoader(args.dataset_path, args.seq_len, args.feature_num,
                                           args.batch_size, args.num_workers, args.artificial_missing_rate)

    model = MODEL_DICT[args.model_type](args)

    # 统计参数量
    args.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Num of total trainable params is: {args.total_params}')

    # device
    if 'cuda' in args.device and torch.cuda.is_available():
        model = model.to(args.device)

    if args.test_mode:

        os.makedirs(args.result_saving_path) if not os.path.exists(args.result_saving_path) else None
        model = load_model(model, args.model_path, logger)

        test_dataloader = unified_dataloader.get_test_dataloader()
        test(model, test_dataloader)

        # 获得还原后的插补数据
        if args.save_imputations:
            test_data = unified_dataloader.prepare_all_data_for_imputation()
            impute_all_missing_data(model, test_data)
    else: 
        logger.info(f'Creating {args.optimizer_type} optimizer...')

        optimizer = OPTIMIZER[args.optimizer_type](model.parameters(), lr=args.lr,
                                                   weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        logger.info('Entering training mode...')
        train_dataloader, val_dataloader = unified_dataloader.get_train_val_dataloader()
        training_controller = Controller(args.early_stop_patience)  # 训练控制器

        train_set_size = unified_dataloader.train_set_size
        logger.info(f'train set len is {train_set_size}, batch size is {args.batch_size},'
                    f'so each epoch has {math.ceil(train_set_size / args.batch_size)} steps')

        train(model, optimizer, scheduler, train_dataloader, val_dataloader, training_controller, logger)

        # test
        logger.info('Entering testing mode...')
        logger.info(f'Config model path: {args.model_path}')
        args.model_path = os.path.join(args.model_saving, 'checkpoints.ckpt')
        model = load_model(model, args.model_path, logger)
        test_dataloader = unified_dataloader.get_test_dataloader()
        test(model, test_dataloader)


    logger.info('All Done.')
