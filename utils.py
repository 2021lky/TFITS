import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

plt.rcParams['savefig.dpi'] = 300  # pixel
plt.rcParams['figure.dpi'] = 300  # resolution
plt.rcParams["figure.figsize"] = [8, 4]  # figure size


def masked_mae_cal(inputs, target, mask):
    """计算平均绝对误差"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_mse_cal(inputs, target, mask):
    """计算均方误差"""
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    """计算均方根误差"""
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    """计算平均相对误差"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(torch.abs(target * mask)) + 1e-9)


def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true=y_test, probas_pred=y_pred)
    area = metrics.auc(recalls, precisions)
    return area, precisions, recalls, thresholds


def auc_roc(y_pred, y_test):
    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds


def auc_to_recall(recalls, precisions, recall=0.01):
    precisions_mod = precisions.copy()
    ind = np.where(recalls < recall)[0][0] + 1
    precisions_mod[:ind] = 0
    area = metrics.auc(recalls, precisions_mod)
    return area

def cal_classification_metrics(probabilities, labels, pos_label=1, class_num=1):
    """
    pos_label: 正类别的标签。
    """
    if class_num == 1:
        class_predictions = (probabilities >= 0.5).astype(int)
    elif class_num == 2:
        class_predictions = np.argmax(probabilities, axis=1)
    else:
        assert 'args.class_num>2, class need to be specified for precision_recall_fscore_support'
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, class_predictions,
                                                                       pos_label=pos_label, warn_for=())
    precision, recall, f1 = precision[1], recall[1], f1[1]
    precisions, recalls, _ = metrics.precision_recall_curve(labels, probabilities[:, -1], pos_label=pos_label)
    acc_score = metrics.accuracy_score(labels, class_predictions)
    ROC_AUC, fprs, tprs, thresholds = auc_roc(probabilities[:, -1], labels)
    PR_AUC = metrics.auc(recalls, precisions)
    classification_metrics = {
        'classification_predictions': class_predictions,
        'acc_score': acc_score, 'precision': precision, 'recall': recall, 'f1': f1,
        'precisions': precisions, 'recalls': recalls, 'fprs': fprs, 'tprs': tprs,
        'ROC_AUC': ROC_AUC, 'PR_AUC': PR_AUC,
    }
    return classification_metrics
def plot_AUCs(pdf_file, x_values, y_values, auc_value, title, x_name, y_name, dataset_name):
    """绘制AUC曲线"""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, '.', label=f'{dataset_name}, AUC={auc_value:.3f}', rasterized=True)
    l = ax.legend(fontsize=10, loc='lower left')
    l.set_zorder(20)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title, fontsize=12)
    pdf_file.savefig(fig)


def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


def setup_logger(log_name):
    """设置日志文件"""
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.propagate = False  # 防止子记录器将日志传播到根记录器（两次），不是必需的
    return logger

class Controller:
    def __init__(self, early_stop_patience):
        # early_stop_patience耐心值，如果在连续的50个训练周期里（epoch） ，如果模型的性能没有得到提升，就会停止训练。
        self.original_early_stop_patience_value = early_stop_patience
        self.early_stop_patience = early_stop_patience
        self.state_dict = {
            # `step`用于训练阶段
            'train_step': 0,
            # 以下用于验证阶段
            'val_step': 0,
            'epoch': 0,
            'best_imputation_MAE': 1e9,  # 记录最好的插补损失
            'should_stop': False,  # 用于控制停止训练
            'save_model': False  # 用于控制保存模型
        }

    def epoch_num_plus_1(self):
        self.state_dict['epoch'] += 1

    def __call__(self, stage, info=None, logger=None):
        if stage == 'train':
            self.state_dict['train_step'] += 1
        else:
            self.state_dict['val_step'] += 1
            self.state_dict['save_model'] = False
            current_imputation_MAE = info['imputation_MAE']
            imputation_MAE_dropped = False  # 用于降低早停耐心的标志

            # 更新最佳损失
            if current_imputation_MAE < self.state_dict['best_imputation_MAE']:
                logger.info(f'best_imputation_MAE已更新为 {current_imputation_MAE}')
                self.state_dict['best_imputation_MAE'] = current_imputation_MAE
                imputation_MAE_dropped = True
            if imputation_MAE_dropped:
                self.state_dict['save_model'] = True

            if self.state_dict['save_model']:
                self.early_stop_patience = self.original_early_stop_patience_value
            else:
                # 如果使用了早停，更新其耐心
                if self.early_stop_patience > 0:
                    self.early_stop_patience -= 1
                elif self.early_stop_patience == 0:
                    logger.info('早停耐心已耗尽，现在停止训练')
                    self.state_dict['should_stop'] = True  # 停止训练过程
                else:
                    pass  # 这意味着早停耐心值设为-1，不起作用

        return self.state_dict


def check_saving_dir_for_model(args, time_now):
    saving_path = os.path.join(args.result_saving_base_dir, args.model_name)
    if not args.test_mode:
        model_saving = os.path.join(saving_path, 'models')
        sub_model_saving = os.path.join(model_saving, time_now)
        [os.makedirs(dir_) for dir_ in [model_saving, sub_model_saving] if not os.path.exists(dir_)]
        return sub_model_saving
    else:
        return args.model_path


def save_model(model, optimizer, model_state_info, args, saving_path):
    """保存模型"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(), # 不保存优化器，考虑到GAN有两个优化器
        'training_step': model_state_info['train_step'],
        'epoch': model_state_info['epoch'],
        'model_state_info': model_state_info,
        'args': args
    }
    torch.save(checkpoint, saving_path)


def load_model(model, checkpoint_path, logger):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info(f"已从检查点 {checkpoint_path} 恢复模型")
    return model


def load_model_saved_with_module(model, checkpoint_path, logger):
    """
    加载并行训练并保存的模型（需要删除 'module.'）
    """
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = dict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:]  # 删除 'module.'
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    logger.info(f"已从检查点 {checkpoint_path} 恢复模型")
    return model