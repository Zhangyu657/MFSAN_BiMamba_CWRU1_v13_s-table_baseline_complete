import math
import torch
import logging
import importlib
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data.dataset import ConcatDataset

import utils


class TrainerBase(object):
    
    def __init__(self, args):
        self.args = args
        if args.cuda_device:
            self.device = torch.device("cuda:" + args.cuda_device)
            logging.info('using {} / {} gpus'.format(len(args.cuda_device.split(',')), torch.cuda.device_count()))
        else:
            self.device = torch.device("cpu")
            logging.info('using cpu')
        if args.train_mode == 'source_combine':
            self.num_source = 1
        else:
            self.num_source = len(args.source_name)

    
    def _get_lr_scheduler(self, optimizer):
        '''
        Get learning rate scheduler for optimizer.
        '''
        args = self.args
        assert args.lr_scheduler in ['step', 'exp', 'stepLR', 'fix'], f"lr scheduler should be 'step', 'exp', 'stepLR' or 'fix', but got {args.lr_scheduler}"
        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            lr_scheduler = None
        return lr_scheduler
    
    
    def _get_optimizer(self, model):
        '''
        Get optimizer for model.
        '''
        args = self.args
        if type(model) == list:
            par =  [{'params': md.parameters()} for md in model]
        else:
            par = model.parameters()
        
        # Define the optimizer
        assert args.opt in ['sgd', 'adam'], f"optimizer should be 'sgd' or 'adam', but got {args.opt}"
        if args.opt == 'sgd':
            optimizer = optim.SGD(par, lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            optimizer = optim.Adam(par, lr=args.lr, betas=args.betas,
                                   weight_decay=args.weight_decay)
        return optimizer
    
    
    def _get_tradeoff(self, tradeoff_list, epoch=None):
        '''
        Get trade-off parameters for loss.
        '''
        tradeoff = []
        for item in tradeoff_list:
            if item == 'exp':
                tradeoff.append(2 / (1 + math.exp(-self.args.zeta * (epoch-1) / max(self.args.max_epoch-1, 1))) - 1)
            elif type(item) == float or type(item) == int:
                tradeoff.append(item)
            else:
                raise Exception(f"unknown trade-off type {item}")
        return tradeoff
    
    
    def _get_actual_label(self, labels, idx=None, label_set=None):
        if idx is not None:
            label_set = self.args.label_sets[idx]
        else: assert label_set is not None
        actual_labels = []
        if len(labels.size()) > 1:
            labels = labels.argmax(dim=1)
        for label in labels.cpu():
            actual_labels.append(label_set[label])
        return torch.tensor(actual_labels).to(labels.device)
    
        
    def _get_train_label(self, labels, idx=None, label_set=None):
        if idx is not None:
            label_set = self.args.label_sets[idx]
        else: assert label_set is not None
        train_labels = []
        for label in labels.cpu():
            train_labels.append(label_set.index(label))
        return torch.tensor(train_labels).to(labels.device)
    

    def _combine_prediction(self, preds, idx, weights=None):
        # Ensure weights is not None and has the same length as labels and idx
        if weights is None:
            weights = [1.0] * len(preds)
        assert len(preds) == len(idx) == len(weights), "labels, idx, and weights must have the same length"
        
        batch_size = preds[0].size(0)
        # Dictionary to accumulate weighted predictions for each class
        class_predictions = defaultdict(lambda: torch.zeros(batch_size, dtype=torch.float, device=preds[0].device))
        class_weights = defaultdict(lambda: torch.zeros(1, dtype=torch.float, device=preds[0].device))

        # Process each source domain's predictions
        for label, domain_idx, weight in zip(preds, idx, weights):
            label_set = self.args.label_sets[domain_idx]
            
            # Calculate the weighted sum of predictions for each class
            for class_index in range(label.size(1)):
                actual_class_label = label_set[class_index]
                class_predictions[actual_class_label] += label[:, class_index] * weight
                class_weights[actual_class_label] += weight

        # Convert class_predictions to a tensor of shape (batch_size, num_classes)
        unique_classes = sorted(class_predictions.keys())
        predictions_tensor = torch.stack([class_predictions[cls]/class_weights[cls] if class_weights[cls] > 0 else class_predictions[cls] \
                                          for cls in unique_classes], dim=1)

        # Take the argmax to get the most confident predictions
        predicted_indices = predictions_tensor.argmax(dim=1)
        
        # Map indices back to actual labels
        predicted_labels = [unique_classes[idx.item()] for idx in predicted_indices]
        predicted_labels_tensor = torch.tensor(predicted_labels, device=preds[0].device)
        return predicted_labels_tensor
    
    
    def _get_accuracy(self, preds, targets, return_acc=True, idx=None, mode='normal'):
        assert preds.shape[0] == targets.shape[0]
        if len(preds.size()) > 1:
            preds = preds.argmax(dim=1)
        total = preds.shape[0]
        if mode != "normal":
            if isinstance(idx, list):
                # Combine all self.args.label_sets with the indices in idx
                label_set = torch.cat([torch.tensor(self.args.label_sets[i]) for i in idx])
            else:
                label_set = torch.tensor(self.args.label_sets[idx])
            targets = torch.where(torch.isin(targets, label_set), targets, -1)
        if mode == "closed-set":
            unknown_num = torch.sum(targets == -1).item()
            total -= unknown_num
        correct = torch.eq(preds.cpu(), targets.cpu()).float().sum().item()
        if return_acc:
            accuracy = correct/total if total > 0 else 0
            return accuracy
        else:
            return correct, total
    
    
    def _get_next_batch(self, src, return_actual=False):
        try:
            inputs, actual_labels = next(self.iters[src])
        except StopIteration:
            self.iters[src] = iter(self.dataloaders[src])
            inputs, actual_labels = next(self.iters[src])
        
        if return_actual:
            output = [inputs, actual_labels]
        else:
            src_idx = self.dataset_keys.index(src)
            if src in ['train', 'val']:
                src_idx = -1
            output = [inputs, self._get_train_label(actual_labels, src_idx)]
        output = [item.to(self.device) for item in output]
        return output
    
    
    def _init_data(self):
        '''
        Initialize the datasets.
        '''
        args = self.args
        
        self.datasets = {}
        for i, source in enumerate(args.source_name):
            dataset, condition, _ = utils.get_info_from_name(source)
            if condition is not None:
                Dataset = importlib.import_module("data_loader.conditional_load").dataset
                self.datasets[source] = Dataset(
                    args, dataset, i+1 if args.train_mode == 'source_combine' else i,
                    condition=condition,
                    balance_data=bool(getattr(args, 'source_balance_data', False)),
                ).data_preprare(is_src=True)
            else:
                Dataset = importlib.import_module("data_loader.load").dataset
                self.datasets[source] = Dataset(
                    args, dataset, i+1 if args.train_mode == 'source_combine' else i,
                    balance_data=bool(getattr(args, 'source_balance_data', False)),
                ).data_preprare(is_src=True)
        for key in self.datasets.keys():
            logging.info('Source set {} number of samples: {}.'.format(key, len(self.datasets[key])))
            self.datasets[key].summary()
        
        dataset, condition, _ = utils.get_info_from_name(args.target)
        if condition is not None:
            Dataset = importlib.import_module("data_loader.conditional_load").dataset
            self.datasets['train'], self.datasets['val'] = Dataset(args, dataset, -1, condition=condition).data_preprare(is_src=False)
        else:
            Dataset = importlib.import_module("data_loader.load").dataset
            self.datasets['train'], self.datasets['val'] = Dataset(args, dataset, -1).data_preprare(is_src=False)           
        logging.info('Target train set number of samples: {}.'.format(len(self.datasets['train'])))
        self.datasets['train'].summary()
        logging.info('Target test set number of samples: {}.'.format(len(self.datasets['val'])))
        self.datasets['val'].summary()
        
        if args.train_mode == 'source_combine':
            self.datasets['concat_source'] = ConcatDataset([self.datasets[s] for s in args.source_name])
            self.dataset_keys = ['concat_source', 'train', 'val']
        else:
            self.dataset_keys = args.source_name + ['train', 'val']

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                              batch_size=args.batch_size,
                                              shuffle=(False if x == 'val' else True),
                                              num_workers=args.num_workers,
                                              drop_last=(False if x == 'val' else True),
                                              pin_memory=(True if self.device == 'cuda' else False))
                                              for x in self.dataset_keys}
        self.iters = {x: iter(self.dataloaders[x]) for x in self.dataset_keys}

    def _train_one_epoch(self):
        raise NotImplementedError("Subclasses should implement '_train_one_epoch' method")
    
    def _set_to_train(self):
        raise NotImplementedError("Subclasses should implement '_set_to_train' method")
        
    def _eval(self, data, actual_labels, correct, total):
        raise NotImplementedError("Subclasses should implement '_eval' method")
        
    def _set_to_eval(self):
        raise NotImplementedError("Subclasses should implement '_set_to_eval' method")
    
    def _log_epoch_info(self, epoch_loss, epoch_acc, num_iter):
        # Print the train and val information via each epoch
        for key in epoch_loss.keys():
            logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
        for key in epoch_acc.keys():
            logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key]/num_iter))

    def _checkpoint_selection_score(self):
        """Return the user-selected target-test checkpoint score.

        The original project selected only by target-test accuracy.  V11 keeps
        that behavior as the default, while allowing a class-aware score for
        diagnosing the CWRU_1 / ball_21 collapse.
        """
        metrics = getattr(self, 'last_eval_metrics', {}) or {}
        acc = float(metrics.get('accuracy', 0.0))
        macro_f1 = float(metrics.get('macro_f1', acc))
        mode = str(getattr(self.args, 'best_metric', 'accuracy')).lower()

        if mode == 'macro_f1':
            return macro_f1
        if mode != 'class_aware':
            return acc

        focus_class = int(getattr(self.args, 'best_focus_class', -1))
        per_class_recall = metrics.get('per_class_recall', {}) or {}
        focus_recall = float(per_class_recall.get(focus_class, 0.0))

        wa = max(float(getattr(self.args, 'best_accuracy_weight', 0.45)), 0.0)
        wf = max(float(getattr(self.args, 'best_macro_f1_weight', 0.35)), 0.0)
        wr = max(float(getattr(self.args, 'best_focus_recall_weight', 0.20)), 0.0)
        denom = wa + wf + wr
        if denom <= 0.0:
            return acc
        return (wa * acc + wf * macro_f1 + wr * focus_recall) / denom

    def train(self):
        args = self.args
        best_score = float('-inf')
        best_acc = 0.0
        best_epoch = 0

        for epoch in range(1, args.max_epoch + 1):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            epoch_acc = defaultdict(float)
            self._set_to_train()
            epoch_loss = defaultdict(float)
            self.tradeoff = self._get_tradeoff(args.tradeoff, epoch)
            epoch_acc, epoch_loss = self._train_one_epoch(epoch_acc, epoch_loss)
            self._log_epoch_info(epoch_loss, epoch_acc, self.num_iter)

            if bool(getattr(args, 'eval_each_epoch', True)):
                new_acc = self.test()
                if bool(getattr(args, 'select_best_on_target', True)):
                    new_score = self._checkpoint_selection_score()
                    metric_name = str(getattr(args, 'best_metric', 'accuracy'))
                    if new_score >= best_score:
                        best_score = new_score
                        best_acc = new_acc
                        best_epoch = epoch
                        if getattr(args, 'save', False) and getattr(args, 'save_best', True):
                            if hasattr(self, 'save_best_model'):
                                self.save_best_model()
                            else:
                                self.save_model()
                            logging.info(
                                'Best model updated at epoch {}, target-test-acc {:.4f}, '
                                '{} score {:.4f}'.format(
                                    best_epoch, best_acc, metric_name, best_score
                                )
                            )
                    logging.info(
                        'The best model epoch {}, target-test-acc {:.4f}, '
                        '{} score {:.4f}'.format(
                            best_epoch, best_acc, metric_name, best_score
                        )
                    )
                else:
                    logging.info('Target test was reported but not used for checkpoint selection.')

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        if not bool(getattr(args, 'eval_each_epoch', True)):
            logging.info(
                'Training finished; evaluating the held-out target set once (strict mode).'
            )
            self.test()

    def test(self):
        self._set_to_eval()
        correct = defaultdict(int)
        total = defaultdict(int)
        self._eval_pred_list = []
        self._eval_label_list = []
        self.last_eval_metrics = {}

        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for _ in tqdm(range(num_iter), ascii=True):
                target_data, actual_labels = next(iters)
                target_data = target_data.to(self.device)
                correct, total = self._eval(
                    target_data, actual_labels, correct, total
                )

        for key in correct.keys():
            logging.info(
                'Target-Test-{}: {:.4f}'.format(key, correct[key] / total[key])
            )

        accuracy = correct['acc'] / max(total['acc'], 1)
        self.last_eval_metrics['accuracy'] = float(accuracy)

        if len(self._eval_pred_list) > 0 and len(self._eval_label_list) > 0:
            preds = torch.cat(self._eval_pred_list, dim=0).cpu()
            labels = torch.cat(self._eval_label_list, dim=0).cpu()
            unique_classes = sorted(
                list(set(labels.tolist()) | set(preds.tolist()))
            )

            eps = 1e-12
            f1_list = []
            recall_list = []
            support_list = []
            per_class_precision = {}
            per_class_recall = {}
            per_class_f1 = {}
            per_class_support = {}

            for cls in unique_classes:
                tp = ((preds == cls) & (labels == cls)).sum().item()
                fp = ((preds == cls) & (labels != cls)).sum().item()
                fn = ((preds != cls) & (labels == cls)).sum().item()
                support = (labels == cls).sum().item()

                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1 = 2 * precision * recall / (precision + recall + eps)

                per_class_precision[int(cls)] = float(precision)
                per_class_recall[int(cls)] = float(recall)
                per_class_f1[int(cls)] = float(f1)
                per_class_support[int(cls)] = int(support)
                f1_list.append(f1)
                recall_list.append(recall)
                support_list.append(support)

                logging.info(
                    'Target-Test-Class-{} | Precision: {:.4f} | Recall: {:.4f} '
                    '| F1: {:.4f} | Support: {}'.format(
                        cls, precision, recall, f1, support
                    )
                )

            macro_f1 = sum(f1_list) / max(len(f1_list), 1)
            total_support = sum(support_list)
            weighted_f1 = sum(
                f1 * support for f1, support in zip(f1_list, support_list)
            ) / (total_support + eps)
            macro_recall = sum(recall_list) / max(len(recall_list), 1)
            weighted_recall = sum(
                recall * support
                for recall, support in zip(recall_list, support_list)
            ) / (total_support + eps)

            self.last_eval_metrics.update({
                'macro_f1': float(macro_f1),
                'weighted_f1': float(weighted_f1),
                'macro_recall': float(macro_recall),
                'weighted_recall': float(weighted_recall),
                'per_class_precision': per_class_precision,
                'per_class_recall': per_class_recall,
                'per_class_f1': per_class_f1,
                'per_class_support': per_class_support,
            })

            logging.info('Target-Test-F1-macro: {:.4f}'.format(macro_f1))
            logging.info('Target-Test-F1-weighted: {:.4f}'.format(weighted_f1))
            logging.info('Target-Test-Recall-macro: {:.4f}'.format(macro_recall))
            logging.info(
                'Target-Test-Recall-weighted: {:.4f}'.format(weighted_recall)
            )

            if bool(getattr(self.args, 'log_confusion_matrix', True)):
                predicted_counts = {
                    int(cls): int((preds == cls).sum().item())
                    for cls in unique_classes
                }
                logging.info(
                    'Target-Test predicted class counts: {}'.format(
                        ', '.join(
                            'c{}={}'.format(cls, predicted_counts[cls])
                            for cls in unique_classes
                        )
                    )
                )
                logging.info(
                    'Target-Test confusion matrix rows=true, cols=pred; class_order={}'.format(
                        unique_classes
                    )
                )
                confusion_matrix = []
                for true_cls in unique_classes:
                    row = [
                        int(((labels == true_cls) & (preds == pred_cls)).sum().item())
                        for pred_cls in unique_classes
                    ]
                    confusion_matrix.append(row)
                    logging.info(
                        'Target-Test confusion true_c{}: {}'.format(
                            true_cls, ','.join(str(value) for value in row)
                        )
                    )
                self.last_eval_metrics['predicted_counts'] = predicted_counts
                self.last_eval_metrics['confusion_matrix'] = confusion_matrix
                self.last_eval_metrics['class_order'] = [int(x) for x in unique_classes]

        if hasattr(self, '_eval_source_weight_sum') and hasattr(
            self, '_eval_source_weight_count'
        ):
            if self._eval_source_weight_count > 0:
                w = self._eval_source_weight_sum / float(
                    self._eval_source_weight_count
                )
                logging.info(
                    'Target-Test source fusion weights: {}'.format(
                        ', '.join(
                            [
                                'src{}={:.4f}'.format(i, w[i].item())
                                for i in range(w.numel())
                            ]
                        )
                    )
                )
            del self._eval_source_weight_sum
            del self._eval_source_weight_count

        self._eval_pred_list = []
        self._eval_label_list = []
        return float(accuracy)

