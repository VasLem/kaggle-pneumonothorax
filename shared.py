"""Shared methods and classes between notebooks
"""
import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset
from torchnet.meter import AverageValueMeter
from tqdm import tqdm_notebook as tqdm

from config import CONFIG
from custom_pytorch.custom_logs import Logger
from custom_pytorch.custom_utils import check_stage
from custom_pytorch.custom_utils.train import Trainer as _Trainer
from custom_pytorch.custom_utils.test import Tester as _Tester
from transformations import handle_transformations

CURR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DIR_ABOVE = os.sep.join(CURR_FILE_PATH.split(os.sep)[:-1])
sys.path.insert(0, os.path.join(
    DIR_ABOVE, 'input/siim-acr-pneumothorax-segmentation'))

DATA_PATH = os.path.join(DIR_ABOVE, "input/pneumonothorax-data/")
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')
MODEL_SAVE_DIR = os.path.join(DIR_ABOVE, "input/models/pneumonothorax")
LOGS_SAVE_DIR = os.path.join(DIR_ABOVE, 'input/logs/pneumonothorax')
try:
    os.makedirs(MODEL_SAVE_DIR)
except OSError:
    pass
try:
    os.makedirs(LOGS_SAVE_DIR)
except OSError:
    pass
train_folder_size = len(os.listdir(os.path.join(TRAIN_PATH, 'images')))
test_folder_size = len(os.listdir(os.path.join(TEST_PATH, 'images')))
print('Training folder data size:', train_folder_size)
print('Testing folder data size:', test_folder_size)


_Epoch = smp.utils.train.Epoch


def epoch_run_override(self, dataloader, _logs=None):

    self.on_epoch_start()

    logs = {}
    loss_meter = AverageValueMeter()
    metrics_meters = {metric.__name__: AverageValueMeter()
                      for metric in self.metrics}
    if _logs is not None:
        _logs.clear()
    with tqdm(dataloader, desc=self.stage_name,
              file=sys.stdout, disable=not (self.verbose)) as iterator:
        for item in iterator:
            x = item['images']
            y = item['masks']
            x, y = x.to(self.device), y.to(self.device)
            loss, y_pred = self.batch_update(x, y, logs=_logs)

            # update loss logs
            loss_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_value)
            loss_logs = {self.loss.__name__: loss_meter.mean}
            logs.update(loss_logs)

            # update metrics logs
            for metric_fn in self.metrics:
                metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                metrics_meters[metric_fn.__name__].add(metric_value)
            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
            logs.update(metrics_logs)
            try:
                logs.update({'LR': self.optimizer.param_groups[0]['lr']})
            except AttributeError:
                pass
            if self.verbose:
                s = self._format_logs(logs)
                iterator.set_postfix_str(s)

    return logs


def train_batch_update_with_logs(self, x, y, logs=None):
    self.optimizer.zero_grad()
    prediction = self.model.forward(x)

    loss = self.loss(prediction, y, logs=logs)
    loss.backward()
    self.optimizer.step()
    return loss, prediction


def valid_batch_update_with_logs(self, x, y, logs=None):
    with torch.no_grad():
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y, logs=logs)
    return loss, prediction


smp.utils.train.TrainEpoch.run = epoch_run_override
smp.utils.train.ValidEpoch.run = epoch_run_override
smp.utils.train.TrainEpoch.batch_update = train_batch_update_with_logs
smp.utils.train.ValidEpoch.batch_update = valid_batch_update_with_logs


class PneumothoraxDataset(Dataset):
    def __init__(self, path, stage, apply_augmentation=True, metadata_encoder=None):
        super().__init__()
        self.test = False
        self.train = False
        self.valid = False
        if stage == 'train':
            self.train = True
        elif stage == 'valid':
            self.valid = True
        elif stage == 'test':
            self.test = True
        else:
            raise ValueError(f"Provided stage ({stage}) is not"
                             " any of the accepted values 'train', 'test' or 'valid'")
        self.path = path
        self._images_files = None
        self.apply_augmentation = apply_augmentation
        self.metadata = pd.read_csv(os.path.join(
            self.path, 'metadata.csv'), index_col='ImageId')
        from sklearn.preprocessing import OneHotEncoder
        metadata_subset = self.metadata[['PatientSex', 'ViewPosition']]
        if metadata_encoder is None:
            self.encoder = OneHotEncoder(sparse=False)
            self.encoder.fit(metadata_subset.to_numpy().tolist())

        encoded = self.encoder.transform(metadata_subset.to_numpy().tolist())
        self.metadata = pd.DataFrame(
            encoded, index=list(metadata_subset.index.values))

    def __len__(self):
        return len(self.images_files)

    def _handle_single(self, name):
        image = cv2.imread(os.path.join(self.path, 'images', name), 0)
        metadata = self.metadata.loc[os.path.splitext(
            name)[0]].values.astype(int)
        if self.train or self.valid:
            try:
                mask = cv2.imread(os.path.join(self.path, 'masks', name), 0)
                if mask is None:
                    raise OSError
            except OSError:
                mask = np.zeros(image.shape, np.uint8)
        else:
            mask = None
        ret = handle_transformations(
            image, mask, augment=self.apply_augmentation)
        ret['name'] = name
        ret['metadata'] = metadata
        try:
            ret['mask'] = ret['mask'][:, :, :, 0]
        except (KeyError, IndexError):
            pass
        return ret

    def __iter__(self):
        for name in self.images_files:
            yield self._handle_single(name)

    def tta(self, index, tta_size):
        for _ in range(tta_size):
            yield self[index]

    def compute_pixels_weights(self, index):
        assert self.train, "This cannot be called unless in training"
        try:
            with open('pixels-weights.pkl', 'rb') as inp:
                dic = pickle.load(inp)
                if dic['size'] == CONFIG.im_size and np.all(
                        np.array(dic['index']) == np.array(index)):
                    return dic['weights']
                raise IOError
        except IOError:

            summated_mask = np.zeros((CONFIG.im_size, CONFIG.im_size), int)
            for ind in tqdm(index):
                mask = self[ind][-1].cpu().data.numpy().astype(int).squeeze()
                assert np.all((mask == 0) | (mask == 1))
                summated_mask += mask
            tol = 1.0
            summated_mask = (summated_mask + tol) / (float(len(index)) + tol)
            weights = 1 / summated_mask
            weights = weights / np.max(weights)
            dic = {'index': np.array(
                index), 'weights': weights, 'size': CONFIG.im_size}
            with open('pixels-weights.pkl', 'wb') as out:
                pickle.dump(dic, out)
            return weights

    @property
    def images_files(self):
        if self._images_files is None:
            self._images_files = os.listdir(os.path.join(self.path, 'images'))
        return self._images_files

    def __getitem__(self, index):
        if not isinstance(index, list):
            index = [index]
        self.imgs_size = []
        rets = []
        for fil_index in index:
            if isinstance(fil_index, str):
                for cnt, name in enumerate(self.images_files):
                    if name.startswith(fil_index):
                        fil_index = cnt
                        break
                else:
                    raise BaseException(
                        f'Provided Image Id {fil_index} was not found')
            else:
                name = self.images_files[fil_index]
            rets.append(self._handle_single(name))

        if len(index) == 1:
            return rets[0]
        return rets


def collate_fn(batch):
    ret = dict(
        files=[item['name'] for item in batch],
        images=torch.stack([item['image'] for item in batch]),
        metadata=[item['metadata'] for item in batch])
    try:
        ret['masks'] = torch.stack([item['mask'] for item in batch])

    except KeyError:
        pass
    return ret


def create_weights(dataset):
    exists = []
    print("Creating weights")
    for batch in tqdm(dataset):
        exists.append(np.any(batch['mask'].cpu().data.numpy()))
    exists = np.array(exists)
    freq = np.sum(exists)
    weights = np.array(exists).astype(float)
    weights[exists] = 1 / freq
    weights[~exists] = 1 / (exists.size - freq)
    return weights


def get_dataset(stage):
    return check_stage(
        stage,
        train=PneumothoraxDataset(TRAIN_PATH, 'train'),
        valid=PneumothoraxDataset(
            TRAIN_PATH, 'valid'),
        test=PneumothoraxDataset(TEST_PATH, 'test', apply_augmentation=False))


def get_weights(train_dataset):
    try:
        with open('weights.pkl', 'rb') as inp:
            weights = pickle.load(inp)
    except BaseException:
        weights = create_weights(train_dataset)
        with open('weights.pkl', 'wb') as out:
            pickle.dump(weights, out)
    return weights


def compare_with_low_thres(inp, thres):
    return inp <= CONFIG.im_size ** 2 * thres


class Trainer(_Trainer):

    def __init__(self, model, optimizer, loss_function, metric_functions,
                 device='cuda', verbose=True):

        train_dataset = get_dataset('train')
        valid_dataset = get_dataset('valid')
        weights = get_weights(train_dataset)
        super().__init__(config=CONFIG, train_dataset=train_dataset, valid_dataset=valid_dataset,
                         inp_index='images', gt_index='masks',
                         collate_fn=collate_fn, model=model, optimizer=optimizer,
                         loss_function=loss_function, metric_functions=metric_functions,
                         samples_weights=weights)
        self.partial_losses_logger = Logger(
            CONFIG, 'partial_losses_logs', create_dir=True)
        self.train_loss_logs = {}
        self.valid_loss_logs = {}

    def write_logs(self, step_logs, valid):
        super().write_logs(step_logs, valid)
        step = self.epoch
        partial_logs = self.train_loss_logs
        logs = self.train_logs
        if valid:
            logs = self.valid_logs
            partial_logs = self.valid_loss_logs
            self.valid_loss_logs = {}
        else:
            self.train_loss_logs = {}
        partial_logs = {key: partial_logs[key].mean for key in partial_logs}
        logs[step]['partial losses'] = partial_logs
        self.partial_losses_logger.update(
            step, logs[step]['partial losses'], valid=valid)

    def find_best_binary_thresholds(self):
        masks = []
        outs = []
        for batch in tqdm(self.valid_loader):
            ret = self.valid_step(batch)[0]
            masks.extend([r for r in ret[1].cpu().data.numpy()])
            outs.extend([r for r in ret[2].cpu().data.numpy()])

        low_thres_mes = np.linspace(0, 0.7, 100)
        hi_thres_mes = np.linspace(0.1, 1, 100)
        metric_mat = np.zeros((100, 100))
        preds = np.array(outs)
        masks = np.array(masks)
        for low_cnt, low_thres in enumerate(low_thres_mes):
            for hi_cnt, hi_thres in enumerate(low_thres_mes):
                c_preds = preds.copy()
                c_preds[np.sum(preds.reshape(
                    (preds.shape[0], -1)), axis=1) < CONFIG.im_size ** 2 * low_thres, :, :] = 0
                c_preds = c_preds > hi_thres
                metric_mat[low_cnt, hi_cnt] = self.metric_function(
                    torch.from_numpy(c_preds), torch.from_numpy(masks))
        inds = np.unravel_index(np.argmax(metric_mat), metric_mat.shape)
        noise_th = low_thres_mes[inds[0]]
        best_thr = hi_thres_mes[inds[1]]
        return {'noise_th': noise_th, 'comp_th': best_thr}


class Tester(_Tester):
    def __init__(self, model, device='cuda'):
        dataset = get_dataset('test')
        super().__init__(dataset=dataset, inp_index='images', model=model, device=device)

    def compute_rles(self, out, noise_th, best_th, keep_largest, h=1024, w=1024):
        from mask_functions import mask2rle
        out = out.cpu().data.numpy()
        out_masks = []
        rles = []
        zcnt = 0
        for mask in out:
            mask = mask.squeeze()
            if compare_with_low_thres(np.sum(mask), noise_th):
                mask = np.zeros_like(mask)
            else:
                mask = 255 * (mask > best_th).astype(np.uint8)
            if not np.any(mask):
                rles.append('')
                zcnt += 1
                out_masks.append(np.zeros((h, w), np.uint8))
                continue
            if keep_largest:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                areas = [cv2.contourArea(c) for c in contours]
                mask[:] = 0
                cv2.drawContours(
                    mask, [contours[np.argmax(areas)]], 0, 255, -1)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            out_masks.append(mask)
            rles.append(mask2rle(mask, w, h))
        return rles, zcnt
