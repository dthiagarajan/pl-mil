import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional.classification import accuracy, auroc, precision_recall
import torch
import torch.nn as nn

from core.data.samplers import TopKSampler
from core.distributed.ops import all_gather_op


class MILModel(LightningModule):
    def __init__(self, model, topk=2, aggregation='max'):
        super(MILModel, self).__init__()
        self.model = model
        self.loss = nn.BCELoss()
        # Below parameters and associated functionality can be moved out into a top-k module
        self.topk = topk
        self.aggregation = np.amax if aggregation == 'max' else np.mean
        self.training_log_step, self.validation_log_step, self.testing_log_step = 0, 0, 0
        self.training_log_epoch, self.validation_log_epoch, self.testing_log_epoch = 0, 0, 0

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        index, image, label = batch
        # Log images here
        output = torch.sigmoid(self(image))
        loss = self.loss(output, label.float())
        self.logger.log_metrics({'Training/Step Loss': loss}, step=self.training_log_step)
        self.training_log_step += 1
        return {'index': index, 'prob': output.detach(), 'label': label, 'loss': loss}

    def training_epoch_end(self, outputs):
        outputs = self.all_gather_outputs(outputs)
        loss, indices, probs, labels = outputs
        self.trainer.datamodule.train_dataset_reference.dataset.loc[indices, 'trained_prob'] = probs
        acc, auc, precision, recall = self.get_metrics(
            self.trainer.datamodule.train_dataset_reference.dataset,
            indices,
            prob_col_name='trained_prob',
            group='id'
        )
        self.logger.log_metrics(
            {
                f'Training/Epoch {k}': v for k, v in
                {'acc': acc, 'auc': auc, 'precision': precision, 'recall': recall}.items()
            },
            self.training_log_epoch
        )
        self.training_log_epoch += 1
        self.training_metrics = {'acc': acc, 'auc': auc, 'precision': precision, 'recall': recall}

    def test_step(self, batch, batch_idx):
        index, image, label = batch
        output = torch.sigmoid(self(image))
        loss = self.loss(output, label.float())
        self.logger.log_metrics({'Testing/Step Loss': loss}, step=self.testing_log_step)
        self.testing_log_step += 1
        return {'index': index, 'prob': output, 'label': label, 'loss': loss}

    def test_epoch_end(self, outputs):
        outputs = self.all_gather_outputs(outputs)
        loss, indices, probs, labels = outputs
        self.trainer.datamodule.inference_dataset_reference.dataset.loc[indices, 'prob'] = probs
        self.topk_indices = self.get_topk(
            self.trainer.datamodule.inference_dataset_reference.dataset,
            prob_col_name='prob',
            group='id'
        )
        acc, auc, precision, recall = self.get_metrics(
            self.trainer.datamodule.inference_dataset_reference.dataset,
            self.topk_indices,
            prob_col_name='prob',
            group='id'
        )
        self.logger.log_metrics(
            {
                f'Testing/Epoch {k}': v for k, v in
                {'acc': acc, 'auc': auc, 'precision': precision, 'recall': recall}.items()
            },
            self.testing_log_epoch
        )
        self.testing_log_epoch += 1
        self.trainer.datamodule.train_sampler = TopKSampler(self.topk_indices)
        return {'acc': acc, 'auc': auc, 'precision': precision, 'recall': recall}

    def configure_optimizers(self):
        return torch.optim.AdamW([{'params': self.model.parameters(), 'lr': 1e-3}])

    def all_gather_outputs(self, outputs):
        losses = torch.stack([x['loss'] for x in outputs])
        probs = torch.cat([x['prob'] for x in outputs])
        indices = torch.cat([x['index'] for x in outputs])
        labels = torch.cat([x['label'] for x in outputs])

        if 'CPU' in self.trainer.accelerator_backend.__class__.__name__:
            return (
                losses.mean(),
                indices,
                probs,
                labels
            )

        return (
            all_gather_op(losses).mean(),
            all_gather_op(indices),
            all_gather_op(probs),
            all_gather_op(labels)
        )

    def get_topk(self, df, prob_col_name='prob', group='id'):
        return np.hstack(df.groupby(group).apply(
            lambda gdf: gdf.sort_values(prob_col_name, ascending=False).index.values[:self.topk]
        ))

    def get_metrics(self, df, topk_indices, prob_col_name='prob', group='id'):
        sub_df = df.loc[np.hstack(topk_indices)]
        grouped_sub_df = sub_df.groupby('id')
        probs = torch.from_numpy(grouped_sub_df.apply(
            lambda gdf: self.aggregation(gdf[prob_col_name])).values
        )
        preds = torch.from_numpy(grouped_sub_df.apply(
            lambda gdf: self.aggregation(gdf[prob_col_name]) > 0.5).values.astype('int')
        )
        labels = torch.from_numpy(grouped_sub_df.apply(
            lambda gdf: self.aggregation(gdf.label)).values
        )
        acc = accuracy(preds, labels)
        auc = auroc(probs, labels)
        precision, recall = precision_recall(preds, labels)
        return acc.item(), auc.item(), precision.item(), recall.item()
