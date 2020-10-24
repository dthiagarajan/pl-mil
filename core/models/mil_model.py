from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional.classification import accuracy, auroc, precision_recall
import torch
import torch.nn as nn

from core.data.samplers import TopKSampler
from core.distributed.ops import all_gather_op
from core.processing import TopKProcessor


class MILModel(LightningModule):
    def __init__(self, model, topk=2, aggregation='max'):
        super(MILModel, self).__init__()
        self.model = model
        self.loss = nn.BCELoss()
        self.topk_processor = TopKProcessor(topk=topk, aggregation=aggregation)
        self.training_log_step, self.validation_log_step, self.testing_log_step = 0, 0, 0
        self.training_log_epoch, self.validation_log_epoch, self.testing_log_epoch = 0, 0, 0

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        index, image, label = batch
        # Log images here
        if batch_idx == 0:
            self.logger.experiment.add_images(
                'Top-K Images', image, global_step=self.training_log_step
            )
        output = torch.sigmoid(self(image))
        loss = self.loss(output, label.float())
        self.logger.log_metrics({'Training/Step Loss': loss}, step=self.training_log_step)
        self.training_log_step += 1
        return {'index': index, 'prob': output.detach(), 'label': label, 'loss': loss}

    def training_epoch_end(self, outputs):
        outputs = self.all_gather_outputs(outputs)
        loss, indices, probs, labels = outputs
        self.trainer.datamodule.train_dataset_reference.dataset.loc[indices, 'trained_prob'] = probs
        probs, preds, labels = self.topk_processor.aggregate(
            self.trainer.datamodule.train_dataset_reference.dataset,
            indices,
            prob_col_name='prob',
            group='id'
        )
        acc, auc, precision, recall = self.get_metrics(probs, preds, labels)
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
        self.topk_indices = self.topk_processor(
            self.trainer.datamodule.inference_dataset_reference.dataset,
            prob_col_name='prob',
            group='id'
        )
        probs, preds, labels = self.topk_processor.aggregate(
            self.trainer.datamodule.inference_dataset_reference.dataset,
            self.topk_indices,
            prob_col_name='prob',
            group='id'
        )
        acc, auc, precision, recall = self.get_metrics(probs, preds, labels)
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

    def get_metrics(self, probs, preds, labels):
        acc = accuracy(preds, labels)
        auc = auroc(probs, labels)
        precision, recall = precision_recall(preds, labels)
        return acc.item(), auc.item(), precision.item(), recall.item()
