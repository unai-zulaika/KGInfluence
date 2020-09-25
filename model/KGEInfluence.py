from typing import Optional, Dict, Union
import os
import importlib

import torch

import numpy as np

from hvp_operator import HVPOperator

from kge.model import KgeModel
from kge.util import KgeLoss
from kge.model.kge_model import RelationalScorer
from kge import Config, Configurable, Dataset


class KGEInfluence(KgeModel):
    r"""Extend model from KGE library"""
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        scorer: Union[RelationalScorer, type],
        initialize_embedders=True,
        configuration_key=None,
    ):
        super().__init__(config, dataset, scorer, initialize_embedders,
                         configuration_key)

        self.loss = KgeLoss.create(config)

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def get_grad_of_influence_wrt_input(self, train_set, test_set):
        """
        TODO: descrp.
        """
        # Parameters
        params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}

        trainloader = torch.utils.data.DataLoader(train_set, **params)
        testloader = torch.utils.data.DataLoader(test_set, **params)

        # TODO: STEP 1 - compute test loss  gradients wrt to model's parameters
        # z_test
        test_grad_loss = self.get_test_grad_loss_no_reg_val(testloader)

        # TODO:  STEP 2 - approximate hessian and multiply by test gradients

        # TODO:  STEP 3 - compute train loss gradients

        # TODO: STEP 4 - compute influence

        return 0

        # criterion = torch.nn.BCELoss()
        # operator = HVPOperator(self, testloader, criterion)

        # return operator.apply(y)

    def get_test_grad_loss_no_reg_val(self, test_loader):
        """
        Get gradient for test data
        """
        losses = []
        loss_value = 0
        avg_loss = 0

        batch_size = test_loader.batch_size

        # batch loader
        for test_batch, test_labels in test_loader:
            scores = self.score_sp(
                test_batch[:, 0],
                test_batch[:, 1])  # scores of all objects for (s,p,?)

            print(test_batch.shape)
            print(test_labels.shape)
            # loss_value = (self.loss(scores, test_labels) / batch_size)
            loss_value = self.loss(scores, test_labels) / batch_size
            loss_value.backward()
            losses.append(loss_value.item())

        avg_loss = torch.mean(torch.FloatTensor(losses))

        grad_test_loss = torch.autograd.grad(avg_loss, self.parameters())

        return grad_test_loss

    @staticmethod
    def create_from(
        checkpoint: Dict,
        dataset: Optional[Dataset] = None,
        use_tmp_log_folder=True,
        new_config: Config = None,
    ) -> "KgeModel":
        """Loads a model from a checkpoint file of a training job or a packaged model.
        If dataset is specified, associates this dataset with the model. Otherwise uses
        the dataset used to train the model.
        If `use_tmp_log_folder` is set, the logs and traces are written to a temporary
        file. Otherwise, the files `kge.log` and `trace.yaml` will be created (or
        appended to) in the checkpoint's folder.
        """
        config = Config.create_from(checkpoint)
        if new_config:
            config.load_config(new_config)

        if use_tmp_log_folder:
            import tempfile

            config.log_folder = tempfile.mkdtemp(prefix="kge-")
        else:
            config.log_folder = checkpoint["folder"]
            if not config.log_folder or not os.path.exists(config.log_folder):
                config.log_folder = "."
        dataset = Dataset.create_from(checkpoint,
                                      config,
                                      dataset,
                                      preload_data=False)

        model = KGEInfluence.create(config, dataset, init_for_load_only=True)
        model.load(checkpoint["model"])
        model.eval()
        return model

    @staticmethod
    def create(
        config: Config,
        dataset: Dataset,
        configuration_key: Optional[str] = None,
        init_for_load_only=False,
    ) -> "KgeModel":
        """Factory method for model creation."""

        try:
            if configuration_key is not None:
                model_name = config.get(configuration_key + ".type")
            else:
                model_name = config.get("model")
            class_name = config.get(model_name + ".class_name")
            module = importlib.import_module("model")
        except:
            raise Exception(
                "Can't find {}.type in config".format(configuration_key))

        try:
            model = getattr(module, class_name)(
                config=config,
                dataset=dataset,
                configuration_key=configuration_key,
                init_for_load_only=init_for_load_only,
            )
            model.to(config.get("job.device"))
            return model
        except ImportError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(
                "Can't find class {} in 'kge.model' for model {}".format(
                    class_name, model_name))
