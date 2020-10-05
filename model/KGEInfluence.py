from typing import Optional, Dict, Union
import os
import importlib

import torch

import numpy as np

from scipy.optimize import fmin_ncg

from hvp_operator import HVPOperator

from kge.model import KgeModel
from kge.util import KgeLoss
from kge.model.kge_model import RelationalScorer
from kge import Config, Configurable, Dataset

from tqdm import tqdm


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

    # obtain gradients in vector form
    def get_embeddings_parameters(self, test_batch):
        """
        Obtain vectorized gradients of the corresponding embeddings
        """

        s_emb = self._entity_embedder._embeddings(test_batch[0][0].long())
        r_emb = self._relation_embedder._embeddings(test_batch[0][1].long())
        o_emb = self._entity_embedder._embeddings(test_batch[0][2].long())

        return torch.cat((s_emb, r_emb, o_emb))

    # obtain gradients in vector form
    def get_embeddings_grad(self, embedding_indices, grads):
        """
        Obtain vectorized gradients of the corresponding embeddings
        """
        s_emb_grad = grads[0][embedding_indices[0][0].long()]
        r_emb_grad = grads[1][embedding_indices[0][1].long()]
        o_emb_grad = grads[0][embedding_indices[0][2].long()]

        return torch.cat((s_emb_grad, r_emb_grad, o_emb_grad))

    def get_fmin_loss_fn(self, v):
        def get_fmin_loss(x):
            return 0.5 * np.dot(self.hvp, x) - np.dot(v, x)

        return get_fmin_loss

    def get_fmin_grad_fn(self, v):
        # hessian_vector_val = self.minibatch_hessian_vector_val(
        #     self.vec_to_list(x))

        # return np.concatenate(hessian_vector_val) - np.concatenate(v)
        return (self.hvp - v)

    # def get_cg_callback(self, v, hvp, verbose):
    #     fmin_loss_fn = self.get_fmin_loss_fn(v, hvp)

    #     def fmin_loss_split(x):
    #         hessian_vector_val = self.minibatch_hessian_vector_val(
    #             self.vec_to_list(x))

    #         return 0.5 * np.dot(np.concatenate(hessian_vector_val),
    #                             x), -np.dot(np.concatenate(v), x)

    #     def cg_callback(x):
    #         # x is current params
    #         v = self.vec_to_list(x)
    #         idx_to_remove = 5

    #         single_train_feed_dict = self.fill_feed_dict_with_one_ex(
    #             self.data_sets.train, idx_to_remove)
    #         train_grad_loss_val = self.sess.run(
    #             self.grad_total_loss_op, feed_dict=single_train_feed_dict)
    #         predicted_loss_diff = np.dot(
    #             np.concatenate(v),
    #             np.concatenate(train_grad_loss_val)) / self.num_train_examples

    #         if verbose:
    #             print('Function value: %s' % fmin_loss_fn(x, hvp))
    #             quad, lin = fmin_loss_split(x)
    #             print('Split function value: %s, %s' % (quad, lin))
    #             print('Predicted loss diff on train_idx %s: %s' %
    #                   (idx_to_remove, predicted_loss_diff))

    #     return cg_callback

    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(
            self.vec_to_list(p))

        return np.concatenate(hessian_vector_val)

    def get_inverse_hvp_cg(self, v, verbose=True):
        """
        TODO: descrp.
        """
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        fmin_grad_fn = self.get_fmin_grad_fn(v)
        # cg_callback = self.get_cg_callback(v, hvp, verbose)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            # x0=np.concatenate(v),
            x0=v,
            fprime=fmin_grad_fn,
            fhess_p=self.hess_p,
            # callback=cg_callback,
            avextol=1e-8,
            maxiter=100)

        return self.vec_to_list(fmin_results)

    def compute_hessian(self, train_loader, test_loader, test_grad):
        """
        TODO: descrp.
        """
        hvps = []

        for i in tqdm(range(test_grad.shape[0])):
            self.zero_grad()
            # batch loader
            for train_batch in tqdm(train_loader):
                batch = train_batch.to(self.device)
                labels = (train_batch[:, 2]).long().to(self.device)

                scores = self.score_sp(
                    batch[:, 0], batch[:,
                                       1])  # scores of all objects for (s,p,?)
                loss_value = self.loss(scores, labels)
                loss_value.backward(retain_graph=True)

            grads = torch.autograd.grad(loss_value,
                                        self.parameters(),
                                        create_graph=True)

            grads = self.get_embeddings_grad(test_loader, grads)

            flatten_grads = torch.cat(
                [g.reshape(-1) for g in grads if g is not None])

            # for g in grads:
            #     print(g.shape)
            # # print(test_grad[0].shape)
            # # print(flatten_grads.shape)
            # for p in self.parameters():
            #     print(p.shape)
            # flatten_embeddings_grad =
            # print(self)
            # test_grad = torch.Tensor(test_grad)
            # print(flatten_grads.shape)
            # print(test_grad.shape)
            hvps.append(
                torch.autograd.grad([flatten_grads @ test_grad[i]],
                                    self.parameters(),
                                    allow_unused=True))
        print(len(hvps))
        exit()
        return torch.stack(hvps)

    def hess_p(self, train_loader, p):
        """
        TODO: descrp.
        """
        operator = HVPOperator(self, train_loader, self.loss)
        self.hvp = operator.apply(p).cpu()

        return self.hvp

    def get_influence(self, train_loader, test_loader):
        """
        TODO: descrp.
        """

        # TODO: STEP 1 - compute s_test = inv_hessian * gradient of the loss given test_sample
        # compute gradient of the loss given test_sample wrt to trained model's parameters
        print('\n Computing test examples gradients: \n')
        test_grad_loss = self.test_gradients(test_loader)

        # TODO:  STEP 2 - approximate hessian and multiply by test gradients
        # perform hessian vector product
        print('\n Computing hessian vector products: \n')
        self.compute_hessian(train_loader, test_loader, test_grad_loss)
        test_grad_loss = test_grad_loss.cpu()

        # now approximate inverse from hvp
        print('\n Computing inverse: \n')
        inverse_hvp = self.get_inverse_hvp_cg(test_grad_loss)

        # TODO:  STEP 3 - compute train loss gradients
        print('\n Computing train loss: \n')

        # TODO: STEP 4 - compute influence
        influences = 0

        return influences

    def test_gradients(self, test_loader):
        """
        Get gradients for test data
        """

        gradients = []
        loss_value = 0

        # batch loader
        for test_batch in tqdm(test_loader):
            self.zero_grad()
            test_batch = test_batch.to(self.device)
            test_labels = (test_batch[:, 2]).long().to(self.device)

            scores = self.score_sp(
                test_batch[:, 0],
                test_batch[:, 1])  # scores of all objects for (s,p,?)

            # loss_value = (self.loss(scores, test_labels) / batch_size)
            loss_value = self.loss(scores, test_labels)
            # loss_value.backward(retain_graph=True)
            grad_dict = torch.autograd.grad(loss_value, self.parameters())
            gradients.append(self.get_embeddings_grad(test_batch, grad_dict))

        # avg_loss = torch.FloatTensor(losses)
        # avg_loss.requires_grad = True
        # avg_loss = torch.mean(avg_loss)

        # now get gradients
        # grad_test_loss = self.get_embeddings_grad()
        # print(grad_test_loss)

        return torch.stack(gradients)

    @staticmethod
    def create_from(
        checkpoint: Dict,
        device,
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
        model.device = device
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
