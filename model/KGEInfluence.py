from typing import Optional, Dict, Union
import os
import importlib
import time

import torch

import numpy as np

from scipy.optimize import fmin_ncg, minimize

from kge.model.distmult import DistMult
from kge.model import KgeModel
from kge.util import KgeLoss
from kge.model.kge_model import RelationalScorer
from kge import Config, Configurable, Dataset

from tqdm import tqdm
"""
https://www.cs.princeton.edu/courses/archive/fall18/cos597G/pytorch.pdf
"""


class KGEInfluence(KgeModel):
    r"""Extend model from KGE library"""
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        scorer: Union[RelationalScorer, type],
        # damping=1e-02,
        damping=1,
        avextol=1e-03,
        initialize_embedders=True,
        configuration_key=None,
        hvps_save_dir='hvps',
        verbose=True,
    ):
        super().__init__(config, dataset, scorer, initialize_embedders,
                         configuration_key)
        print('CREATED KGEINFLUENCE MODEL')
        self.loss = KgeLoss.create(config)
        self.verbose = verbose
        self.hvps_save_dir = hvps_save_dir
        self.damping = damping
        self.avextol = avextol
        self.compute = False

    # obtain gradients in vector form
    def get_embeddings_parameters(self, triple):
        """
        Obtain vectorized gradients of the corresponding embeddings
        """
        for i, parameter in enumerate(self.parameters()):
            print(parameter)
            if i == 0:
                s_emb = parameter[triple[0]]
                o_emb = parameter[triple[2]]
            if i == 1:
                r_emb = parameter[triple[1]]
        # s_emb = self._entity_embedder._embeddings(test_batch[0][0].long())
        # r_emb = self._relation_embedder._embeddings(test_batch[0][1].long())
        # o_emb = self._entity_embedder._embeddings(test_batch[0][2].long())

        return (torch.stack([s_emb, o_emb]), torch.stack([r_emb]))

    # obtain gradients in vector form
    def get_embeddings_grad(self, embedding_indices, grads):
        """
        Obtain vectorized gradients of the corresponding embeddings
        """
        embedding_indices = embedding_indices.squeeze()

        s_emb_grad = grads[0][embedding_indices[0].long()]
        r_emb_grad = grads[1][embedding_indices[1].long()]
        o_emb_grad = grads[0][embedding_indices[2].long()]

        return torch.cat((s_emb_grad, r_emb_grad, o_emb_grad))

    def get_fmin_loss_fn(self, train_loader, test_example_gradient):
        def get_fmin_loss(x):
            if self.verbose: print('FMIN STANDARD')
            hessian = self.compute_hessian(train_loader, x)
            vt = test_example_gradient.squeeze()

            return (0.5 * np.dot(hessian, x) - np.dot(vt, x))

        return get_fmin_loss

    def get_fmin_grad_fn(self, train_loader, test_example_gradient):
        if self.verbose: print('FMIN GRAD')

        def get_fmin_grad(x):
            hessian = self.compute_hessian(train_loader, x).cpu()
            return (hessian - test_example_gradient)

        return get_fmin_grad

    def get_fhess(self, train_loader, test_example_gradient):
        if self.verbose: print('F HESSIAN')

        def fhess(x, p):
            hessian = self.compute_hessian(train_loader, p)
            return hessian.cpu().numpy()

        return fhess

    def callback(self, train_loader, v):
        def callbackF(x):
            loss_value = self.get_fmin_loss_fn(train_loader, v)
            if self.verbose: print('Function value: {0}'.format(loss_value(x)))

        return callbackF

    def get_inverse_hvp_cg(self, train_loader, test_example_gradient):
        """
        TODO: descrp.
        """
        test_example_gradient = test_example_gradient.cpu()

        fmin_loss_fn = self.get_fmin_loss_fn(train_loader,
                                             test_example_gradient)
        fmin_grad_fn = self.get_fmin_grad_fn(train_loader,
                                             test_example_gradient)
        hess_p = self.get_fhess(train_loader, test_example_gradient)

        callback = self.callback(train_loader, test_example_gradient)

        fmin_results = minimize(
            fmin_loss_fn,
            test_example_gradient,
            method='Newton-CG',
            jac=fmin_grad_fn,
            hessp=hess_p,
            callback=callback,
            options={
                'xtol': self.avextol,
                'disp': True,
                # 'eps': 1.49e-11,
                'maxiter': 100
            })

        return torch.FloatTensor(fmin_results.x).to(self.device)

    def compute_hessian(self, train_loader, v):
        """
        TODO: descrp.
        """
        if not torch.is_tensor(v):
            v = torch.from_numpy(v).float().unsqueeze(0)
        v = v.to(self.device)
        hvp = 0
        # batch loader
        for train_batch in tqdm(train_loader):
            self.zero_grad()
            batch = train_batch.to(self.device)
            labels = (train_batch[:, 2]).long().to(self.device)

            scores = self.score_sp(
                batch[:, 0], batch[:, 1])  # scores of all objects for (s,p,?)
            loss_value = self.loss(scores, labels)
            loss_value.backward(retain_graph=True)

            # print(embeddings_parameters.shape)
            # First gradient
            grads = torch.autograd.grad(loss_value,
                                        self.parameters(),
                                        create_graph=True,
                                        only_inputs=True)
            
            grads = self.get_embeddings_grad(train_batch, grads)
            flatten_grads = torch.cat(
                [g.reshape(-1) for g in grads if g is not None]).flatten()
            if len(v) == 1:
                v = v.squeeze()
            # Second gradient
            grad_grad = torch.autograd.grad([flatten_grads @ v],
                                            self.parameters(),
                                            allow_unused=False,
                                            only_inputs=True)
            
            grad_grad = self.get_embeddings_grad(
                self.test_indices[self.current_test_index], grad_grad)
            
            flatten_grad_grad = torch.cat([
                g.reshape(-1) for g in grad_grad if g is not None
            ]) / train_loader.dataset.shape[0]

            # add damping at least one time
            # TODO: damping hyperpameter
            flatten_grad_grad = torch.FloatTensor([
                torch.add(a, torch.mul(self.damping, b))
                for (a, b) in zip(flatten_grad_grad, v)
            ])
            hvp += flatten_grad_grad

        # i = 0
        # while True:
        # if still not pd keep damping
        # try:
        #     print(flatten_grad_grad.shape)
        #     print(
        #         np.all(
        #             np.linalg.eigvals(flatten_grad_grad.unsqueeze(0)) > 0))
        #     exit()
        #     decomp = np.linalg.cholesky(flatten_grad_grad)
        #     break
        #     # print("hessian is positive definite now")
        # except Exception as e:
        #     print(e)
        #     # print("hessian is not positive definite")
        #     # print("Adding damping term...")
        #     i += 1
        #     flatten_grad_grad = torch.FloatTensor([
        #         torch.add(a, torch.mul(self.damping, b))
        #         for (a, b) in zip(flatten_grad_grad, v)
        #     ])
        # try:
        #     decomp = np.linalg.cholesky(flatten_grad_grad)
        #     print("hessian is positive definite now")
        # except:
        #     print("hessian is still not positive definite now")

        # print(f'Damped total of {i} times')
        return hvp.cpu()

    def compute_hvps(self, train_loader, test_gradients):
        # compute inverse hessian vector product
        return self.get_inverse_hvp_cg(train_loader, test_gradients)


    def get_influence(self, train_loader, single_train_loader, test_loader, verbose=False):
        """
        TODO: descrp.
        """
        self.current_test_index = torch.where((test_loader == self.test_indices).all(dim=1))[0]
        
        if verbose:
            print(f'Memory pre test examples gradients')
            print(torch.cuda.memory_summary(device=None, abbreviated=True))
            print('###' * 30)
        # STEP 1 - compute s_test = inv_hessian * gradient of the loss given test_sample
        # compute gradient of the loss given test_sample wrt to trained model's parameters
        print('\n Computing test examples gradients: \n')
        test_gradients = self.get_test_gradients(test_loader)
        
        if verbose:
            print(f'Memory post test examples gradients')
            print(torch.cuda.memory_summary(device=None, abbreviated=True))
            print('###' * 30)

        approx_filename = os.path.join(
            self.hvps_save_dir, '%s-%s.pt' %
            (self.model, self.dataset.config.options['dataset']['name']))
        if os.path.exists(approx_filename) and not self.compute:
        # if 2 + 2 == 5:
            hvps = torch.load(approx_filename)
            print('Loaded inverse HVP from %s' % approx_filename)
        else:
            # perform hessian vector product
            print('\n Computing hessian vector products via NCG: \n')
            hvps = self.compute_hvps(single_train_loader, test_gradients)

            torch.save(hvps, approx_filename)
            print('Saved inverse HVP to %s' % approx_filename)

        if verbose:
            print(f'Memory post HVPs')
            print(torch.cuda.memory_summary(device=None, abbreviated=True))
            print('###' * 30)
        

        # TODO:  STEP 3 - compute train loss gradients
        print('\n Computing train loss: \n')
        train_grad_loss = self.get_train_gradients(single_train_loader,
                                                   test_loader)
    
        if verbose:
            print(f'Memory pre train examples gradients')
            print(torch.cuda.memory_summary(device=None, abbreviated=True))
            print('###' * 30)

        # TODO: STEP 4 - compute influence
        influences = self.compute_influences(hvps, train_grad_loss)

        return influences

    def get_test_gradients(self, test_example):
        """
        Get gradients for test data
        """
        # we need to unsqueeze to match scoring function :(
        test_example = test_example.unsqueeze(dim=0)
        loss_value = 0
        
        self.zero_grad()
        test_example = test_example.to(self.device)
        test_label = (test_example[:, 2]).long().to(self.device)
        
        score = self.score_sp(
            test_example[:, 0],
            test_example[:, 1])  # scores of all objects for (s,p,?)

        loss_value = self.loss(score, test_label)

        grad_test_example = torch.autograd.grad(loss_value,
                                                self.parameters())
        
        return self.get_embeddings_grad(test_example,
                                                 grad_test_example)

    def get_train_gradients(self, single_train_loader, test_loader):
        """
        Get gradients for test data
        """
        approx_filename = os.path.join(
            'gradients/train/', '%s-%s.pt' %
            (self.model, self.dataset.config.options['dataset']['name']))
        if os.path.exists(approx_filename) and not self.compute:
        # if 2 + 2 == 5:
            all_gradients = torch.load(approx_filename)
            print('Loaded train gradients from %s' % approx_filename)
        else:
            loss_value = 0

            gradients = []
            # batch loader
            for train_batch in single_train_loader:
                self.zero_grad()
                train_batch = train_batch.to(self.device)
                train_labels = (train_batch[:, 2]).long().to(self.device)
                scores = self.score_sp(
                    train_batch[:, 0],
                    train_batch[:, 1])  # scores of all objects for (s,p,?)

                loss_value = self.loss(scores, train_labels)

                grad_dict = torch.autograd.grad(loss_value,
                                                self.parameters())
                gradients.append(
                    self.get_embeddings_grad(train_batch, grad_dict))

            all_gradients = torch.stack(gradients)

            torch.save(all_gradients, approx_filename)
            print('Saved train gradients to %s' % approx_filename)

        return all_gradients

    def compute_influences(self, hvp, train_grad_loss):
        # compute every influence for each test example
        influences = []
        train_grad_loss = train_grad_loss.squeeze()
        for train_gradient in train_grad_loss:
            # influence = torch.matmul(hvp, train_gradient)
            influence = torch.matmul(
                hvp, train_gradient) / self.dataset.config.options[
                    'dataset']['files']['train']['size']
            influences.append(influence.item())

        return torch.FloatTensor(influences)



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
                print('Loaded configuration key')
            else:
                model_name = config.get("model")
            class_name = config.get(model_name + ".class_name")
            module = importlib.import_module("model")

        except:
            raise Exception(
                "Can't find {}.type in config".format(configuration_key))

        try:
            print(module)
            print(class_name)
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
