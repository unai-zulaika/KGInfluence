"""
Code base from https://github.com/noahgolmant/pytorch-hessian-eigenthings

This module defines a linear operator to compute the hessian-vector product
for a given pytorch model using subsampled data.
"""
import torch
from torch.autograd import Variable

from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos


class HVPOperator(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or notyo bie
    max_samples: max number of examples per batch using all GPUs.
    """
    def __init__(
        self,
        model,
        dataloader,
        criterion,
        use_gpu=True,
        full_dataset=True,
        max_samples=256,
    ):
        size = int(sum(p.numel() for p in model.parameters()))
        super(HVPOperator, self).__init__(size)
        self.grad_vec = torch.zeros(size)
        self.model = model
        if use_gpu:
            self.model = self.model.cuda()
        self.dataloader = dataloader
        # Make a copy since we will go over it a bunch
        self.dataloader_iter = iter(dataloader)
        self.criterion = criterion
        self.use_gpu = use_gpu
        self.full_dataset = full_dataset
        self.max_samples = max_samples

    def apply(self, vec):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        if self.full_dataset:
            return self._apply_full(vec)
        else:
            return self._apply_batch(vec)

    def _apply_batch(self, vec):
        # compute original gradient, tracking computation graph
        self.zero_grad()
        # first gradient vectorized
        grad_vec, preds = self.prepare_grad()
        self.zero_grad()

        vec = vec.unsqueeze(0).cuda()
        # vec = Variable(vec, requires_grad=True)

        vec_pred = torch.sigmoid(self.model.score_sp(vec[:, 0], vec[:, 1]))
        vec_pred = torch.max(vec_pred, dim=-1)  # get max score value

        print(grad_vec.shape)
        # print(sum([param.nelement() for param in self.model.parameters()]))
        # print(64 * 14541)
        # vec = torch.ones(grad_vec.shape[0]).cuda()
        # print(preds.flatten().shape)
        # exit()
        vec_grad = torch.autograd.grad(vec_pred, self.model.parameters())

        # take the second gradient
        grad_grad = torch.autograd.grad(grad_vec,
                                        self.model.parameters(),
                                        grad_outputs=vec_grad,
                                        only_inputs=True)

        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat(
            [g.contiguous().view(-1) for g in grad_grad])
        return hessian_vec_prod

    def _apply_full(self, vec):
        n = len(self.dataloader)
        hessian_vec_prod = None
        for _ in range(n):
            if hessian_vec_prod is not None:
                hessian_vec_prod += self._apply_batch(vec)
            else:
                hessian_vec_prod = self._apply_batch(vec)
        hessian_vec_prod = hessian_vec_prod / n
        return hessian_vec_prod

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_grad(self):
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        try:
            all_inputs, all_targets = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            all_inputs, all_targets = next(self.dataloader_iter)

        num_chunks = max(1, len(all_inputs) // self.max_samples)

        grad_vec = None

        input_chunks = all_inputs.chunk(num_chunks)
        target_chunks = all_targets.chunk(num_chunks)
        for input, target in zip(input_chunks, target_chunks):
            if self.use_gpu:
                input = input.cuda()
                target = target.long().cuda()

            output = torch.sigmoid(
                self.model.score_sp(input[:, 0], input[:, 1]))

            targets = torch.zeros(output.shape).cuda()
            targets[:, target] = 1.0

            loss = self.criterion(output, targets)
            grad_dict = torch.autograd.grad(loss,
                                            self.model.parameters(),
                                            create_graph=True)
            if grad_vec is not None:
                grad_vec += torch.cat(
                    [g.contiguous().view(-1) for g in grad_dict])
            else:
                grad_vec = torch.cat(
                    [g.contiguous().view(-1) for g in grad_dict])
        grad_vec /= num_chunks
        self.grad_vec = grad_vec

        return self.grad_vec, output
