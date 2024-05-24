import wandb
import math
import transformers
import torch
import bitsandbytes as bnb
from .lion import Lion

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        
    def project(self, full_rank_grad, iter, name = None, update_proj_stepsize_ratio = 1.0):
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(),full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad,self.ortho_matrix.t())
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()
        elif self.proj_type == 'random':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(torch.randn_like(full_rank_grad), self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(torch.randn_like(full_rank_grad), self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)         
        elif 'continuous' in self.proj_type:
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                    if "adafactor" in self.proj_type:
                        self.ortho_matrix_optim = transformers.optimization.Adafactor(
                                                    [self.ortho_matrix],
                                                    lr=1/self.update_proj_gap,
                                                    eps=(1e-30, 1e-3),
                                                    clip_threshold=1.0,
                                                    decay_rate=-0.8,
                                                    relative_step=False,
                                                    scale_parameter=False,
                                                    warmup_init=False,
                                                )
                    elif "lion" in self.proj_type:
                        self.ortho_matrix_optim = Lion([self.ortho_matrix], lr=1/self.update_proj_gap)
                    elif "sgd" in self.proj_type:
                        self.ortho_matrix_optim = torch.optim.SGD([self.ortho_matrix], lr = 1/self.update_proj_gap)
                    elif "adam8bit" in self.proj_type:
                        self.ortho_matrix_optim = bnb.optim.Adam8bit([self.ortho_matrix], lr=1/self.update_proj_gap)
                    else:
                        self.ortho_matrix_optim = torch.optim.AdamW([self.ortho_matrix], lr = 1/self.update_proj_gap)     
                else:
                    with torch.enable_grad():
                        self.ortho_matrix.requires_grad=True
                        self.ortho_matrix.grad = None 
                        projection = self.ortho_matrix.t() @ self.ortho_matrix
                        normalized_full_rank_grad = full_rank_grad/torch.norm(full_rank_grad)
                        loss = torch.norm(normalized_full_rank_grad @ projection - normalized_full_rank_grad) ** 2
                        loss.backward()
                        if name is not None:
                            wandb.log({name:wandb.Histogram(self.ortho_matrix.grad.cpu().float()), name+"_norm": torch.norm(self.ortho_matrix.grad).item(), name+'_proj_loss': loss.item()})
                        update_proj_stepsize = 1/self.update_proj_gap * update_proj_stepsize_ratio
                        for group in self.ortho_matrix_optim.param_groups:
                            group["lr"] = update_proj_stepsize
                        self.ortho_matrix.requires_grad=False

                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                    if "adafactor" in self.proj_type:
                        self.ortho_matrix_optim = transformers.optimization.Adafactor(
                                                    [self.ortho_matrix],
                                                    lr=1/self.update_proj_gap,
                                                    eps=(1e-30, 1e-3),
                                                    clip_threshold=1.0,
                                                    decay_rate=-0.8,
                                                    relative_step=False,
                                                    scale_parameter=False,
                                                    warmup_init=False,
                                                )
                    elif "lion" in self.proj_type:
                        self.ortho_matrix_optim = Lion([self.ortho_matrix], lr=1/self.update_proj_gap)
                    elif "sgd" in self.proj_type:
                        self.ortho_matrix_optim = torch.optim.SGD([self.ortho_matrix], lr = 1/self.update_proj_gap)
                    elif "adam8bit" in self.proj_type:
                        self.ortho_matrix_optim = bnb.optim.Adam8bit([self.ortho_matrix], lr=1/self.update_proj_gap)
                    else:
                        self.ortho_matrix_optim = torch.optim.AdamW([self.ortho_matrix], lr = 1/self.update_proj_gap)
                            
                else:
                    with torch.enable_grad():
                        self.ortho_matrix.requires_grad = True
                        self.ortho_matrix.grad = None
                        projection = self.ortho_matrix @ self.ortho_matrix.t()
                        normalized_full_rank_grad = full_rank_grad/torch.norm(full_rank_grad)
                        loss = torch.norm(projection @ normalized_full_rank_grad - normalized_full_rank_grad) ** 2
                        loss.backward()
                        if name is not None:
                            wandb.log({name:wandb.Histogram(self.ortho_matrix.grad.cpu().float()), name+"_norm": torch.norm(self.ortho_matrix.grad).item(), name+'_proj_loss': loss.item()})
                        update_proj_stepsize = 1/self.update_proj_gap * update_proj_stepsize_ratio
                        for group in self.ortho_matrix_optim.param_groups:
                            group["lr"] = update_proj_stepsize
                        self.ortho_matrix.requires_grad=False
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        return low_rank_grad

    def project_back(self, low_rank_grad):

        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
        elif self.proj_type == 'random':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif 'continuous' in self.proj_type:
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            self.ortho_matrix_optim.step()
            self.ortho_matrix.grad = None
        return full_rank_grad * self.scale
        
        
    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
            
        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
        
        #make the smaller matrix always to be orthogonal matrix
        if type=='right':
            A = U[:, :rank] @ torch.diag(s[:rank])
            B = Vh[:rank, :]
            
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            A = U[:, :rank]
            B = torch.diag(s[:rank]) @ Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')



