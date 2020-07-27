from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import torch.nn.init as initilize
import pickle
from numpy.random import RandomState
import numpy as np
from lib.utils import trace

class PMF(nn.Module):
	def __init__(self, args, is_sparse=False):
		super(PMF, self).__init__()
		self.n_users = args.m
		self.n_items = args.n
		self.n_factors = args.k
		self.random_state = RandomState(args.seed)
		self.alpha=args.alpha
		self.lam_u = args.lamb
		self.lam_v = args.lamb
		self.lam_p = args.lamb
		self.lam_q = args.lamb

		self.I = torch.ones(args.k).to(args.device)

		self.user_embeddings = nn.Embedding(self.n_users, self.n_factors, sparse=is_sparse)
		self.user_embeddings.weight.data = torch.from_numpy(0.001 * self.random_state.rand(self.n_users, self.n_factors)).float()

		self.item_embeddings = nn.Embedding(self.n_items, self.n_factors, sparse=is_sparse)
		self.item_embeddings.weight.data = torch.from_numpy(0.001 * self.random_state.rand(self.n_items, self.n_factors)).float()

		self.P_para=nn.Embedding(self.n_factors, 1, sparse=is_sparse)
		self.P_para.weight.data = torch.from_numpy(0.001 * self.random_state.rand(self.n_factors, 1)).float()

		self.Q_para=nn.Embedding(self.n_factors, 1, sparse=is_sparse)
		self.Q_para.weight.data = torch.from_numpy(np.ones((self.n_factors, 1))).float()

		self.relu = nn.ReLU()

	def forward(self, users_index, items_index):
		user_h1 = self.user_embeddings(users_index)
		item_h1 = self.item_embeddings(items_index)
		R_h = (user_h1 * item_h1).sum(1)

		return R_h

	def __call__(self, *args):
		return self.forward(*args)
	
	def recommend(self):
		U = self.user_embeddings.weight.detach()
		V = self.item_embeddings.weight.detach()
		return torch.matmul(U, torch.t(V))

	def update_U(self, R, Y, user_idx):
		V = self.item_embeddings.weight.detach()
		Vp=torch.matmul(V, torch.diag(self.P_para.weight.data.squeeze(1)))
		Vq=torch.matmul(V, torch.diag(self.Q_para.weight.data.squeeze(1)))
		VV = torch.matmul(torch.t(Vp), Vp) + self.alpha * torch.matmul(torch.t(Vq), Vq) + self.lam_u * self.I.diag()
		RV = torch.matmul(R, Vp)+self.alpha * torch.matmul(Y, Vq)
		self.user_embeddings.weight.data[user_idx] = torch.matmul(RV, torch.inverse(VV))[user_idx]

	def update_V(self, R, Y):
		U = self.user_embeddings.weight.data
		Up=torch.matmul(U, torch.diag(self.P_para.weight.data.squeeze(1)))
		Uq=torch.matmul(U, torch.diag(self.Q_para.weight.data.squeeze(1)))
		UU = torch.matmul(torch.t(Up), Up) + self.alpha * torch.matmul(torch.t(Uq), Uq) + self.lam_v * self.I.diag()
		RU = torch.matmul(torch.t(R), Up)+self.alpha * torch.matmul(torch.t(Y), Uq)
		self.item_embeddings.weight.data = torch.matmul(RU, torch.inverse(UU))

	def update_P(self, R):
		U = self.user_embeddings.weight.data
		V = self.item_embeddings.weight.data
		UVuns = U.unsqueeze(1) * V.unsqueeze(0)
		Fengmu = self.lam_p+torch.sum(UVuns * UVuns)
		Fengzi=torch.sum(U.unsqueeze(1) * V.unsqueeze(0) * R.unsqueeze(-1), dim=[0, 1])
		self.P_para.weight.data = (Fengzi/Fengmu).unsqueeze(-1)

	def update_Q(self, Y):
		U = self.user_embeddings.weight.data
		V = self.item_embeddings.weight.data
		UVuns = U.unsqueeze(1) * V.unsqueeze(0)
		Fengmu = self.lam_q+torch.sum(UVuns * UVuns)
		Fengzi=torch.sum(U.unsqueeze(1) * V.unsqueeze(0) * Y.unsqueeze(-1), dim=[0, 1])
		self.Q_para.weight.data = Fengzi/Fengmu

class PMFLoss(torch.nn.Module):
	def __init__(self, lam_u=0.1, lam_v=0.1, lam_p=0.1, lam_q=0.1):
		super().__init__()
		self.lam_u = lam_u
		self.lam_v = lam_v
		self.lam_p = lam_p
		self.lam_q = lam_q

	def forward(self, R, u_features, v_features, alpha, Y, P, Q): #ignore prediction_error_y and q_regularization
		non_zero_mask = (R != 0).type(torch.FloatTensor)
		predicted=torch.matmul(torch.t(P)*u_features,torch.t(v_features))
		diff = (R - predicted) ** 2
		prediction_error = torch.sum(diff * non_zero_mask)/2

		non_zero_mask_Y = (Y != 0).type(torch.FloatTensor)
		predicted_y = torch.matmul(torch.t(Q) * u_features, torch.t(v_features))
		diff_y = (Y - predicted_y) ** 2
		prediction_error_y = alpha*torch.sum(diff_y * non_zero_mask_Y) / 2

		u_regularization = self.lam_u * trace(u_features)/2
		v_regularization = self.lam_v * trace(v_features)/2
		p_regularization = self.lam_p * trace(P)/2
		q_regularization = self.lam_q * trace(Q)/2

		return prediction_error + u_regularization + v_regularization + p_regularization

