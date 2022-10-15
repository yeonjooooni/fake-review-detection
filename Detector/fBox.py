import numpy as np
from numpy.linalg import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


"""
	Implement the fBox detector
	Paper: Spotting Suspicious Link Behavior with fBox- An Adversarial Perspective, ICDM'17
"""

class fBox():
	def __init__(self, user_product_graph):
		"""
			fBox only takes a binary user-product graph
		"""
		self.u_id2idx = {}
		self.idx2u_id = {}
		self.p_id2idx = {}
		self.idx2p_id = {}

		# construct a sparse matrix from the graph
		row_idx = []
		col_idx = []
		data = []

		user_idx = 0
		product_idx = 0
		for k, reviews in user_product_graph.items():
			self.u_id2idx[k] = user_idx
			self.idx2u_id[user_idx] = k

			user_idx += 1

			for t in reviews:
				p_id = t[0]
				if p_id not in self.p_id2idx:
					self.p_id2idx[p_id] = product_idx
					self.idx2p_id[product_idx] = p_id
					product_idx += 1

				row_idx.append(self.u_id2idx[k])
				col_idx.append(self.p_id2idx[p_id])
				data.append(1)

		self.num_users = user_idx
		self.num_products = product_idx
		self.matrix = csr_matrix((data, (row_idx, col_idx)), shape=(user_idx, product_idx)).asfptype()

	def run(self, tau, k):
		"""
			run the algorithm.
			tau: the percentile in reconstructed degree threshold under which a node is considered suspicious
		"""
		# k = 50 is selected based on Figure 3 of the paper
		u, s, vt = svds(self.matrix, k=k)
		# reconstructed out degree
		self.recOutDeg = norm(u.dot(np.diag(s)), axis=1)
		# reconstructed in degree
		self.recInDeg = norm(vt.T.dot(np.diag(s)), axis=1)

		# detect users
		out_deg = self.matrix.sum(axis=1)
		self.out_deg = np.array(out_deg).reshape(-1, )
		self.unique_out_deg = np.unique(self.out_deg)

		# print ('unique number of out degrees %d' % self.unique_out_deg.shape[0])

		# store the indices of suspicious users
		suspicious_users = {}
		thresholds = {}
		for d in self.unique_out_deg:
			# find users with original degree = d
			users = (self.out_deg == d)
			user_deg = self.recOutDeg[users]
			thresholds[d] = np.percentile(user_deg, tau)

		for i in range(self.num_users):
			user_d = self.out_deg[i]
			if (self.recOutDeg[i] < thresholds[user_d]):

				if user_d not in suspicious_users:
					suspicious_users[user_d] = []
				suspicious_users[user_d].append(self.idx2u_id[i])

		# detect products
		in_deg = self.matrix.sum(axis=0)
		self.in_deg = np.array(in_deg).reshape(-1, )
		self.unique_in_deg = np.unique(self.in_deg)

		# print ('unique number of in degrees %d' % self.unique_in_deg.shape[0])

		# store the indices of suspicious users
		suspicious_products = {}
		thresholds = {}

		for d in self.unique_in_deg:
			prods = (self.in_deg == d)
			prod_deg = self.recInDeg[prods]
			thresholds[d] = np.percentile(prod_deg, tau)

		for i in range(self.num_products):
			prod_d = self.in_deg[i]
			if (self.recInDeg[i] < thresholds[prod_d]):
				if prod_d not in suspicious_products:
					suspicious_products[prod_d] = []
				suspicious_products[prod_d].append(self.idx2p_id[i])

		return suspicious_users, suspicious_products

	def get_srms(self):
		"""
			return two matrices one for use the other for products
			each matrix has rows as reconstruction degree and column as old degree in the graph.
		"""

		hist, edges = np.histogram(self.recOutDeg, bins=100)
		data = []
		rows = []
		cols = []

		for d in self.unique_out_deg:
			user_deg = self.recOutDeg[self.out_deg == d]
			bin_indices = np.digitize(user_deg, edges)
			for i in bin_indices:
				data.append(1)
				rows.append(i)
				cols.append(d)

		self.osrm = csr_matrix((data, (rows, cols)), shape=(len(edges) + 1, max(self.unique_out_deg) + 1))

		hist, edges = np.histogram(self.recInDeg, bins=10)
		data = []
		rows = []
		cols = []
		for d in self.unique_in_deg:
			prod_deg = self.recInDeg[self.in_deg == d]
			bin_indices = np.digitize(prod_deg, edges)
			for i in bin_indices:
				data.append(1)
				rows.append(i)
				cols.append(d)
		self.isrm = csr_matrix((data, (rows, cols)))

		return self.osrm, self.isrm
