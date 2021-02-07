import torch

def pairwise_distances(embeddings, device, squared=False):
	"""
	||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
	"""
	# get dot product (batch_size, batch_size)
	dot_product = embeddings.mm(embeddings.t())

	# a vector
	square_sum = dot_product.diag()

	distances = square_sum.unsqueeze(1) - 2*dot_product + square_sum.unsqueeze(0)

	distances = distances.clamp(min=0)

	if not squared:
		epsilon=1e-16
		mask = torch.eq(distances, 0).float()
		distances += mask * epsilon
		# distances = distances + mask * epsilon

		distances = torch.sqrt(distances)
		# distances *= (1-mask)
		distances = distances * (1-mask)
	return distances

def get_valid_positive_mask(labels, device):
	"""
	To be a valid positive pair (a,p),
		- a and p are different embeddings
		- a and p have the same label
	"""
	indices_equal = torch.eye(labels.size(0)).byte()
	indices_not_equal = ~indices_equal

	label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

	mask = indices_not_equal.cuda() & label_equal
	return mask

def get_valid_negative_mask(labels, device):
	"""
	To be a valid negative pair (a,n),
		- a and n are different embeddings
		- a and n have the different label
	"""
	indices_equal = torch.eye(labels.size(0)).byte()
	indices_not_equal = ~indices_equal

	label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

	mask = indices_not_equal.cuda() & label_not_equal
	return mask


def get_valid_triplets_mask(labels, device):
	"""
	To be valid, a triplet (a,p,n) has to satisfy:
		- a,p,n are distinct embeddings
		- a and p have the same label, while a and n have different label
	"""
	indices_equal = torch.eye(labels.size(0)).byte()
	indices_not_equal = ~indices_equal
	i_ne_j = indices_not_equal.unsqueeze(2)
	i_ne_k = indices_not_equal.unsqueeze(1)
	j_ne_k = indices_not_equal.unsqueeze(0)
	distinct_indices = i_ne_j & i_ne_k & j_ne_k

	label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
	i_eq_j = label_equal.unsqueeze(2)
	i_eq_k = label_equal.unsqueeze(1)
	i_ne_k = ~i_eq_k
	valid_labels = i_eq_j & i_ne_k

	# print("i_eq_j:", i_eq_j)
	# print("i_ne_k:", i_ne_k)

	# print('distinct_indices:', distinct_indices)
	# print('valid_labels:', valid_labels)

	mask = distinct_indices.bool().cuda() & valid_labels
	return mask

def batch_all_triplet_loss(labels, embeddings, device, margin=1, squared=False):
	"""
	get triplet loss for all valid triplets and average over those triplets whose loss is positive.
	"""

	distances = pairwise_distances(embeddings, device, squared=squared)

	anchor_positive_dist = distances.unsqueeze(2)
	anchor_negative_dist = distances.unsqueeze(1)
	triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

	# print('anchor_positive_dist:', anchor_positive_dist)
	# print('anchor_negative_dist:', anchor_negative_dist)

	# get a 3D mask to filter out invalid triplets
	mask = get_valid_triplets_mask(labels, device)

	triplet_loss = triplet_loss * mask.float()
	triplet_loss.clamp_(min=0)

	# count the number of positive triplets
	epsilon = 1e-16
	num_positive_triplets = (triplet_loss > 0).float().sum()
	num_valid_triplets = mask.float().sum()
	fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + epsilon)

	triplet_loss = triplet_loss.sum() / (num_positive_triplets + epsilon)

	return triplet_loss, fraction_positive_triplets

def batch_hard_triplet_loss(labels, embeddings, device, margin=1, squared=False):
	"""
	- compute distance matrix
	- for each anchor a0, find the (a0,p0) pair with greatest distance s.t. a0 and p0 have the same label
	- for each anchor a0, find the (a0,n0) pair with smallest distance s.t. a0 and n0 have different label
	- compute triplet loss for each triplet (a0, p0, n0), average them
	"""
	distances = pairwise_distances(embeddings, device, squared=squared)

	mask_positive = get_valid_positive_mask(labels, device)
	hardest_positive_dist = (distances * mask_positive.float()).max(dim=1)[0]

	mask_negative = get_valid_negative_mask(labels, device)
	max_negative_dist = distances.max(dim=1,keepdim=True)[0]
	distances = distances + max_negative_dist * (~mask_negative).float()
	hardest_negative_dist = distances.min(dim=1)[0]

	triplet_loss = (hardest_positive_dist - hardest_negative_dist + margin).clamp(min=0)
	triplet_loss = triplet_loss.mean()

	return triplet_loss