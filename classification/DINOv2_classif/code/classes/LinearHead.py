import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
	def __init__(self, embedding_size=512, num_classes=3, n_features=1):
		super(LinearHead, self).__init__()
		self.n_features = n_features
		self.embedding_size = embedding_size * self.n_features
		self.num_classes = num_classes
		self.head = nn.Sequential(
			nn.BatchNorm2d(self.embedding_size),
			nn.Conv2d(self.embedding_size, self.num_classes, kernel_size=1, padding=0, bias=True),
		)

	def forward(self, inputs):
		features = inputs["features"]
		img_shape = inputs["image"].shape[-1]
		patch_feature_size = img_shape // 14
		if self.n_features > 1:
			features = torch.cat(features, dim=-1)
		features = features[:, 1:].permute(0, 2, 1).reshape(-1, self.embedding_size, patch_feature_size, patch_feature_size)
		logits = self.head(features)
		logits = F.interpolate(input=logits, size=(int(img_shape), int(img_shape)), mode="bilinear", align_corners=False)
		return logits
