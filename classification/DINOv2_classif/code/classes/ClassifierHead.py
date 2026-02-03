import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
	def __init__(self, embedding_size=768, num_classes=10, n_feature=1):
		super(ClassifierHead, self).__init__()
		self.n_feature = n_feature
		self.embedding_size = embedding_size * self.n_feature

		# Simplistic convolutional block
		if self.n_feature == 1:
			self.conv = nn.Sequential(
				nn.Conv2d(embedding_size, 128, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				# dropout
				nn.AdaptiveAvgPool2d(1),  # Global average pooling to reduce spatial dimensions
			)
		else:
			"""
            self.conv = nn.Sequential(
                nn.Conv2d(self.embedding_size, self.embedding_size // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.embedding_size // 2),
                nn.ReLU(),
                nn.Conv2d(self.embedding_size // 2, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            """
			self.conv = nn.Sequential(
				nn.Conv2d(self.embedding_size, 1024, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(1024),
				nn.ReLU(inplace=True),
				nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(512),
				nn.ReLU(inplace=True),
				nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Conv2d(256, 128, kernel_size=1, bias=False),  # Bottleneck
				nn.BatchNorm2d(128),
				nn.ReLU(inplace=True),
				nn.AdaptiveAvgPool2d(1),  # Output: [B, 128, 1, 1]
			)

		# Linear layer for classification
		self.fc = nn.Linear(128, num_classes)

	def forward(self, inputs):
		features = inputs["features"]
		img_res = inputs["image"].shape[-1]
		patch_size = img_res // 14

		if isinstance(features, list):
			# Remove CLS token before concat
			features_cleaned = []
			for i, f in enumerate(features):
				f = f[:, 1:, :]  # Remove CLS
				features_cleaned.append(f)
				features = torch.cat(features_cleaned, dim=-1)  # [B, 256, 2304]
		else:
			features = features[:, 1:, :]  # [B, 256, 768]

		B, S, D = features.shape
		# assert S == 256
		# assert D == self.embedding_size

		features = features.permute(0, 2, 1).contiguous()  # [B, 2304, 256]
		features = features.view(B, D, patch_size, patch_size)  # [B, 2304, 16, 16]

		x = self.conv(features)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x
