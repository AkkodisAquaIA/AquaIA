import torch
import torch.nn as nn
from transformers import AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from classes.ClassifierHead import ClassifierHead


class DinoV2Segmentor(nn.Module):

	REQUIRED_PARAMS = {
		'num_classes': int
	}

	OPTIONAL_PARAMS = {
		'channels': int,
		'linear_head' : bool,
		'k_size': int, 
		'activation': str, 
		'size' : str,
		'n_features' : int,
		'peft' : bool,
		'quantize' : bool,
		'r' : int,
		'lora_alpha' : int,
		'lora_dropout' : float,
		'inference_mode' : bool
	}

	emb_size = {
		"small" : 384,
		"base" : 768,
		"large" : 1024,
	}

	def __init__(
			self, 
			channels=512, 
			num_classes=3, 
			k_size=3, 
			activation='relu', 
			size='base', 
			n_features=1, 
			peft=True, 
			quantize=True, 
			r=32,
			lora_alpha=32,
			lora_dropout=0.1,
			inference_mode=False
		):

		super(DinoV2Segmentor, self).__init__()
		self.inference_mode=inference_mode
		self.channels = channels
		self.num_classes = num_classes
		self.k_size = k_size
		self.activation = activation
		assert size in self.emb_size.keys(), "Invalid size embedding size"
		self.embedding_size = self.emb_size[str(size)]
		self.n_features = n_features
		self.peft = peft
		self.quantize = quantize
		self.r = r
		self.lora_alpha = lora_alpha
		self.lora_dropout = lora_dropout

		if self.quantize :
			self.quantization_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_use_double_quant=True,
				bnb_4bit_compute_dtype=torch.bfloat16,
			)
			self.backbone = AutoModel.from_pretrained(f'facebook/dinov2-{size}', quantization_config=self.quantization_config)
			self.backbone = prepare_model_for_kbit_training(self.backbone)
		else:
			self.backbone = AutoModel.from_pretrained(f'facebook/dinov2-{size}')

		if self.peft:
			peft_config = LoraConfig(inference_mode=self.inference_mode, r=self.r , lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout, target_modules="all-linear", use_rslora=True)
			self.backbone = get_peft_model(self.backbone, peft_config)
			self.backbone.print_trainable_parameters()

		
		self.seg_head = ClassifierHead(self.embedding_size, self.num_classes, self.n_features)
		
		print(f"Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

	def forward(self, x):
		# frozen weights of dino
		with torch.set_grad_enabled(self.peft and self.training):
			if self.n_features == 1:
				features = self.backbone(pixel_values=x).last_hidden_state
			else:
				features = list(self.backbone(pixel_values=x, output_hidden_states=True)['hidden_states'])[-self.n_features:]
		inputs = {"features" : features, "image" : x}
		return self.seg_head(inputs)

