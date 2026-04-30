# providers.py
# Provides helper functions for detecting hardware accelerators and ]
# loading models.
# Python 3.11
# Windwos/MacOS/Linux


import shutil
from typing import Dict, Tuple, Union
import os
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


def detect_device(force_cpu: bool = False, get_count: bool = False) -> Union[str, Tuple[str, int]]:
	'''
	Detect what hardware accelerator devices are available on the 
		system.
	@param: force_cpu (bool), whether to force the use of CPU even if other
		devices are available. Default is False.
	@param: get_count (bool), whether to return the count of available
		devices (only applicable for CUDA). Default is False.
	@return: returns the name of the best available device as a string. 
		If get_count is True and CUDA is available, returns a tuple of 
		the device name and the count of available CUDA devices.
	'''
	# Detect GPU.
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"

		# Get the number of GPUs available for Nvidia CUDA devices.
		if get_count:
			count = torch.cuda.device_count()
	elif torch.backends.mps.is_available():
		device = "mps"
		
	# If the user wants to force CPU, override the detected device.
	if force_cpu:
		device = "cpu"

	# If the user is planning on using Nvidia CUDA devices (has not
	# forced CPU usage), return the device and device count.
	if device == "cuda" and get_count:
		return device, count
		
	# Return the deivce as a string.
	return device


def get_model_metadata(model_save_path: str) -> Dict[str, Union[str, int]]:
	'''
	Retrieve the model metadata around its context length and embedding 
		dimensions.
	@param: model_save_path (str), 
	@return: returns a dictionary containing the model metadata, 
		including the model ID, maximum token length, and embedding 
		dimensions.
	'''
	# This only downloads the config.json, not the weights
	config = AutoConfig.from_pretrained(model_save_path)
	
	# Common attribute names across BERT, RoBERTa, etc.
	# Note: 'max_position_embeddings' is the standard for max_tokens
	# 'hidden_size' is the standard for embedding dimensions
	return {
		"model_id": getattr(config, "_name_or_path", "N/A"),
		"max_tokens": getattr(config, "max_position_embeddings", "N/A"),
		"dims": getattr(config, "hidden_size", "N/A")
	}


def load_model(
		model_id: str,
		model_save_root: str = Path.home() / ".cache" / "local-vectors" / "models",
		device: str = "cpu"
) -> Tuple[AutoTokenizer, AutoModel]:
	'''
	Load the tokenizer and model. Download them if they're not found 
		locally.
	@param: model_id (str), the ID of the model as it is saved in
		Hugging Face.
	@param: model_save_root (str), the root directory where the model
		is saved locally. Default is "~/.cache/local-vectors/models".
	@param: device (str), tells where to map the model. Default is 
		"cpu".
	@return: returns the tokenizer and model for embedding the text.
	'''
	# Check for the local copy of the model. If the model doesn't have
	# a local copy (the path doesn't exist), download it.
	model_path = model_save_root / model_id.replace("/", "_")
	
	# Check for path and that path is a directory. Make it if either is
	# not true.
	if not os.path.exists(model_path) or not os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)

	# Check for path the be populated with files (weak check). Download
	# the tokenizer and model and clean up files once done.
	if len(os.listdir(model_path)) == 0:
		print(f"Model {model_id} needs to be downloaded.")

		# Check for internet connection (also checks to see that
		# huggingface is online as well). Exit if fails.
		response = requests.get("https://huggingface.co/")
		if response.status_code != 200:
			print(f"Request to huggingface.co returned unexpected status code: {response.status_code}")
			print(f"Unable to download {model_id} model.")
			exit(1)

		# Create cache path folders.
		cache_path = str(model_save_root / model_id.replace("/", "_")) + "_tmp"
		os.makedirs(cache_path, exist_ok=True)
		os.makedirs(model_path, exist_ok=True)

		# Load tokenizer and model.
		tokenizer = AutoTokenizer.from_pretrained(
			model_id, cache_dir=cache_path, device_map=device
		)
		model = AutoModel.from_pretrained(
			model_id, cache_dir=cache_path, device_map=device,
			trust_remote_code=True, use_safetensors=True
		)

		# Load the model metadata and save it to the save path.
		AutoConfig.from_pretrained(
			model_id, 
			cache_dir=model_path
		)

		# Save the tokenizer and model to the save path.
		tokenizer.save_pretrained(model_path)
		model.save_pretrained(model_path)

		# Delete the cache.
		shutil.rmtree(cache_path)
	
	# Load the tokenizer and model.
	tokenizer = AutoTokenizer.from_pretrained(
		model_path, 
		device_map=device
	)
	model = AutoModel.from_pretrained(
		model_path, 
		device_map=device,
		trust_remote_code=True, 
		use_safetensors=True
	)

	# Return the tokenizer and model.
	return tokenizer, model