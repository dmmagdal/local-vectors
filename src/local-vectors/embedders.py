# embedders.py
# Provides a class that encapsulates all the required functions for 
# embedding text data to vectors. This includes functions for 
# preprocessing the text data (chunking) and embedding the text data in 
# batches.
# Python 3.11
# Windwos/MacOS/Linux


import copy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from providers import load_model, get_model_metadata


def vector_preprocessing(
		article_text: str, 
		overlap: int, 
		model_config: Dict[str, Union[str, int]],
		tokenizer: AutoTokenizer, 
		truncate: bool = False,
		recursive_split: bool = False
) -> List[Dict]:
	'''
	Preprocess the text to yield a list of chunks of the tokenized 
		text. Each chunk is the longest possible set of text that can 
		be passed to the embedding model tokenizer.
	@param: text (str), the raw text that is to be processed for
		storing to vector database.
	@param: overlap (int), the number of overlapping tokens between
		consecutive chunks.
	@param: model_config (Dict), the configuration JSON containing all
		model-specific parameters. This is used to pull the model's 
		context length for the chunking.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@param: truncate (bool), whether to truncate the text if it exceeds
		the model context length. Is false by default.
	@param: recursive_split (bool), whether to use the recursive 
		splitting scheme for long text chunks or a more basic one. Is
		false by default.
	@return: returns a List[Dict] of the text metadata. This metadata 
		includes the split text's token sequence, index (with respect
		to the input text), and length of the text split for each split
		in the text.
	'''
	# Pull the model's context length from the configuration.
	context_length = model_config["max_tokens"]

	# Make sure that the overlap does not exceed the model context
	# length.
	# assert overlap < context_length, f"Number of overlapping tokens ({overlap}) must NOT exceed the model context length ({context_length})"

	# NOTE:
	# Initially there were plans to have text tokenized and chunked by
	# token (chunk lengths would be context_length with overlap number
	# of tokens overlapping). This proved to be more complicated than
	# thought because it required tokens be decoded back to the
	# original text exactly, something that is left up to the
	# implementation of each model's tokenizer. To allow for support of
	# so many models, there had to be a more general method to handle
	# text tokenization while keeping track of the original text
	# metadata. 

	# NOTE:
	# Splitting scheme 1 (recursive split):
	# 1) Split into paragraphs (split by newline ("\n\n", "\n") 
	#	characters). This is covered by the high_level_split() 
	#	recursive function.
	# 2) Split paragraphs that are too long (split by " " (word level 
	#	split) and "" (character level split)). This is covered by the
	#	low_level_split() recursive function that is called by the
	#	high_level_split() recursive function when such is the case.
	# Splitting scheme 2 (direct/basic split):
	# 1) Split into paragraphs (split by newline ("\n\n", "\n") 
	#	characters). 
	# 2) Split paragraphs that are too long (split by token lengths
	#	with some overlap, recycle the text metadata for all chunks in
	# 	that paragraph).

	# Initialize splitters list and text metadata list. The splitters
	# are the same as default on RecursiveCharacterTextSplitter.
	splitters = ["\n\n", "\n", " ", ""] 
	metadata = []

	# Add to the metadata list by passing the text to the high level
	# recursive splitter function.
	if recursive_split:
		metadata += high_level_split(
			article_text, 0, tokenizer, context_length, splitters, truncate
		)
	else:
		metadata = direct_split(
			article_text, overlap, tokenizer, context_length, splitters[0], truncate
		)
	
	# Return the text metadata.
	return metadata


def direct_split(
		text: str, 
		offset: int, 
		tokenizer: AutoTokenizer, 
		context_length: int, 
		splitter: str,
		truncate: bool = False,
) -> List[Dict]:
	'''
	(Directly) split the text into paragraphs and extract the 
		metadata from the text slices of the input text. If the 
		paragraphs are too large, chunk the text with some overlap and
		extract the metadata from there too.
	@param: text (str), the text that is to be processed for storing to
		vector database.
	@param: offset (int), the index of the input text with respect to
		the original text.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@param: context_length (int), the maximum number of tokens 
		supported by the model. This helps us chunk the text if the 
		tokenized output is "too long".
	@param: splitters (str), the string that will be used to split the 
		text. For this function, we expect the "top-most" strings to be 
		either in the set ("\n\n", "\n").
	@param: truncate (bool), whether to truncate the text if it exceeds
		the model context length. Is false by default.
	@return: returns a List[Dict] of the text metadata. This metadata 
		includes the split text's token sequence, index (with respect
		to the input text), and length of the text split for each split
		in the text.
	'''
	assert offset < context_length, \
		f"offset ({offset}) must be less than the maximum context length ({context_length})"
	
	# Initialize the metadata list.
	metadata = []

	# Split the text.
	if truncate:
		text_splits = [text]
	else:
		text_splits = text.split(splitter)

	# Iterate through the list 
	for split in text_splits:
		# Skip the split if it is an empty string.
		if split == "":
			continue

		# Get the split metadata (index with respect to original text 
		# plus offset and split length).
		split_idx = text.index(split) #+ offset
		split_len = len(split)

		# Tokenize the split.
		tokens = tokenizer.encode(split, add_special_tokens=False)

		if len(tokens) <= context_length or truncate:
			# If the token sequence is less than or equal to the 
			# context length, tokenize the text split again (this time
			# with padding), and add the entry to the metadata.
			tokens = tokenizer.encode(
				split, 
				add_special_tokens=False, 
				padding="max_length",
				max_length=context_length,
			)
			metadata.append({
				"tokens": tokens,
				"text_idx": split_idx,
				"text_len": split_len
			})
		else:
			# If the token sequence is greater than the context length,
			# split the embeddings/tokens with some overlap and recycle
			# the text metadata for all splits.
			step = context_length - offset
			pad_token_id = tokenizer.pad_token_id

			for start in range(0, len(tokens), step):
				end = start + context_length
				chunk = tokens[start:end]

				# Pad last chunk if shorter than context_length
				if len(chunk) < context_length:
					chunk = chunk + [pad_token_id] * (context_length - len(chunk))

				metadata.append({
					"tokens": chunk,
					"text_idx": split_idx,
					"text_len": split_len
				})

				if end >= len(tokens):
					break

	assert(
		all(len(data["tokens"]) == context_length for data in metadata)
	), f"Expected all tokens to be the {context_length} long."

	# Return the metadata.
	return metadata


def high_level_split(
		text: str, 
		offset: int, 
		tokenizer: AutoTokenizer, 
		context_length: int, 
		splitters: List[str],
		truncate: bool = False,
) -> List[Dict]:
	'''
	(Recursively) split the text into paragraphs and extract the 
		metadata from the text slices of the input text. If the 
		paragraphs are too large, call the low_level_split() recursive 
		function and extract the metadata from there too.
	@param: text (str), the text that is to be processed for storing to
		vector database.
	@param: offset (int), the index of the input text with respect to
		the original text.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@param: context_length (int), the maximum number of tokens 
		supported by the model. This helps us chunk the text if the 
		tokenized output is "too long".
	@param: splitters (List[str]), the list of strings that will be 
		used to split the text. For this function, we expect the 
		"top-most" strings to be either in the set ("\n\n", "\n").
	@param: truncate (bool), whether to truncate the text if it exceeds
		the model context length. Is false by default.
	@return: returns a List[Dict] of the text metadata. This metadata 
		includes the split text's token sequence, index (with respect
		to the input text), and length of the text split for each split
		in the text.
	'''
	# Check that the splitters is non-empty.
	assert len(splitters) >= 1, "Expected high_level_split() argument 'splitters' to be populated"
	
	# Check the "top"/"first" splitter. Make sure that it is for
	# splitting the text at the paragraph level.
	valid_splitters = ["\n\n", "\n"]
	splitters_copy = copy.deepcopy(splitters)
	splitter = splitters_copy.pop(0)
	assert splitter in valid_splitters, "Expected first element for high_level_split() argument 'splitter' to be either '\\n\\n' or '\\n'"

	# Initialize the metadata list.
	metadata = []

	# Split the text.
	if truncate:
		text_splits = [text]
	else:
		text_splits = text.split(splitter)

	# Iterate through the list 
	for split in text_splits:
		# Skip the split if it is an empty string.
		if split == "":
			continue

		# Get the split metadata (index with respect to original text 
		# plus offset and split length).
		split_idx = text.index(split) + offset
		split_len = len(split)

		# Tokenize the split.
		tokens = tokenizer.encode(split, add_special_tokens=False)

		if len(tokens) <= context_length or truncate:
			# If the token sequence is less than or equal to the 
			# context length, tokenize the text split again (this time
			# with padding), and add the entry to the metadata.
			tokens = tokenizer.encode(
				split, 
				add_special_tokens=False, 
				padding="max_length"
			)
			metadata.append({
				"tokens": tokens,
				"text_idx": split_idx,
				"text_len": split_len
			})
		else:
			# If the token sequence is greater than the context length,
			# pass the text over to the next splitter. Check the next
			# splitter and use the appropriate function.
			next_splitter = splitters_copy[0]
			if next_splitter in valid_splitters:
				metadata += high_level_split(
					split, 
					split_idx, 
					tokenizer, 
					context_length, 
					splitters_copy
				)
			else:
				metadata += low_level_split(
					split, 
					split_idx, 
					tokenizer, 
					context_length, 
					splitters_copy
				)

	# Return the metadata.
	return metadata


def low_level_split(
		text: str, 
		offset: int, 
		tokenizer: AutoTokenizer, 
		context_length: int, 
		splitters: List[str]
) -> List[Dict]:
	'''
	(Recursively) split the text into words or characters and extract
		the metadata from the text slices of the input text. If the 
		splits are too large, recursively call the function until the 
		text becomes manageable.
	@param: text (str), the text that is to be processed for storing to
		vector database.
	@param: offset (int), the index of the input text with respect to
		the original text.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@param: context_length (int), the maximum number of tokens 
		supported by the model. This helps us chunk the text if the 
		tokenized output is "too long".
	@param: splitters (List[str]), the list of strings that will be 
		used to split the text. For this function, we expect the 
		"top-most" strings to be either in the set (" ", "").
	@return: returns a List[Dict] of the text metadata. This metadata 
		includes the split text's token sequence, index (with respect
		to the input text), and length of the text split for each split
		in the text.
	'''
	# Check that the splitters is non-empty.
	assert len(splitters) >= 1, "Expected low_level_split() argument 'splitters' to be populated"
	
	# Check the "top"/"first" splitter. Make sure that it is for
	# splitting the text at the paragraph level.
	valid_splitters = [" ", ""]
	splitters_copy = copy.deepcopy(splitters)	# deep copy because this variable is modified
	splitter = splitters_copy.pop(0)
	assert splitter in valid_splitters, "Expected first element for low_level_split() argument 'splitter' to be either ' ' or ''"

	# Initialize the metadata list.
	metadata = []

	# Initialize a boolean to determine if the function needs to use
	# the next splitter in the recursive call or stick with the current
	# one. Initialize to True.
	use_next_spitter = True

	# Split the text.
	if splitter != "":
		# Split text "normally" (splitter is not an empty string "").
		text_splits = text.split(splitter)
	else:
		# Split text here if the splitter is "". The empty string "" is
		# not recognized as a valid text separator.
		text_splits = list(text)

	# Aggregate the splits according to the splitter. Current
	# aggregation strategy is to chunk the splits by half.
	half_len = len(text_splits) // 2
	if half_len > 0:	# Same as len(text_splits) > 1
		# This aggregation only takes affect if the number of items
		# resulting from the split is more than 1. Otherwise, there is
		# no need to aggregate.
		text_splits = [
			splitter.join(text_splits[:half_len]),
			splitter.join(text_splits[half_len:]),
		]

		# Flip boolean to False while the split list is still longer
		# than one item.
		use_next_spitter = False

	# Iterate through the list 
	for split in text_splits:
		# Skip the split if it is an empty string.
		if split == "":
			continue

		# Get the split metadata (index with respect to original text 
		# plus offset and split length).
		split_idx = text.index(split) + offset
		split_len = len(split)

		# Tokenize the split.
		tokens = tokenizer.encode(split, add_special_tokens=False)

		if len(tokens) <= context_length:
			# If the token sequence is less than or equal to the 
			# context length, tokenize the text split again (this time
			# with padding), and add the entry to the metadata.
			tokens = tokenizer.encode(
				split, 
				add_special_tokens=False, 
				padding="max_length"
			)
			metadata.append({
				"tokens": tokens,
				"text_idx": split_idx,
				"text_len": split_len
			})
		else:
			# If the token sequence is greater than the context length,
			# pass the text over to the next splitter. Since we are
			# already on the low level split function, we'll just
			# recursively call the function again.
			if not use_next_spitter:
				# If the boolean around using the next splitter is
				# False, re-insert the current splitter to the
				# beginning of the splitters list before it is passed
				# down to the recursive function call.
				splitters_copy.insert(0, splitter)

			metadata += low_level_split(
				split, 
				split_idx, 
				tokenizer, 
				context_length, 
				splitters_copy
			)

	# Return the metadata.
	return metadata


def batch_embed_text(
		text: Union[List[str], List[List[int]]], 
		tokenizer: AutoTokenizer, 
		model: AutoModel,
		device: str = "cpu",
		to_binary: bool = False
	) -> Tuple[np.array]:
	'''
	Embed the text in batches.
	@param: text (List[str] | List[List[int]]), the text that is to be 
		embedded. Text is either already batched in raw string form 
		(str) or tokenized form (List[int]).
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@param: model (AutoModel), the embedding model.
	@param: device (str), tells where to map the model. Default is 
		"cpu".
	@param: to_binary (bool), whether to also return the binary 
		embeddings. Default is False.
	@return: returns a tuple containing either the full precision fp32
		embeddings or both the full precision embeddings and binary 
		embeddings if to_binary is True. All embeddings are numpy 
		arrays.
	'''
	not_list = not isinstance(text, list)
	list_of_str = all(isinstance(txt, str) for txt in text)
	list_of_list_of_int = all((all(val, int) for val in int_list) for int_list in text)

	if not_list or (not list_of_str and not list_of_list_of_int):
		raise ValueError(f"Expected text to be either string or List[int]. Recieved {type(text)}")

	# Disable gradients.
	with torch.no_grad():
		if list_of_str:
			# Pass original text chunk to tokenizer. Ensure the data is
			# passed to the appropriate (hardware) device.
			output = model(
				**tokenizer(
					text,
					add_special_tokens=False,
					padding="max_length",
					return_tensors="pt"
				).to(device)
			)
		elif list_of_list_of_int:
			# input_ids = torch.tensor([text]).to(device)
			input_ids = torch.tensor(text).to(device)
			attention_mask = torch.tensor(
				[
					get_attention_mask(text_item, tokenizer.pad_token_id)
					for text_item in text
				]
			).to(device)
			output = model(
				input_ids=input_ids, attention_mask=attention_mask
			)
		else:
			raise ValueError(f"Expected text to be either string or List[int]. Recieved {type(text)}")

		# Compute the embedding by taking the mean of the last 
		# hidden state tensor across the seq_len axis.
		embedding = output[0].mean(dim=1)

		# Apply the following transformations to allow the
		# embedding to be compatible with being stored in the
		# vector DB (lancedb):
		#	1) Send the embedding to CPU (if it's not already
		#		there)
		#	2) Convert the embedding to numpy and flatten the
		# 		embedding to a 1D array
		embedding = embedding.to("cpu")
		embedding = embedding.numpy()#[0]

		# Generate binary embeddings if specified.
		if to_binary:
			binary_embedding = (embedding > 0).astype(np.uint8)
			binary_embedding = np.packbits(binary_embedding, axis=-1)
			return (embedding, binary_embedding)
	
	# Return the embedding.
	return (embedding)


def get_attention_mask(tokens: List[int], pad_token_id: int) -> List[int]:
	'''
	Generate the attention mask for the tokens.
	@param: tokens (List[int]), the list of token ids.
	@param: pad_token_id (int), the id of the padding token.
	@return: returns the attention mask where the value is 1 for all 
		non-padding tokens and 0 for all padding tokens.
	'''
	return [0 if t == pad_token_id else 1 for t in tokens]


class LocalEmbedder:
	'''
	Class for embedding the text data to the vector database. This class
		contains methods for embedding the text data in batches and 
		managing the embedding process.
	'''
	def __init__(
		self, 
		model_id: str, 
		model_save_root: str = Path.home() / ".cache" / "local-vectors" / "models", 
		token_overlap: int = 128,
		device: str = "cpu",
	) -> None:
		self.tokenizer, self.model = load_model(
			model_id, model_save_root, device
		)
		self.model_metadata = get_model_metadata(
			model_save_root / model_id.replace("/", "_")
		)
		self.model_id = model_id
		self.overlap = token_overlap
		self.device = device


	def set_device(self, device: str) -> None:
		self.device = device
		self.model.to(device)
	

	def embed_text(
		self, 
		text: str, 
		truncate: bool = False, 
		to_binary: bool = False
	) -> List[Dict[str, Union[np.array, bytes]]]:
		
		chunk_metadata = list()

		# Preprocess text (chunk it) for embedding.
		new_chunk_metadata = vector_preprocessing(
			text, self.overlap, self.model_metadata, self.tokenizer, truncate
		)

		# Embed each chunk and update the metadata.
		for idx, chunk in enumerate(new_chunk_metadata):
			# Get original text chunk from text.
			text_idx = chunk["text_idx"]
			text_len = chunk["text_len"]
			chunk.update(
				{"text_idx": text_idx, "text_len": text_len}
			)

			# Pad out the token sequence if necessary.
			length_diff = self.model_metadata["max_tokens"] - len(chunk["tokens"])
			if length_diff != 0:
				tokens = chunk_tokens["tokens"]
				tokens.extend([self.tokenizer.pad_token_id] * length_diff)
				chunk.update({"tokens": tokens})

			# Update the chunk dictionary with the embedding
			# and set the value of that chunk in the metadata
			# list to the (updated) chunk.
			new_chunk_metadata[idx] = chunk

		# Save the new chunk metadata to the master chunk metadata list
		# by extending it.
		chunk_metadata.extend(new_chunk_metadata)

		# Initialize a list containing the tokens for each chunk in
		# the metadata. Embed those tokens with the model
		chunk_tokens = [chunk["tokens"] for chunk in chunk_metadata]
		output_embeddings = batch_embed_text(
			chunk_tokens, self.tokenizer, self.model, self.device, to_binary
		)
		if to_binary:
			embeddings, binary_embeddings = output_embeddings
		else:
			embeddings = output_embeddings

		# Iterate through the embeddings, saving them with the 
		# respective chunk.
		for idx, chunk in enumerate(chunk_metadata):
			del chunk["tokens"]
			chunk.update({
				"vector_full": embeddings[idx, :],
			})
			
			if to_binary:
				chunk.update({
					"vector_binary": binary_embeddings[idx, :]
				})

			chunk_metadata[idx] = chunk

		return chunk_metadata
