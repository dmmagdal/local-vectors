
import math
from typing import Dict, List, Union
import gc
from datetime import timedelta
from tqdm import tqdm
import os

import lancedb
import pyarrow as pa


class LancedDBConnection:
	def __init__(self, path: str):
		self.db = lancedb.connect(path)
	

	def table_names(self) -> List[str]:
		'''
		Get the names of the tables in the database.
		@return: returns a List[str] of the table names in the database.
		'''
		return self.db.table_names()


	def create_table(self, table_name: str, schema: pa.Schema) -> None:
		'''
		Create a table in the database with the given name and schema.
		@param: table_name (str), the name of the table to be created.
		@param: schema (pa.Schema), the schema of the table to be created.
		@return: returns nothing.
		'''
		self.db.create_table(table_name, schema=schema)
		

	def open_table(self, table_name: str) -> lancedb.Table:
		'''
		Open a table in the database with the given name.
		@param: table_name (str), the name of the table to be opened.
		@return: returns the opened table as a lancedb.Table object.
		'''
		return self.db.open_table(table_name)
	

	def delete_table(self, table_name: str) -> None:
		'''
		Delete a table in the database with the given name.
		@param: table_name (str), the name of the table to be deleted.
		@return: returns nothing.
		'''
		self.db.delete_table(table_name)


	def update_table(self, table_name: str, data: List[Dict], mode: str = "append") -> None:
		'''
		Update a table in the database with the given name and data.
		@param: table_name (str), the name of the table to be updated.
		@param: data (List[Dict]), the data to be used for updating the
			table. This should be a list of dictionaries where each 
			dictionary represents a row in the table and the keys of
			the dictionary correspond to the column names in the table.
		@param: mode (str), the mode for updating the table. This can be
			either "append" or "overwrite". Default is "append".
		@return: returns nothing.
		'''
		table = self.open_table(table_name)
		table.add(data, mode=mode)

		table.optimize(
			cleanup_older_than=timedelta(seconds=30)
		)
		gc.collect()


def embed_docs(
		db: DBConnection, 
		config: Dict,
		model_name: str, 
		model_config: Dict[str, Union[str, int]], 
		tokenizer: AutoTokenizer, 
		model: AutoModel, 
		data: Dataset, 
		batch_size: int,
		device: str = "cpu"
) -> None:
	'''
	Embed the documents/data to the database.
	@param: db (DBConnection), the vector database.
	@param: config (Dict), the configuration JSON containing all 
		relevant settings and parameters for the embedding models.
	@param: model_name (str), the name of the model.
	@param: model_config (Dict), the configuration JSON containing
		all relevant information for the model.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@param: model (AutoModel), the embedding model.
	@param: data (Dataset), the dataset that will be embedded to the 
		vector database.
	@param: batch_size (int), the size of the batches when passing the
		data to the embedding model.
	@param: device (str), tells where to map the model. Default is 
		"cpu".
	@return: returns nothing.
	'''
	# Initialize the table(s).
	dims = model_config["dims"]
	full_prec_table_name = f"{model_name}_fp32"
	binary_prec_table_name = f"{model_name}_binary"
	table_names = db.table_names()
	if full_prec_table_name not in table_names:
		schema = pa.schema([
			pa.field("wikidata_id", pa.string()),
			pa.field("text_idx", pa.int32()),
			pa.field("text_len", pa.int32()),
			pa.field("vector_full", pa.list_(pa.float32(), dims))
		])
		db.create_table(full_prec_table_name, schema=schema)

	if binary_prec_table_name not in table_names:
		dim_bytes = math.ceil(dims / 8)
		schema = pa.schema([
			pa.field("wikidata_id", pa.string()),
			pa.field("text_idx", pa.int32()),
			pa.field("text_len", pa.int32()),
			pa.field("vector_binary", pa.list_(pa.uint8(), dim_bytes))
		])
		db.create_table(binary_prec_table_name, schema=schema)

	full_prec_table = db.open_table(full_prec_table_name)
	binary_prec_table = db.open_table(binary_prec_table_name)

	chunk_metadata = list()

	# Iterate through the document ids.
	for entry in tqdm(data, desc="Embedding"):
		article_id, article_text = entry["wikidata_id"], entry["cleaned_text"]

		# Preprocess text (chunk it) for embedding.
		new_chunk_metadata = vector_preprocessing(
			article_text, config, model_config, tokenizer
		)

		# Embed each chunk and update the metadata.
		for idx, chunk in enumerate(new_chunk_metadata):
			# Update/add the metadata for the source filename
			# and article SHA1.
			chunk.update(
				{"wikidata_id": article_id}
			)

			# Get original text chunk from text.
			text_idx = chunk["text_idx"]
			text_len = chunk["text_len"]
			# text_chunk = article_text[text_idx: text_idx + text_len]
			# text_chunk = chunk["tokens"]
			chunk.update(
				{"text_idx": text_idx, "text_len": text_len}
			)

			# Pad out the token sequence if necessary.
			length_diff = model_config["max_tokens"] - len(chunk["tokens"])
			if length_diff != 0:
				tokens = chunk_tokens["tokens"]
				tokens.extend([tokenizer.pad_token_id] * length_diff)
				chunk.update({"tokens": tokens})

			# Embed the text chunk.
			# embeddings = embed_text(
			# 	text_chunk, tokenizer, model, device, to_binary=True
			# )
			# embedding, binary_embedding = embeddings
			# del chunk["tokens"]

			# NOTE:
			# Originally I had embeddings stored into the metadata
			# dictionary under the "embedding", key but lancddb
			# requires the embedding data be under the "vector"
			# name.

			# Update the chunk dictionary with the embedding
			# and set the value of that chunk in the metadata
			# list to the (updated) chunk.
			# chunk.update({"embedding": embedding})
			# chunk.update({"vector": embedding})
			# chunk.update({
			# 	"vector_full": embedding,
			# 	"vector_binary": binary_embedding,
			# })
			new_chunk_metadata[idx] = chunk

		# NOTE:
		# We need new_chunk_metadata to be separate from chunk_metadata
		# otherwise we risk overriding the existing metadata values
		# with those from the current document. We just attached/
		# append/extend chunk_metadata with new_chunk_metadata when all
		# the updates are finished.

		# Save the new chunk metadata to the master chunk metadata list
		# by extending it.
		chunk_metadata.extend(new_chunk_metadata)

		# If the chunk metadata list is sufficiently large, embed the 
		# text data within.
		if len(chunk_metadata) >= batch_size:
			# Initialize a list containing the tokens for each chunk in
			# the metadata. Embed those tokens with the model
			chunk_tokens = [chunk["tokens"] for chunk in chunk_metadata]
			embeddings, binary_embeddings = batch_embed_text(
				chunk_tokens, tokenizer, model, device, True
			)

			# Iterate through the embeddings, saving them with the 
			# respective chunk.
			for idx, chunk in enumerate(chunk_metadata):
				del chunk["tokens"]
				chunk.update({
					"vector_full": embeddings[idx, :],
					"vector_binary": binary_embeddings[idx, :]
				})

				chunk_metadata[idx] = chunk

			# Add chunk metadata to the vector database. Should be on
			# "append" mode by default.
			# table.add(chunk_metadata, mode="append")
			full_prec_table.add(
				[
					{
						k: v for k, v in chunk.items() 
						if "binary" not in k
					} for chunk in chunk_metadata
				], 
				mode="append"
			)
			binary_prec_table.add(
				[
					{
						k: v for k, v in chunk.items() 
						if "full" not in k
					} for chunk in chunk_metadata
				], 
				mode="append"
			)

			# Cleanup artifacts.
			full_prec_table.optimize(
				cleanup_older_than=timedelta(seconds=30)
			)
			binary_prec_table.optimize(
				cleanup_older_than=timedelta(seconds=30)
			)

			# Clear/reset chunk metadata list.
			chunk_metadata = list()
			gc.collect()