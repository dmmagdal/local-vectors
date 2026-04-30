
import math
from typing import Dict, List, Union
import gc
from datetime import timedelta
from tqdm import tqdm
import os

import lancedb
import pyarrow as pa


class LanceDBConnection:
	def __init__(self, path: str):
		'''
		Initialize a connection to a LanceDB database at the given path.
		@param: path (str), the file path to the LanceDB database. If the
			database does not exist at this path, it will be created.
		@return: returns nothing.
		'''
		self.db = lancedb.connect(path)
		self.valid_metrics = ["cosine", "l2", "dot", "hamming"]
	

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
		self.db.drop_table(table_name)


	def delete_all_tables(self) -> None:
		'''
		Delete all tables in the database.
		@return: returns nothing.
		'''
		self.db.drop_all_tables()


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


	def search_table(self, table_name: str, query_vector: List[Union[float, int]], top_k: int = 5, metric: str = "cosine") -> List[Dict]:
		'''
		Search a table in the database with the given name and query vector.
		@param: table_name (str), the name of the table to be searched.
		@param: query_vector (List[float]), the query vector to be used for
			searching the table. This should be a list of floats that 
			represents the vector to be used for searching the table.
		@param: top_k (int), the number of results to return. Default is 5.
		@param: metric (str), the distance metric to use for searching. Default is "cosine".
		@return: returns a List[Dict] of the search results. Each dictionary
			in the list represents a search result and contains the 
			following keys:
			- "id": the ID of the search result.
			- "vector": the vector of the search result (could be vector_full
				or vector_binary depending on the table schema).
			- "metadata": the metadata of the search result.
		'''
		table = self.open_table(table_name)

		if top_k <= 0:
			raise ValueError("top_k must be a positive integer")

		if metric not in self.valid_metrics:
			raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {self.valid_metrics}")
		
		return table.search(query_vector)\
			.metric(metric)\
			.limit(top_k)\
			.to_list()