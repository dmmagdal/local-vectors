
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