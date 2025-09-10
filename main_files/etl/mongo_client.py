import os
from typing import List, Optional, Dict, Any
import pandas as pd
from pymongo import MongoClient, errors
from main_files.loggings.logger import logging  # adjust to your module
from main_files.constants import train_pipeline
from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL=os.getenv("MONGO_DB_URL")


class MongoDBClient:
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None, server_selection_timeout_ms: int = 5000):
        """
        Initialize MongoDB client and verify connectivity.
        """
        uri = uri or MONGO_DB_URL
        if not uri:
            raise ValueError("MONGO_DB_URL is not set in the environment")

        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=server_selection_timeout_ms)
            # fail fast: ping server
            self.client.admin.command("ping")
        except errors.PyMongoError as e:
            logging.exception("Unable to connect to MongoDB")
            raise

        self.db_name = db_name or train_pipeline.DATA_INGESTION_DATABASE_NAME
        self.db = self.client[self.db_name]
        logging.info(f"Connected to MongoDB database: {self.db_name}")

    def close(self) -> None:
        """Close underlying MongoClient connection."""
        try:
            self.client.close()
            logging.info("MongoDB client closed")
        except Exception:
            logging.exception("Error closing MongoDB client")

    def insert_one(self, collection_name: str, doc: Dict[str, Any]):
        """Insert single document. Returns inserted_id (ObjectId)."""
        try:
            col = self.db[collection_name]
            res = col.insert_one(doc)
            logging.debug(f"Inserted one into {collection_name}, id={res.inserted_id}")
            return res.inserted_id
        except errors.PyMongoError:
            logging.exception("insert_one failed")
            raise

    def insert_many(self, collection_name: str, docs: List[Dict[str, Any]], ordered: bool = False):
        """Insert many documents. Returns list of inserted ids."""
        if not docs:
            logging.warning("insert_many called with empty docs list")
            return []
        try:
            col = self.db[collection_name]
            res = col.insert_many(docs, ordered=ordered)
            logging.debug(f"Inserted {len(res.inserted_ids)} docs into {collection_name}")
            return res.inserted_ids
        except errors.BulkWriteError:
            logging.exception("Bulk write error during insert_many")
            raise
        except errors.PyMongoError:
            logging.exception("insert_many failed")
            raise

    def find(self, collection_name: str, query: Optional[Dict] = None, projection: Optional[Dict] = None,
             sort: Optional[List] = None, limit: Optional[int] = None, batch_size: Optional[int] = None) -> List[Dict]:
        """
        Run a find and return list of dict documents.
        - projection example: {'_id': 0, 'field1': 1}
        - sort example: [('timestamp', -1)]
        """
        try:
            col = self.db[collection_name]
            cursor = col.find(filter=query or {}, projection=projection)
            if sort:
                cursor = cursor.sort(sort)
            if batch_size:
                cursor = cursor.batch_size(batch_size)
            if limit:
                cursor = cursor.limit(limit)
            docs = list(cursor)
            logging.info(f"find: returned {len(docs)} docs from {collection_name}")
            return docs
        except errors.PyMongoError:
            logging.exception("find failed")
            raise

    def find_as_df(self, collection_name: str, query: Optional[Dict] = None, projection: Optional[Dict] = None,
                   convert_object_id: bool = False, limit: Optional[int] = None, batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Returns a pandas DataFrame from the query results.
        convert_object_id: if True, convert _id to its string representation instead of dropping it.
        """
        docs = self.find(collection_name=collection_name, query=query, projection=projection, limit=limit, batch_size=batch_size)
        if not docs:
            logging.info("No documents found; returning empty DataFrame")
            return pd.DataFrame()
        df = pd.DataFrame(docs)

        if "_id" in df.columns:
            if convert_object_id:
                df["_id"] = df["_id"].astype(str)
            else:
                df = df.drop(columns=["_id"])
        # sanitize common BSON types (example Decimal128 -> float) if needed here
        logging.debug(f"Converted query to DataFrame with shape {df.shape}")
        return df

    def drop_collection(self, collection_name: str):
        try:
            self.db[collection_name].drop()
            logging.info(f"Dropped collection: {collection_name}")
        except errors.PyMongoError:
            logging.exception("drop_collection failed")
            raise

    def ensure_index(self, collection_name: str, keys: list, unique: bool = False):
        """
        Create an index on the collection.
        keys example: [('timestamp', -1), ('farmer_id', 1)]
        """
        try:
            self.db[collection_name].create_index(keys, unique=unique)
            logging.info(f"Created index on {collection_name}: {keys}")
        except errors.PyMongoError:
            logging.exception("ensure_index failed")
            raise
