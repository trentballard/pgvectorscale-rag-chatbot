import logging
import time
import uuid
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
import psycopg2
import psycopg2.extras
import numpy as np
from app.config.settings import get_settings
from openai import OpenAI


class VectorStore:
    """A class for managing vector operations and database interactions using pgvector."""

    def __init__(self):
        """Initialize the VectorStore with settings and OpenAI client."""
        self.settings = get_settings()
        self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model
        self.vector_settings = self.settings.vector_store
        self.table_name = self.vector_settings.table_name
        self.connection_string = self.settings.database.service_url
        self.embedding_dimensions = self.vector_settings.embedding_dimensions
        
        # Set a default connection string if none is found in settings
        if not self.connection_string:
            logging.warning("No connection string found in settings, using default")
            self.connection_string = "postgres://postgres:password@localhost:5432/postgres"
        
        logging.info(f"Using database connection: {self.connection_string}")

    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(self.connection_string)

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = (
            self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_model,
            )
            .data[0]
            .embedding
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self) -> None:
        """Create the necessary tables in the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Create extension if it doesn't exist
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create table with vector support
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY,
                    metadata JSONB,
                    contents TEXT,
                    embedding VECTOR({self.embedding_dimensions}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """)
                conn.commit()
                logging.info(f"Created table {self.table_name}")
        finally:
            conn.close()

    def create_index(self) -> None:
        """Create an index to speed up similarity search"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Create IVFFlat index for faster search
                cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                ON {self.table_name}
                USING ivfflat (embedding vector_l2_ops)
                WITH (lists = 100);
                """)
                conn.commit()
                logging.info(f"Created index on {self.table_name}")
        finally:
            conn.close()

    def drop_index(self) -> None:
        """Drop the index in the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP INDEX IF EXISTS {self.table_name}_embedding_idx;")
                conn.commit()
                logging.info(f"Dropped index from {self.table_name}")
        finally:
            conn.close()

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                psycopg2.extras.register_uuid()
                
                # Convert DataFrame to a list of tuples for insertion
                data = [(
                    uuid.UUID(row.id), 
                    psycopg2.extras.Json(row.metadata),  # Convert dict to Json object
                    row.contents, 
                    row.embedding
                ) for _, row in df.iterrows()]
                
                # Use execute_values for efficient batch insertion
                psycopg2.extras.execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table_name} (id, metadata, contents, embedding)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        metadata = EXCLUDED.metadata,
                        contents = EXCLUDED.contents,
                        embedding = EXCLUDED.embedding
                    """,
                    data,
                    template="(%s, %s, %s, %s)"
                )
                
                conn.commit()
                logging.info(f"Inserted {len(df)} records into {self.table_name}")
        finally:
            conn.close()

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[dict] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.

        Args:
            query_text: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary for equality-based metadata filtering.
            predicates: A dictionary for complex metadata filtering (not fully implemented).
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.
        """
        query_embedding = self.get_embedding(query_text)
        start_time = time.time()
        
        # Format the embedding as a string array for PostgreSQL - using proper vector format with []
        embedding_array = f"[{','.join(str(x) for x in query_embedding)}]"
        
        # Base query using proper vector format
        query = f"""
        SELECT id, metadata, contents, embedding, embedding <-> '{embedding_array}'::vector({self.embedding_dimensions}) AS distance
        FROM {self.table_name}
        """
        
        conditions = []
        params = []
        
        # Add metadata filter if provided
        if metadata_filter:
            # For simple equality filters
            for key, value in metadata_filter.items():
                conditions.append(f"metadata->>'{key}' = %s")
                params.append(str(value))
        
        # Add time range filter if provided
        if time_range:
            start_date, end_date = time_range
            conditions.append("created_at BETWEEN %s AND %s")
            params.extend([start_date, end_date])
        
        # Add WHERE clause if there are conditions
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Add ORDER BY and LIMIT
        query += " ORDER BY distance LIMIT %s"
        params.append(limit)
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()
                
            elapsed_time = time.time() - start_time
            logging.info(f"Vector search completed in {elapsed_time:.3f} seconds")
            
            if return_dataframe:
                return self._create_dataframe_from_results(results)
            else:
                return results
        finally:
            conn.close()

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
        """
        # Convert results to DataFrame
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )

        # Expand metadata column if it exists and has values
        if 'metadata' in df.columns and not df.empty:
            metadata_df = df["metadata"].apply(pd.Series)
            if not metadata_df.empty:
                df = pd.concat([df.drop(["metadata"], axis=1), metadata_df], axis=1)

        # Convert id to string for better readability
        if 'id' in df.columns:
            df["id"] = df["id"].astype(str)

        return df

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.

        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.
        """
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )
            
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if delete_all:
                    cur.execute(f"DELETE FROM {self.table_name}")
                    logging.info(f"Deleted all records from {self.table_name}")
                elif ids:
                    uuid_list = [uuid.UUID(id_str) for id_str in ids]
                    cur.execute(
                        f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",
                        (uuid_list,)
                    )
                    logging.info(f"Deleted {len(ids)} records from {self.table_name}")
                elif metadata_filter:
                    conditions = []
                    params = []
                    for key, value in metadata_filter.items():
                        conditions.append(f"metadata->>{key} = %s")
                        params.append(str(value))
                    
                    query = f"DELETE FROM {self.table_name} WHERE " + " AND ".join(conditions)
                    cur.execute(query, params)
                    logging.info(f"Deleted records matching metadata filter from {self.table_name}")
                
                conn.commit()
        finally:
            conn.close()
