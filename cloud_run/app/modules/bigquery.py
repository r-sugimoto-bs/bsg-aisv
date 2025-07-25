from google.cloud import bigquery
import os

class BigQuery:
    def __init__(self):
        self.client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'), location="us-central1")

    def fetch_merchant_info(self) -> bigquery.table.RowIterator:
        query = f"""
        SELECT merchant_name, branch_name
        FROM `{os.getenv('TOTAL_DATA_TABLE')}`
        """
        query_job = self.client.query(query)
        return query_job.result()
