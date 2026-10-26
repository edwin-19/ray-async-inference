import os

import boto3
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
access_key = os.environ["MINIO_ROOT_USER"]
secret_key = os.environ["MINIO_ROOT_PASSWORD"]
bucket_name = os.getenv("MINIO_BUCKET", "speech-data")
secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

client = boto3.client(
    "s3",
    endpoint_url=f"{'https' if secure else 'http'}://{endpoint}",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)


def list_buckets():
    response = client.list_buckets()
    buckets = response.get("Buckets", [])

    print("\n=== Buckets ===")
    for bucket in buckets:
        print(bucket["Name"])

    if not buckets:
        print("No buckets found.")


def create_bucket(name):
    if client.head_bucket(Bucket=name):
        print(f"Bucket already exists: {name}")


def ensure_bucket(name):
    existing = [
        bucket["Name"]
        for bucket in client.list_buckets().get("Buckets", [])
    ]

    if name in existing:
        print(f"Bucket already exists: {name}")
    else:
        client.create_bucket(Bucket=name)
        print(f"Created bucket: {name}")


def list_objects(name):
    print(f"\n=== Objects in {name} ===")

    paginator = client.get_paginator("list_objects_v2")
    found = False

    for page in paginator.paginate(Bucket=name):
        for obj in page.get("Contents", []):
            found = True
            print(f"{obj['Key']} ({obj['Size']} bytes)")

    if not found:
        print("Bucket is empty.")


if __name__ == "__main__":
    list_buckets()
    ensure_bucket(bucket_name)
    list_objects(bucket_name)
