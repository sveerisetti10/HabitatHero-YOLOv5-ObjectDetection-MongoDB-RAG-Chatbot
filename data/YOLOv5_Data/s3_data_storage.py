
# Import Libraries
import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def setup_and_download_from_s3(aws_access_key_id, aws_secret_access_key, bucket_name, s3_folder, local_dir=None):
    """
    Purpose: Download images from an S3 bucket to a local directory
    Input: aws_access_key_id- AWS access key ID
    Input: aws_secret_access_key- AWS secret access key
    Input: bucket_name- Name of the S3 bucket
    Input: s3_folder- Name of the S3 folder
    Input: local_dir- Local directory to save the images
    """

    # Here we create a boto3 client to interact with the S3 bucket
    try:
        s3_client = boto3.client(
            's3',
            # The access keys need to be retrieved from the author of the code
            # See the email at the bottom of the code to get the access keys
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    except PartialCredentialsError:
        print("Your AWS credentials are incomplete.")
        return
    except NoCredentialsError:
        print("Your AWS credentials were not found.")
        return

    def download_s3_folder(bucket_name, s3_folder, local_dir=None):
        """
        Purpose: Download images from an S3 bucket to a local directory
        Input: bucket_name- Name of the S3 bucket
        Input: s3_folder- Name of the S3 folder
        Input: local_dir- Local directory to save the images
        """
        if local_dir is None:
            local_dir = s3_folder
        # Here the paginator is used to iterate over every object in the S3 bucket and download it
        paginator = s3_client.get_paginator('list_objects_v2')
        try:
            # Iterate over every object on current page
            for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
                for obj in page.get('Contents', []):
                    # Captures the key associate with each image (file)
                    file_key = obj['Key']
                    if file_key.endswith('/'):
                        continue 
                    local_file_path = os.path.join(local_dir, os.path.basename(file_key))
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    # Here we use the download_file method to download the image to the local directory
                    s3_client.download_file(bucket_name, file_key, local_file_path)
                    print(f"Downloaded {file_key} to {local_file_path}")
        except NoCredentialsError:
            print("Invalid AWS credentials")
        except s3_client.exceptions.NoSuchBucket:
            print("The bucket does not exist or you have no access.")
        except Exception as e:
            print(e)

    # Finally, we call the function to download the images
    download_s3_folder(bucket_name, s3_folder, local_dir)

# Email me for the AWS credentials: sri.veerisetti@duke.edu
# aws_access_key_id = ''
# aws_secret_access_key = ''
# bucket_name = 'dlhabitathero'
# The different choices for the s3_folder are: images, labels
# s3_folder = 'images'
# local_dir = ''
# Sample Usage: setup_and_download_from_s3(aws_access_key_id, aws_secret_access_key, bucket_name, s3_folder, local_dir)