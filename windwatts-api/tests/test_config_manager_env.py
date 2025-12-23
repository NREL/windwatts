import os
from app.config_manager import ConfigManager

# Set required top-level environment variables
os.environ["REGION_NAME"] = "us-west-2"
os.environ["OUTPUT_LOCATION"] = "s3://test-bucket/"
os.environ["OUTPUT_BUCKET"] = "test-bucket"
os.environ["DATABASE"] = "test_database"
os.environ["ATHENA_WORKGROUP"] = "test_workgroup"

# Set environment variables for two sources: wtk and era5
os.environ["SOURCES_WTK_BUCKET_NAME"] = "wtk-bucket"
os.environ["SOURCES_WTK_ATHENA_TABLE_NAME"] = "wtk_table"
os.environ["SOURCES_WTK_ALT_ATHENA_TABLE_NAME"] = "wtk_alt_table"
os.environ["SOURCES_ERA5_BUCKET_NAME"] = "era5-bucket"
os.environ["SOURCES_ERA5_ATHENA_TABLE_NAME"] = "era5_table"
# alt_athena_table_name is optional

# Instantiate ConfigManager (no secret ARN, no local file)
cm = ConfigManager(secret_arn_env_var="DUMMY_SECRET_ARN", local_config_path=None)

# Get config path
config_path = cm.get_config()
print(f"\nGenerated config file path: {config_path}\n")

# Print the contents for verification
with open(config_path) as f:
    print(f.read())
