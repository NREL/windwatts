import os
import json
import boto3
import tempfile


class ConfigManager:
    def __init__(
        self,
        secret_arn_env_var: str,
        local_config_path: str = None,
        region_name="us-west-2",
    ):
        """
        Initialize the ConfigManager with the ARN of the secret.

        :param secret_arn_env_var: The environment variable name that contains the ARN of the secret.
        :param local_config_path: The path to the local configuration file.
        """
        self.secret_arn = os.getenv(secret_arn_env_var)
        self.local_config_path = local_config_path
        self.client = boto3.client("secretsmanager", region_name=region_name)

    def get_config(self) -> str:
        """
        Retrieve the secret from AWS Secrets Manager, environment variables, or the local configuration file.
        :return: The path to the configuration file.
        """
        # Try to retrieve the secret from AWS Secrets Manager
        if self.secret_arn:
            try:
                response = self.client.get_secret_value(SecretId=self.secret_arn)
                secret = response["SecretString"]
                config_data = json.loads(secret)

                # Save the secret to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                temp_file.close()
                with open(temp_file.name, "w") as f:
                    json.dump(config_data, f)
                return temp_file.name
            except self.client.exceptions.ClientError as e:
                print(f"Unable to retrieve secret: {e}")

        # Try to retrieve config from environment variables
        env_config = self._get_config_from_env()
        if env_config:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            temp_file.close()
            with open(temp_file.name, "w") as f:
                json.dump(env_config, f)
            print("Config loaded from environment variables.")
            return temp_file.name

        # Fallback: return the path to the local configuration file
        if self.local_config_path and os.path.exists(self.local_config_path):
            print("Local configuration file found.")
            return self.local_config_path
        else:
            raise FileNotFoundError(
                "Local configuration file not found and unable to retrieve secret from AWS Secrets Manager or environment variables."
            )

    def _get_config_from_env(self):
        """
        Retrieve configuration from environment variables.
        Dynamically scan env vars with SOURCES_<SOURCE>_FIELD_NAME pattern for `sources` configuration.
        """
        # Top-level keys
        region_name = os.getenv("REGION_NAME")
        output_location = os.getenv("OUTPUT_LOCATION")
        output_bucket = os.getenv("OUTPUT_BUCKET")
        database = os.getenv("DATABASE")
        athena_workgroup = os.getenv("ATHENA_WORKGROUP")

        # Scan for all SOURCES_<SOURCE>_FIELD_NAME env vars
        sources = {}
        prefix = "SOURCES_"
        suffixes = ["_BUCKET_NAME", "_ATHENA_TABLE_NAME", "_ALT_ATHENA_TABLE_NAME"]
        env = os.environ
        source_fields = {}
        for key, value in env.items():
            if key.startswith(prefix):
                rest = key[len(prefix) :]
                for suffix in suffixes:
                    if rest.endswith(suffix):
                        source = rest[: -len(suffix)].lower()
                        field = suffix[1:].lower()  # e.g. 'bucket_name'
                        if source not in source_fields:
                            source_fields[source] = {}
                        source_fields[source][field] = value
        # Package the sources with required fields into `sources`
        for source, fields in source_fields.items():
            if "bucket_name" in fields and "athena_table_name" in fields:
                if "alt_athena_table_name" not in fields:
                    fields["alt_athena_table_name"] = ""
                sources[source] = {
                    "bucket_name": fields["bucket_name"],
                    "athena_table_name": fields["athena_table_name"],
                    "alt_athena_table_name": fields["alt_athena_table_name"],
                }
        # check if all keys have been set
        if (
            all(
                [
                    region_name,
                    output_location,
                    output_bucket,
                    database,
                    athena_workgroup,
                ]
            )
            and sources
        ):
            return {
                "region_name": region_name,
                "output_location": output_location,
                "output_bucket": output_bucket,
                "database": database,
                "athena_workgroup": athena_workgroup,
                "sources": sources,
            }
        else:
            return None
