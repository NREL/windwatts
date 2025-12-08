# API Endpoint for DW-TAP Project

## About

This is an endpoint repository -- it contains the code that runs a Flask app serving the processed WTK data. 
The processing includes height selection, time interval selection, spatial interpolation, vertical interpolation, and wind rose calculations.    

When UI/API is running, you can:

1. See WindWatts-beta by navigating your browswer to:
```
<server's hostname>:<server's port>
```
2. Run example API query, by going to a link like this (change the values if necessary)
```
<server's hostname>:<server's port>/1224?lat=39.76004&lon=-105.14058
```
3. Check server info at:
```
<server's hostname>:<server's port>/status
```

For a local deployment, these links would be:
```
http://localhost:8080
http://localhost:8080/1224?lat=39.76004&lon=-105.14058
http://localhost:8080/status
```

## How To Use

### Local Deployment of New Windwatts App using Docker (Docker required)

#### Prerequisites
1. **Install and Start Docker**: Ensure Docker is installed and running on your machine.
2. **AWS Credentials**: Obtain AWS credentials with access to the Windwatts Data package.
3. **Google Maps API Key**: Create an API key and map ID by following the [Google Maps API Documentation](https://developers.google.com/maps/documentation/javascript).

#### Steps to Deploy Locally
1. **Clone the repository**: The new Windwatts app is hosted on the development branch
   ```shell
   git clone https://github.com/NREL/dw-tap-api.git
   cd dw-tap-api/
   git checkout development # Switch to the development branch where the new app resides
   ```

2. **Configure environment files**:
   - **Root directory (`dw-tap-api/`)**:
     - Create or update a `.env` file with the following variables:
       ```plaintext
       WINDWATTS_DATA_URL=https://windwatts-era5.s3.us-west-2.amazonaws.com/
       AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
       AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
       AWS_SESSION_TOKEN="YOUR_AWS_SESSION_TOKEN"
       ```
       - `WINDWATTS_DATA_URL`: URL for the Windwatts Data package.
       - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`: AWS credentials for accessing resources.

   - **Frontend directory (windwatts-ui)**:
     - Create or update a `.env.development` file:
       ```plaintext
       VITE_API_BASE_URL=http://windwatts-proxy:80
       VITE_MAP_API_KEY=YOUR_MAP_API_KEY
       VITE_MAP_ID=YOUR_MAP_ID
       ```
       - `VITE_API_BASE_URL`: Base URL for the Windwatts API.
       - `VITE_MAP_API_KEY`, `VITE_MAP_ID`: Google Maps API key and map ID.

   - **Backend directory (windwatts-api)**:
     - Create or update `windwatts_data_config.json` under `config/`:
       ```json
        {
          "region_name": "us-west-2",
          "output_location": "S3_BUCKET_URI_FOR_ATHENA_RESULTS",
          "output_bucket": "NAME_OF_S3_BUCKET_FOR_ATHENA_RESULTS",
          "database": "NAME_OF_THE_GLUE_DATABASE",
          "athena_workgroup": "NAME_OF_THE_ATHENA_WORKGROUP",
          "sources": {
            "wtk": {
              "bucket_name": "NAME_OF_THE_WTK_S3_BUCKET",
              "athena_table_name": "NAME_OF_THE_ATHENA_TABLE_FOR_WTK",
              "alt_athena_table_name": ""
            },
            "era5": {
              "bucket_name": "NAME_OF_THE_ERA5_S3_BUCKET",
              "athena_table_name": "NAME_OF_THE_ATHENA_TABLE_FOR_ERA5",
              "alt_athena_table_name": ""
            }
          }
        }
       ```
      - Parameter Description
          -  **region_name** : AWS region where your S3 buckets and Athena services are hosted.
          - **output_location** : S3 bucket URI where Athena will store query results (e.g., s3://bucket-name/).
          - **output_bucket**	: Name of the bucket used above.
          - **database** : AWS Glue database name in which athena tables are created.
          - **athena_workgroup** : Name of the Athena workgroup to use for querying.
          - **sources**.*wtk*	: Configuration specific to WTK-Led Climate dataset.
          - **sources**.*era5* : Configuration specific to ERA5 dataset.
          - **athena_table_name**	: Primary Athena table name for the dataset. This table is used for location specific queries.
          - **alt_athena_table_name** : Optional alternate Athena table name for non-location specific queries.


3. **Deploy the app**:
   - Start Docker containers:
     ```shell
     docker compose up --build
     ```
   - If needed, clean up previous containers and volumes:
     ```shell
     docker compose down --volumes --remove-orphans
     ```

4. **Access the app**:
   - Open your browser and navigate to: `http://localhost:5173/`.

### Deploy as a Container (requires Docker on the host)

Build:
```shell
docker build -t tap-api:latest .
```

Run:
```shell
docker run -p 8080:80 -it tap-api:latest python proto.py --production
```

For troubleshooting inside the container, run (and you will have the prompt change and try the following commands inside the container):
```shell
docker run -p 8080:80 -it tap-api:latest /bin/bash
```

Inside the container, flask app will run on port `80`.  On the host, you can use any available port, e.g, `8080`, like shown above.  

For a simple test (showing info about the endpoint), navigate to the following URL in your browser (running on the host):
```
http://localhost:8080
``` 

For a more comprehensive test (with HSDS connection and spatial + vertical interpolations), navigate to this URL:
```
http://localhost:8080/v1/timeseries/windspeed?height=67.00m&lat=40.7888&lon=-74.0059&start_date=20110302&stop_date=20110303&vertical_interpolation=nearest&spatial_interpolation=idw
```
This should produce a json output with `timestamp` and `windspeed` values.

For other examples of working queries, refer to the file: `dw-tap-api.postman_collection.json` (look for `raw` attributes). This file can be used by the Postman app (e.g., installed locally, on a laptop), which will cycle through all documented queries and show their status.

### Build & Run Natively (without a container)

```shell
conda env create
conda activate dw-tap-api
python api.py --development
```

### Running Modes

Notice the `--development` flag in the command above -- it makes the endpoint run on port `8080`; for short, you can run: `python api.py -d`. 

*Development* is the default mode (run if no flag is specified). In contrast, you can run the endpoint in the *Production* mode using: `python api.py --production` or `python api.py -p` -- this is what is used in the container deployment described above (to see the details, check the end of `Dockerfile`).

To see how these production and development modes are configured, refer to `config.json` and see what `host` and `port` values are specified.  

## Documentation

Interactive HTML page with API documentation is produced using apiDoc. It can be recreated using (requires installing apiDoc locally):
```
apidoc -i . -o docs/ -t apidoc-template
```
The output can be seen by opening `docs/index.html` in a browser. The flask app is configured to serve this documentation page (and related files) at the "/api" route.

For installing apiDoc on osx, run:
```
brew install apidoc
```

### More about the Project

To read a concise summary of the DW-TAP project, please refer to: https://www.energy.gov/sites/prod/files/2020/02/f72/tap-fact-sheet_0.pdf

## Credit

Code in this repository was developed by Dmitry Duplyakin (dmitry.duplyakin@nrel.gov), Caleb Phillips (caleb.phillips@nrel.gov), and Sagi Zisman (sagi.zisman@nrel.gov) to demonstrate the techniques used in distributed wind resource assessment at the National Laboratory of the Rockies in Golden, Colorado, USA.

## License

Refer to the file called: `LICENSE`.
