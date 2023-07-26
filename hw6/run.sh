export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
export S3_ENDPOINT_URL="http://localhost:4566"

docker-compose up localstack -d

sleep 5

aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration

pipenv run python3 integration_test.py

pipenv run python3 batch.py 2022 01

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

docker-compose down