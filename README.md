# azure-ml-cicd-pipeline

This project implements an automated Azure Machine Learning pipeline that predicts the following week’s sales for each Walmart store using the current week’s engineered features. As new sales figures become available each week, the pipeline can process and validate the latest data, update the feature store, train and evaluate new candidate models, and deploy an updated model when it performs better than the existing production model. This creates a repeatable end-to-end workflow for keeping weekly sales forecasts accurate and up to date.


## Azure Environments

The project separates development and production resources across two Azure resource groups.

- Test environment – Used to process new data, engineer features, train candidate models, run validation checks, and test deployments without affecting the live solution.
- Production environment – Contains the approved data assets, models which were promoted from the test environment as well as online endpoints, and batch endpoints used to generate production sales forecasts.

This separation prevents untested changes from affecting production and allows models and data assets to be promoted between environments in a controlled manner.

## Azure Environment Setup

The project uses a Bash setup script to provision separate test and production environments in Azure. Run the script once for each environment, using a unique resource prefix to keep their resource groups and resources isolated. Using separate resource groups also provides a clear security boundary between the environments. Microsoft Entra security groups can be assigned Azure role-based access control (RBAC) roles at the appropriate resource-group scope, allowing access to be managed independently. For example, developers can be granted permission to create and manage resources in the test resource group while receiving read-only or no access to the production resource group. Production deployment permissions can then be restricted to an approved SRE group.

### Prerequisites

Before running the setup script, ensure that:

The Azure CLI is installed and authenticated using az login.
The Azure Machine Learning CLI extension is installed.
jq, envsubst, and openssl are installed.
Your Azure account has permission to create resources, service principals, and role assignments.
The project is opened at its root directory so the script can access the required data files and Azure ML templates.

### Usage
./setup_scripts/env_setup.sh \
  --prefix NAME \
  --owner NAME \
  --env-type test|prod \
  [--storage-type blobstore|datalake]

| Argument | Description |
| --- | --- |
| `--prefix` | Unique prefix used to name the Azure resources. |
| `--owner` | Name added to the Azure resources as an ownership tag. |
| `--env-type` | Environment to create: `test` or `prod`. |
| `--storage-type` | Source storage used by the test environment: `blobstore` or `datalake`. This is not required for production. |

### Resource Creation

| Azure resource or configuration | Test environment | Production environment |
| --- | :---: | :---: |
| Azure resource group | ✅ | ✅ |
| Azure Container Registry | ✅ | ✅ |
| Azure Machine Learning workspace | ✅ | ✅ |
| Training environment | ✅ | — |
| Endpoint environment | ✅ | ✅ |
| Azure Machine Learning compute cluster | ✅ | ✅ |
| Source data asset | ✅ | — |
| Test data asset | ✅ | — |
| Azure Data Lake Storage Gen2 account and datastore | Optional (datalake only) | — |
| Azure Machine Learning feature store | ✅ | — |
| Feature-store entity and feature set | ✅ | — |
| Initial feature materialization and backfill | ✅ | — |
| Engineered-feature data asset | ✅ | — |
| Real-time endpoint | ✅ | ✅ |
| Batch endpoint | ✅ | ✅ |
| GitHub Actions service principal | ✅ | ✅ |
| Azure role assignments | ✅ | ✅ |

### Outputs
The script outputs the required secrets and variables.

| Script Output Name | Type | Test GitHub Name | Production GitHub Name |
| --- | --- | --- | --- |
| `TEST/PROD_AZURE_SECRET` | GitHub secret | `TEST_AZURE_SECRET` | `PROD_AZURE_SECRET` |
| `TEST/PROD_PREFIX` | GitHub variable | `TEST_PREFIX` | `PROD_PREFIX` |
| `TEST_STORAGE_TYPE` | GitHub variable | `TEST_STORAGE_TYPE` | N/A |
| `TEST_TRAINING_ENV_VERSION` | GitHub variable | `TEST_TRAINING_ENV_VERSION` | N/A |
| `TEST/PROD_ENDPOINT_ENV_VERSION` | GitHub variable | `TEST_ENDPOINT_ENV_VERSION` | `PROD_ENDPOINT_ENV_VERSION` |



## CI/CD pipelines

The project uses three GitHub Actions workflows to manage data processing, model training, test deployment, and production promotion.

All three workflows are started manually using `workflow_dispatch`. Deployment-related jobs are restricted to runs from the `main` branch.

| Pipeline | Workflow file | Environment | Purpose |
| --- | --- | --- | --- |
| Test pipeline | `train-and-deploy-to-staging.yml` | Test | Tests the code, trains candidate models, performs champion–challenger evaluation, and deploys a successful model to the endpoints. |
| Data pipeline | `data-pipeline.yml` | Test | Versions new source data, materializes features, registers engineered data, and refreshes the endpoints. |
| Production promotion | `promote-to-prod.yml` | Production | Promotes approved model and data versions from the test workspace and updates the production endpoints. |


### Test and Deployment Pipeline

The test pipeline validates the project, trains candidate models, compares the best candidate with the current champion, and deploys the challenger when it passes the required checks.

1. **Run code-quality checks**
   - Runs Pylint against the source code.
   - Enforces the configured minimum Pylint score.
   - Uploads the Pylint report as a workflow artifact.

2. **Run automated tests**
   - Runs the unit and integration test suites with Pytest.
   - Measures test coverage against the configured threshold.
   - Generates JUnit, coverage, and Pytest output reports.

3. **Retrieve the current Azure ML state**
   - Resolves the existing real-time and batch endpoint names.
   - Retrieves the latest materialized feature date.

4. **Train candidate models**
   - Runs multiple Azure ML training jobs using different XGBoost hyperparameters.
   - Evaluates each candidate against the configured performance thresholds.
   - Saves the metrics and training details for each candidate.

5. **Select and train the challenger**
   - Selects the strongest candidate that passed validation.
   - Retrains the selected configuration using all available training data.
   - Produces the challenger model and forecast outputs.

6. **Perform champion–challenger testing**
   - Invokes the current champion model through the real-time endpoint.
   - Compares champion and challenger predictions against known sales figures.
   - Uses WAPE, RMSE, MAPE, and MAE to determine the winner.

7. **Register the challenger**
   - Registers the challenger in Azure Machine Learning.
   - Tags it as `accepted` or `rejected` based on the comparison result.
   - Records the source training job and assigned model version.

8. **Deploy a successful challenger**
   - Creates a new real-time deployment when the challenger wins.
   - Runs a smoke test against the new deployment.
   - Runs a Locust load test and validates latency and failure-rate thresholds.
   - Routes real-time endpoint traffic to the new deployment only after the tests pass.

9. **Create the batch deployment**
   - Creates a batch deployment using the newly registered model.
   - Invokes the deployment using the current test-data asset.
   - Downloads and validates the generated predictions.
   - Sets the new deployment as the endpoint default.

10. **Clean up deployments**
    - Removes real-time deployments receiving no traffic.
    - Removes old batch deployments while preserving the default deployment.

The workflow retains test, linting, model-selection, comparison, load-testing, and batch-prediction reports as GitHub Actions artifacts.



### Data Pipeline - Datalake storeage only
The data pipeline processes newly released weekly Walmart sales data. At present, it only supports test environments configured with Azure Data Lake Storage Gen2. Test environments configured to use the Azure Machine Learning workspace's default Blob Storage are not currently supported by this workflow.

The pipeline checks the `TEST_STORAGE_TYPE` GitHub variable before processing begins and only runs when its value is set to `datalake`.

A GitHub Actions concurrency group prevents multiple instances of the data pipeline from updating the same Data Lake simultaneously. If another run is already in progress, the new run waits for it to finish instead of cancelling it.


1. **Validate storage configuration**
   - Confirms that the test environment uses `datalake` storage.
   - Stops the workflow when the environment is not configured for Azure Data Lake Storage.

2. **Determine the next data version**
   - Checks the existing source, engineered-feature, and test-data versions.
   - Calculates a shared version number for the new assets.
   - Ensures related data assets remain aligned under the same version.

3. **Update the Data Lake**
   - Validates the new Walmart sales CSV file.
   - Uploads an immutable versioned copy to Azure Data Lake Storage.
   - Updates the current source-data path used by the feature store.

4. **Determine the materialization window**
   - Reads the latest date from the new source data.
   - Identifies the most recent feature-store materialization date.
   - Calculates the start and end dates that require materialization.

5. **Materialize updated features**
   - Submits an Azure ML feature-store backfill for the required date range.
   - Captures the submitted materialization job IDs.
   - Waits for the relevant jobs to complete and fails if a job is unsuccessful.

6. **Register engineered data**
   - Exports the latest materialized features from the offline feature store.
   - Validates the exported rows, stores, and date range.
   - Registers the engineered features and batch test data as versioned Azure ML data assets.

7. **Refresh the endpoints**
   - Creates a new real-time deployment using the latest engineered-feature data.
   - Creates and invokes a new batch deployment.
   - Validates the generated predictions before updating the active deployments.

8. **Clean up deployments**
   - Deletes real-time deployments receiving no traffic.
   - Deletes old batch deployments while retaining the active default deployment.

The workflow writes a GitHub Actions summary containing the new data version, materialization details, registered assets, row counts, store counts, and exported date range.


### Production Promotion Pipeline

The production pipeline promotes explicitly selected model and data versions from the test Azure ML workspace to the production workspace.

The versions to promote are supplied through the following GitHub variables:

- `PROMOTED_MODEL_VERSION`
- `PROMOTED_DATA_VERSION`

1. **Check production versions**
   - Checks whether the selected model, engineered-feature data, and test-data versions already exist in production.
   - Avoids registering duplicate asset versions.

2. **Download assets from test**
   - Downloads the selected model from the test Azure ML workspace.
   - Downloads and validates the selected engineered-feature data.
   - Downloads and validates the corresponding batch test data.
   - Transfers the assets between jobs using GitHub Actions artifacts.

3. **Register assets in production**
   - Registers the model using the same version selected in the test workspace.
   - Registers the engineered-feature and test-data assets using the promoted data version.
   - Verifies each registered asset after creation.

4. **Update the production real-time endpoint**
   - Creates a new real-time deployment using the promoted model and feature-data versions.
   - Runs deployment and scoring checks.
   - Routes production traffic to the new deployment after validation succeeds.

5. **Update the production batch endpoint**
   - Creates a new batch deployment using the promoted model and data.
   - Invokes the deployment against the promoted test-data asset.
   - Downloads and validates the batch predictions.
   - Sets the new deployment as the default batch deployment.

6. **Clean up production deployments**
   - Deletes real-time deployments that no longer receive traffic.
   - Deletes old batch deployments while retaining the active default deployment.

The production workflow preserves matching version numbers across the test and production workspaces, providing traceability between the validated assets and the resources deployed to production.



## Future Improvements

The following improvements could make greater use of Azure Machine Learning’s managed feature-store capabilities and improve production monitoring.

### Feature-store-based batch inference

Replace the current file-based batch scoring process with an Azure Machine Learning batch inference pipeline that uses the built-in feature retrieval component.

The pipeline would retrieve the required materialized features from the feature store’s offline store, join them to the batch observation data, and pass the resulting dataset to the model for inference. This would remove the need to export the engineered features to a separate CSV file in Blob Storage before running batch predictions.

A feature retrieval specification could be packaged with the registered model to ensure that batch inference retrieves the same feature definitions used during training.

### Online feature retrieval for real-time inference

Configure an online store for the managed feature store and enable online materialization for the required feature set.

The real-time scoring script could then use the Azure Machine Learning feature-store SDK to retrieve the latest feature values from the online store using the store identifier supplied in each request. This would replace the current approach of loading engineered features from a CSV file stored in Blob Storage.

Using an online store would provide lower-latency feature retrieval and help maintain consistency between the features used for training and real-time inference.

### Feature-store-based model training

Update the model-training pipeline to retrieve materialized features directly from the feature store instead of independently recreating the feature-engineering logic.

The training pipeline could use a feature retrieval specification together with the built-in Azure Machine Learning feature retrieval component. Reusing the registered feature definitions would reduce duplicated feature-engineering code, improve feature lineage, and help prevent training-serving skew.

### Production model monitoring

Enable Azure Machine Learning model monitoring for the production deployment.

Monitoring could be configured to detect:

- Data drift in model input features
- Prediction drift
- Data-quality issues
- Feature-attribution drift
- Changes in model performance when actual sales figures become available

Monitoring jobs could run on a weekly schedule, matching the release frequency of the sales data. Appropriate thresholds and alerts could provide early warning when the production data or model behaviour changes significantly.