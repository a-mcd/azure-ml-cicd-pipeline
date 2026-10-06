#!/usr/bin/env bash

# Setup workflow in azure ml for the project. This includes:
# - Resource group
# - Container registry
# - AML Workspace
# - AML Environments (training and endpoint)
# - AML Compute
# - Uploading data to the workspace
# - Create permissions 
# - Create feature store
# - Create realtime endpoint
# - Create bacth endpoint


set -euo pipefail

LOC="norwayeast"

mkdir -p tmp
WORKSPACE_YAML="tmp/workspace.yaml"
TRAINING_ENVIRONMENT_YAML="tmp/training-environment.yaml"
ENDPOINT_ENVIRONMENT_YAML="tmp/endpoint-environment.yaml"
ENGINEERED_DATA_PATH="./data/walmart_materialized_features_all_weeks.csv"
RESOURCE_PREFIX=""
OWNER=""
WALMART_SALES_DATASET_VERSION=""
WALMART_SALES_DATASET_PATH=""
MATERIALIZED_WALMART_SALES_DATASET_VERSION=""
MATERIALIZED_WALMART_SALES_DATASET_PATH=""
TEST_DATA_VERSION=""
TEST_DATA_PATH=""
LATEST_WEEK_DATA_VERSION=""
LATEST_WEEK_DATA_PATH=""
AZURE_SECRET=""
AZURE_ACCOUNT_TYPE=""
AZURE_ACCOUNT_USER_NAME=""
AZURE_SUBSCRIPTION_ID=""
AZURE_SUBSCRIPTION_NAME=""
WORKSPACE_ID=""
WORKSPACE_STORAGE_ID=""
FEATURE_STORE_ID=""
FEATURE_STORE_STORAGE_ID=""
OFFLINE_STORE_SCOPE=""
ENDPOINT_ENV_VERSION=""
TRAINING_ENV_VERSION=""
STORAGE_TYPE=""
SOURCE_FILE="data/Walmart_Sales_26-10-2012.csv"
STORAGE_ACCOUNT=""

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 --prefix NAME --owner NAME --env-type test|prod [--storage-type blobstore|datalake]"
  exit 1
fi

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) RESOURCE_PREFIX="$2"; shift 2 ;;
    --owner) OWNER="$2"; shift 2 ;;
    --env-type) ENV_TYPE="$2"; shift 2 ;;
    --storage-type) STORAGE_TYPE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

[[ -n "$RESOURCE_PREFIX" ]] || { echo "ERROR: --prefix is required." >&2; exit 1; }
[[ -n "$OWNER" ]] || { echo "ERROR: --owner is required." >&2; exit 1; }
[[ "${ENV_TYPE:-}" =~ ^(test|prod)$ ]] || { echo "ERROR: --env-type must be 'test' or 'prod'." >&2; exit 1; }
[[ "$STORAGE_TYPE" =~ ^(blobstore|datalake|na)$ ]] || { echo "ERROR: --storage-type must be 'blobstore' or 'datalake' or na for prod only." >&2; exit 1; }

TAGS_ENV="env=dev"
TAGS_OWNER="owner=$OWNER"
TAGS_PROJECT="project=$RESOURCE_PREFIX"
CR_PREFIX="${RESOURCE_PREFIX}cr"
WORKSPACE_NAME="${RESOURCE_PREFIX}_workspace"
RG="${RESOURCE_PREFIX}-rg"
TRAINING_ENVIRONMENT_NAME="${RESOURCE_PREFIX}-training-env"
ENDPOINT_ENVIRONMENT_NAME="${RESOURCE_PREFIX}-endpoint-env"
FEATURE_STORE_WS="${RESOURCE_PREFIX}_fs_ws"
FEATURE_STORE="${RESOURCE_PREFIX}_fs"
DATA_ASSET_NAME="walmart_sales_dataset"
FEATURE_SET_VERSION="1"
FEATURE_SET_NAME="walmart_sales_features"
CLUSTER_NAME="${RESOURCE_PREFIX}-compute"


# Must be globally unique, lowercase, and contain no dashes.
ADLS_ACCOUNT="$(
    printf '%s' "${RESOURCE_PREFIX}walmartlake" |
        tr '[:upper:]' '[:lower:]' |
        tr -cd 'a-z0-9'
)"
if [[ ! "$ADLS_ACCOUNT" =~ ^[a-z0-9]{3,24}$ ]]; then
    echo "ERROR: Invalid ADLS account name: $ADLS_ACCOUNT" >&2
    exit 1
fi

ADLS_VERSION="1"
ADLS_FILESYSTEM="walmart-source"
ADLS_FILE_PATH_CURRENT="walmart-sales/current/Walmart_Sales.csv"
ADLS_FILE_PATH_VERSIONS="walmart-sales/versions/${ADLS_VERSION}/Walmart_Sales.csv"
SOURCE_STORAGE_ACCOUNT="$ADLS_ACCOUNT"

# ========= Helpers =========
fail() { echo "ERROR: $*" >&2; exit 1; }
info() { echo ">>> $*"; }

require_command() {
    local command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 ||
        fail "'$command_name' is not installed. Please install it before running this script."
}

render_template() {
    local template_path="$1"
    local output_path="$2"
    local substitutions="$3"

    mkdir -p "$(dirname "$output_path")"
    [[ -f "$template_path" ]] || fail "Template not found: $template_path"

    envsubst "$substitutions" < "$template_path" > "$output_path"
    [[ -s "$output_path" ]] || fail "Rendered file is empty: $output_path"
}


check_prereq(){
    local azure_account_json

    # ========= Pre-checks =========
    info "Checking prerequisites..."

    require_command jq
    require_command az
    require_command envsubst
    require_command openssl

    # Check that the user is logged into Azure and can obtain an access token
    if ! az account get-access-token --output none >/dev/null 2>&1; then
        fail "You are not logged into Azure, or your Azure session has expired. Run 'az login' and try again."
    fi

    azure_account_json="$(az account show --output json)"
    AZURE_ACCOUNT_USER_NAME="$(jq -r '.user.name' <<<"$azure_account_json")"
    AZURE_ACCOUNT_TYPE="$(jq -r '.user.type' <<<"$azure_account_json")"
    AZURE_SUBSCRIPTION_NAME="$(jq -r '.name' <<<"$azure_account_json")"
    AZURE_SUBSCRIPTION_ID="$(jq -r '.id' <<<"$azure_account_json")"
    WORKSPACE_ID="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${RG}/providers/Microsoft.MachineLearningServices/workspaces/${WORKSPACE_NAME}"
    FEATURE_STORE_ID="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${RG}/providers/Microsoft.MachineLearningServices/workspaces/${FEATURE_STORE_WS}"

    info "Azure login confirmed."
    info "Signed in as: $AZURE_ACCOUNT_USER_NAME"
    info "Active subscription: $AZURE_SUBSCRIPTION_NAME ($AZURE_SUBSCRIPTION_ID)"
    info "All prerequisites are installed."
}

create_resource_group(){
    local rg_json
    local provisioning_state

    # ========= Create Resource Group =========
    info ">>> Creating or updating resource group: $RG in $LOC"

    rg_json="$(az group create -n "$RG" -l "$LOC" --tags "$TAGS_ENV" "$TAGS_OWNER" "$TAGS_PROJECT" --output json)"

    provisioning_state="$(jq -r '.properties.provisioningState' <<<"$rg_json")"

    if [[ "$provisioning_state" == "Succeeded" ]]; then
        info "Resource group '$RG' created successfully in '$LOC'"
    else
        fail "Failed to create resource group '$RG' (provisioningState: $provisioning_state)"
    fi
}

create_container_registry(){
    local create_json
    local existing_registry
    local provisioning_state
    local random_suffix

    # ========= Create Container Registry from YAML =========

    existing_registry="$(az acr list -g "$RG" --query "[?starts_with(name, '${CR_PREFIX}')].name | [0]" -o tsv)"

    if [[ -n "$existing_registry" ]]; then
        info "Container registry '$existing_registry' which starts with '$CR_PREFIX' exists in '$RG'."
        CR_NAME="$existing_registry"
        export CR_NAME
    else

        random_suffix="$(openssl rand -base64 12 | tr -dc 'a-z0-9' | head -c 12)"
        CR_NAME="$CR_PREFIX$random_suffix"

        # Add in check if container with prefix already exists in the workspace
        info "Creating/ensuring Azure Container Registry $CR_NAME '..."

        create_json="$(az acr create --name "$CR_NAME" --resource-group "$RG" --location "$LOC" --sku Standard --admin-enabled false)"

        if [[ "$create_json" == *"error"* ]]; then
            fail "$create_json"
        fi

        provisioning_state="$(jq -r '.provisioningState' <<<"$create_json")"

        if [[ "$provisioning_state" == "Succeeded" ]]; then
            info "Container registry '$CR_NAME' created successfully in '$LOC'"
        else
            fail "Failed to create container registry '$CR_NAME' (provisioningState: $provisioning_state)"
        fi

        export CR_NAME
    fi

}

create_workspace(){
    local CONTAINER_REGISTRY_ID
    local ENV="$TAGS_ENV"
    local LOCATION="$LOC"
    local OWNER="$TAGS_OWNER"
    local attempt
    local workspace_status

    # ========= Create Workspace from YAML =========
    info "Creating/ensuring Azure ML workspace from '$WORKSPACE_YAML'..."

    export WORKSPACE_NAME LOCATION ENV OWNER

    CONTAINER_REGISTRY_ID="$(az acr show -g "$RG" -n "$CR_NAME" --query id -o tsv)"
    export CONTAINER_REGISTRY_ID

    envsubst < azureml/workspace.template.yaml > "$WORKSPACE_YAML"

    # Validate YAML exists
    [[ -f "$WORKSPACE_YAML" ]] || fail "Workspace YAML not found at: $WORKSPACE_YAML"

    az ml workspace create \
        --file "$WORKSPACE_YAML" \
        --resource-group "$RG" \
        --only-show-errors \
        --output none
    info "Workspace create command completed."

    workspace_status="$(az resource show --resource-group "$RG" --resource-type "Microsoft.MachineLearningServices/workspaces" --name "$WORKSPACE_NAME" --query "properties.provisioningState" -o tsv)"

    if [[ "$workspace_status" != "Succeeded" ]]; then
        info "Waiting for workspace to finish provisioning..."
        for ((attempt = 1; attempt <= 30; attempt++)); do
            sleep 10
            workspace_status="$(az resource show --resource-group "$RG" --resource-type "Microsoft.MachineLearningServices/workspaces" --name "$WORKSPACE_NAME" --query "properties.provisioningState" -o tsv)"
            info "Attempt $attempt: $workspace_status"
            [[ "$workspace_status" == "Succeeded" ]] && break
        done
    fi

    if [[ "$workspace_status" == "Succeeded" ]]; then
        info "Workspace '$WORKSPACE_NAME' is ready."
    else
        fail "Workspace '$WORKSPACE_NAME' provisioningState: $workspace_status"
    fi

    WORKSPACE_STORAGE_ID="$(
        az ml workspace show \
            --name "$WORKSPACE_NAME" \
            --resource-group "$RG" \
            --query storage_account \
            --output tsv
    )"
    [[ -n "$WORKSPACE_STORAGE_ID" && "$WORKSPACE_STORAGE_ID" != "null" ]] ||
        fail "Could not resolve the workspace storage account resource ID."
}

create_compute(){
    local COMPUTE_INSTANCES="$1"
    local COMPUTE_SIZE="Standard_DS3_v2"
    local COMPUTE_YAML="tmp/job-compute.yaml"
    local create_json
    local provisioning_state

    # ========= Create Compute from YAML =========

    info "Creating/ensuring Azure ML compute from '$COMPUTE_YAML'..."
    export CLUSTER_NAME
    export COMPUTE_SIZE
    export COMPUTE_INSTANCES

    envsubst '${CLUSTER_NAME} ${COMPUTE_SIZE} ${COMPUTE_INSTANCES}' < azureml/compute.template.yaml > "$COMPUTE_YAML"

    # Validate YAML exists
    [[ -f "$COMPUTE_YAML" ]] || fail "Compute YAML not found at: $COMPUTE_YAML"

    create_json="$(az ml compute create --resource-group "$RG" --workspace-name "$WORKSPACE_NAME" --file "$COMPUTE_YAML")"

    provisioning_state="$(jq -r '.provisioning_state' <<<"$create_json")"

    if [[ "$provisioning_state" == "Succeeded" ]]; then
        info "Compute '$CLUSTER_NAME' created successfully in resource group '$RG', location '$LOC'"
    else
        fail "Failed to create compute '$RG' (provisioningState: $provisioning_state)"
    fi

}

create_training_environment(){
  local create_json

  # ========= Create training environment from YAML =========
  info "Creating training environment"


  export TRAINING_ENVIRONMENT_NAME
  envsubst '${TRAINING_ENVIRONMENT_NAME}' < azureml/docker.environment.template.yaml > $TRAINING_ENVIRONMENT_YAML

  # Validate YAML exists
  [[ -f "$TRAINING_ENVIRONMENT_YAML" ]] || fail "Environment YAML not found at: $TRAINING_ENVIRONMENT_YAML"

  create_json="$(az ml environment create --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --file "$TRAINING_ENVIRONMENT_YAML" \
    -o json)"

  TRAINING_ENV_VERSION="$(jq -r '.version' <<<"$create_json")"
  export TRAINING_ENV_VERSION

  info "Training environment version: $TRAINING_ENV_VERSION"

}

create_endpoint_environment(){
  local create_json

  # ========= Create endpoint environment from YAML =========

  info "Creating endpoint environment"

  export ENDPOINT_ENVIRONMENT_NAME
  envsubst '${ENDPOINT_ENVIRONMENT_NAME}' < azureml/conda.environment.template.yaml > $ENDPOINT_ENVIRONMENT_YAML

  create_json="$(az ml environment create \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --file "$ENDPOINT_ENVIRONMENT_YAML" \
    -o json)"

  ENDPOINT_ENV_VERSION="$(jq -r '.version' <<<"$create_json")"
  export ENDPOINT_ENV_VERSION

  info "Endpoint environment version: $ENDPOINT_ENV_VERSION"

}

create_data_asset() {
    local dataset_name="$1"
    local dataset_path="$2"
    local version_var="$3"
    local path_var="$4"
    local requested_version="${5:-}"

    local asset_json
    local asset_type
    local asset_path
    local asset_version

    info "Creating data asset '$dataset_name' from '$dataset_path'..."

    if [[ -n "$requested_version" ]]; then
        # Reuse an existing registration when this exact asset version
        # has already been created.
        if asset_json="$(
            az ml data show \
                --name "$dataset_name" \
                --version "$requested_version" \
                --workspace-name "$WORKSPACE_NAME" \
                --resource-group "$RG" \
                --output json 2>/dev/null
        )"; then
            info "Data asset already exists: ${dataset_name}:${requested_version}"
        else
            asset_json="$(
                az ml data create \
                    --name "$dataset_name" \
                    --version "$requested_version" \
                    --type uri_file \
                    --path "$dataset_path" \
                    --workspace-name "$WORKSPACE_NAME" \
                    --resource-group "$RG" \
                    --output json
            )"
        fi
    else
        # Allow Azure ML to generate the next asset version.
        asset_json="$(
            az ml data create \
                --name "$dataset_name" \
                --type uri_file \
                --path "$dataset_path" \
                --workspace-name "$WORKSPACE_NAME" \
                --resource-group "$RG" \
                --output json
        )"
    fi

    asset_type="$(jq -r '.type' <<<"$asset_json")"
    asset_path="$(jq -r '.path' <<<"$asset_json")"
    asset_version="$(jq -r '.version' <<<"$asset_json")"

    [[ "$asset_type" == "uri_file" ]] ||
        fail "Type mismatch. Expected 'uri_file', got '$asset_type'."

    [[ -n "$asset_path" && "$asset_path" != "null" ]] ||
        fail "Azure ML returned an invalid asset path."

    [[ -n "$asset_version" && "$asset_version" != "null" ]] ||
        fail "Azure ML returned an invalid asset version."

    printf -v "$version_var" '%s' "$asset_version"
    printf -v "$path_var" '%s' "$asset_path"

    info "Data asset registered: ${dataset_name}:${asset_version}"
    info "Data asset path: ${asset_path}"
}

create_service_principal() {
    local service_principal_name
    local sp_app_id
    local sp_object_id
    local workspace_storage_account
    local workspace_storage_scope
    local adls_storage_scope
    local role_assignment_id
    local attempt
    local max_attempts=24

    service_principal_name="github-action-key-${RG}"

    echo ">>> Creating GitHub Actions service principal."
    echo ">>> Name: $service_principal_name"

    AZURE_SECRET="$(
        az ad sp create-for-rbac \
            --name "$service_principal_name" \
            --role "Contributor" \
            --scopes "/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${RG}" \
            --sdk-auth \
            --output json
    )"

    export AZURE_SECRET

    sp_app_id="$(
        jq -r '.clientId' <<< "$AZURE_SECRET"
    )"

    if [[ -z "$sp_app_id" || "$sp_app_id" == "null" ]]; then
        echo "ERROR: Could not obtain the service principal application ID." >&2
        return 1
    fi

    # A newly created service principal might not immediately be available
    # through Microsoft Graph.
    sp_object_id=""

    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        sp_object_id="$(
            az ad sp show \
                --id "$sp_app_id" \
                --query id \
                --output tsv 2>/dev/null || true
        )"

        if [[ -n "$sp_object_id" ]]; then
            break
        fi

        echo ">>> Waiting for service principal: attempt $attempt/$max_attempts"
        sleep 5
    done

    if [[ -z "$sp_object_id" ]]; then
        echo "ERROR: Could not resolve the service principal object ID." >&2
        return 1
    fi

    echo ">>> Service principal application ID: $sp_app_id"
    echo ">>> Service principal object ID:      $sp_object_id"

    if [[ "$STORAGE_TYPE" == "blobstore" || "$STORAGE_TYPE" == "na" ]]; then

        workspace_storage_account="$(
            az ml datastore show \
                --name workspaceblobstore \
                --resource-group "$RG" \
                --workspace-name "$WORKSPACE_NAME" \
                --query account_name \
                --output tsv
        )"

        if [[ -z "$workspace_storage_account" ]]; then
            echo "ERROR: Could not determine the workspace storage account." >&2
            return 1
        fi

        echo ">>> Workspace storage:                $workspace_storage_account"

        workspace_storage_scope="$(
            az storage account show \
                --name "$workspace_storage_account" \
                --resource-group "$RG" \
                --query id \
                --output tsv
        )"

        if [[ -z "$workspace_storage_scope" ||
          "$workspace_storage_scope" == "null" ]]; then
            echo "ERROR: Could not determine the workspace storage scope." >&2
            return 1
        fi

        # Allows the pipeline to retrieve the workspace storage account key
        # when a script requires it.
        ensure_role_assignment \
            "$sp_object_id" \
            "ServicePrincipal" \
            "Storage Account Key Operator Service Role" \
            "$workspace_storage_scope"

        # Allows reading data from the workspace storage account.
        ensure_role_assignment \
            "$sp_object_id" \
            "ServicePrincipal" \
            "Storage Blob Data Reader" \
            "$workspace_storage_scope"
        
        ensure_role_assignment \
            "$sp_object_id" \
            "ServicePrincipal" \
            "Storage Blob Data Contributor" \
            "$workspace_storage_scope"

    fi


    if [[ "$STORAGE_TYPE" == "datalake" ]]; then

        if [[ -z "$ADLS_ACCOUNT" ]]; then
            echo "ERROR: The generated Data Lake account name is empty." >&2
            return 1
        fi

        echo ">>> Data Lake storage:                $ADLS_ACCOUNT"

        if [[ ${#ADLS_ACCOUNT} -lt 3 || ${#ADLS_ACCOUNT} -gt 24 ]]; then
            echo "ERROR: Invalid Data Lake account name: $ADLS_ACCOUNT" >&2
            echo "Storage account names must contain 3-24 lowercase letters or numbers." >&2
            return 1
        fi

        adls_storage_scope="$(
            az storage account show \
                --name "$ADLS_ACCOUNT" \
                --resource-group "$RG" \
                --query id \
                --output tsv
        )"

        if [[ -z "$adls_storage_scope" ]]; then
            echo "ERROR: Data Lake storage account was not found: $ADLS_ACCOUNT" >&2
            return 1
        fi

        # Allows the GitHub Actions pipeline to upload, overwrite and verify
        # files in the ADLS Gen2 storage account.
        ensure_role_assignment \
            "$sp_object_id" \
            "ServicePrincipal" \
            "Storage Blob Data Contributor" \
            "$adls_storage_scope"

        echo ">>> Waiting for the Data Lake role assignment to become visible."

        for ((attempt = 1; attempt <= max_attempts; attempt++)); do
            role_assignment_id="$(
                az role assignment list \
                    --assignee-object-id "$sp_object_id" \
                    --scope "$adls_storage_scope" \
                    --query "[?roleDefinitionName=='Storage Blob Data Contributor'] | [0].id" \
                    --output tsv 2>/dev/null || true
            )"

            if [[ -n "$role_assignment_id" ]]; then
                echo ">>> Data Lake role assignment is visible."
                break
            fi

            echo ">>> Waiting for role assignment: attempt $attempt/$max_attempts"
            sleep 5
        done

        if [[ -z "${role_assignment_id:-}" ]]; then
            echo "ERROR: Data Lake role assignment did not become visible." >&2
            return 1
        fi
    fi

    echo ">>> Service principal configuration completed."
    echo ">>> Add the following JSON to the TEST_AZURE_SECRET GitHub secret:"
    echo "$AZURE_SECRET"
}

configure_source() {
    local ACCOUNT_NAME
    local CONTAINER_NAME
    local datastore_template
    local source_asset_uri
    local source_relative_path

    if [[ "$STORAGE_TYPE" == "datalake" ]]; then
        datastore_template="azureml/feature_store/datastores/ws_datalake.yml"
        DATASTORE_YAML="tmp/ws_datalake.yml"
        DATASTORE_NAME="walmart_feature_datalake"
        SOURCE_CONTAINER="$ADLS_FILESYSTEM"
        SOURCE_DATA_URI="abfss://${ADLS_FILESYSTEM}@${ADLS_ACCOUNT}.dfs.core.windows.net/${ADLS_FILE_PATH_CURRENT}"

        export ADLS_ACCOUNT ADLS_FILESYSTEM DATASTORE_NAME
        render_template "$datastore_template" "$DATASTORE_YAML" \
            '${ADLS_ACCOUNT} ${ADLS_FILESYSTEM} ${DATASTORE_NAME}'
        return
    fi

    datastore_template="azureml/feature_store/datastores/ws_datastore.yml"
    DATASTORE_YAML="tmp/datastore.yaml"
    DATASTORE_NAME="blob_datastore"
    SOURCE_STORAGE_ACCOUNT="$(az ml datastore show --name workspaceblobstore --workspace-name "$WORKSPACE_NAME" --resource-group "$RG" --query account_name --output tsv)"
    SOURCE_CONTAINER="$(az ml datastore show --name workspaceblobstore --workspace-name "$WORKSPACE_NAME" --resource-group "$RG" --query container_name --output tsv)"
    [[ -n "$SOURCE_STORAGE_ACCOUNT" && "$SOURCE_STORAGE_ACCOUNT" != "null" ]] || fail "Could not determine the Blob source storage account."
    [[ -n "$SOURCE_CONTAINER" && "$SOURCE_CONTAINER" != "null" ]] || fail "Could not determine the Blob source container."

    source_asset_uri="$(az ml data show --name "$DATA_ASSET_NAME" --label latest --resource-group "$RG" --workspace-name "$WORKSPACE_NAME" --query path --output tsv)"
    [[ "$source_asset_uri" == *"/paths/"* ]] || fail "Unexpected data asset URI: $source_asset_uri"
    source_relative_path="${source_asset_uri#*/paths/}"
    SOURCE_DATA_URI="azureml://subscriptions/${AZURE_SUBSCRIPTION_ID}/resourcegroups/${RG}/workspaces/${FEATURE_STORE_WS}/datastores/${DATASTORE_NAME}/paths/${source_relative_path}"

    ACCOUNT_NAME="$SOURCE_STORAGE_ACCOUNT"
    CONTAINER_NAME="$SOURCE_CONTAINER"
    export ACCOUNT_NAME CONTAINER_NAME DATASTORE_NAME
    render_template "$datastore_template" "$DATASTORE_YAML" \
        '${ACCOUNT_NAME} ${CONTAINER_NAME} ${DATASTORE_NAME}'
}

create_feature_store(){
    local ENTITY_NAME="walmart_sales_entity"
    local ENTITY_VERSION="1"
    local FS_STAGE
    local entity_template="azureml/feature_store/entities/ws_entity.yml"
    local entity_yaml="tmp/entity.yaml"
    local feature_set_spec_template="azureml/feature_store/feature_sets/walmart_sales_features/spec/FeatureSetSpec.yaml"
    local feature_set_spec_yaml="tmp/spec/FeatureSetSpec.yaml"
    local feature_set_template="azureml/feature_store/feature_sets/walmart_sales_features/ws_feature_set.yml"
    local feature_set_yaml="tmp/ws_feature_set.yml"
    local transformation_code="tmp/spec/transformation_code/walmart_transformer.py"
    local transformation_template="azureml/feature_store/feature_sets/walmart_sales_features/spec/transformation_code/walmart_transformer.py"

    # ------------------------------------------------------------------
    # Determine feature-store stage
    # ------------------------------------------------------------------
    
    if [[ "$ENV_TYPE" == "prod" ]]; then
        FS_STAGE="Production"
    else
        FS_STAGE="Development"
    fi

    # ------------------------------------------------------------------
    # Create feature store workspace
    # ------------------------------------------------------------------

    if az ml feature-store show \
        --name "$FEATURE_STORE_WS" \
        --resource-group "$RG" \
        >/dev/null 2>&1; then

        info "Feature store already exists: $FEATURE_STORE_WS"
    else
        az ml feature-store create \
        --name "$FEATURE_STORE_WS" \
        --resource-group "$RG" \
        --location "$LOC"

        info "Feature Store $FEATURE_STORE_WS created."
    fi

    FEATURE_STORE_STORAGE_ID="$(
        az resource show \
            --ids "$FEATURE_STORE_ID" \
            --query properties.storageAccount \
            --output tsv
    )"
    OFFLINE_STORE_SCOPE="$(
        az ml feature-store show \
            --name "$FEATURE_STORE_WS" \
            --resource-group "$RG" \
            --query offline_store.target \
            --output tsv
    )"
    [[ -n "$FEATURE_STORE_STORAGE_ID" && "$FEATURE_STORE_STORAGE_ID" != "null" ]] ||
        fail "Could not resolve the feature-store storage account resource ID."
    [[ -n "$OFFLINE_STORE_SCOPE" && "$OFFLINE_STORE_SCOPE" != "null" ]] ||
        fail "The feature store has no offline store configured."

    # ------------------------------------------------------------------
    # Resolve the selected source and render its datastore YAML.
    # ------------------------------------------------------------------

    configure_source

    if az ml datastore show \
        --name "$DATASTORE_NAME" \
        --workspace-name "$FEATURE_STORE_WS" \
        --resource-group "$RG" \
        >/dev/null 2>&1; then 
        
        info "Datastore already exists: $DATASTORE_NAME"
    else

        az ml datastore create \
            --file "$DATASTORE_YAML" \
            --workspace-name "$FEATURE_STORE_WS" \
            --resource-group "$RG"
        
        info "Datastore created in feature store: $FEATURE_STORE_WS"
    fi

    # ------------------------------------------------------------------
    # Create entity in the feature store
    # ------------------------------------------------------------------

    export ENTITY_VERSION FS_STAGE ENTITY_NAME
    render_template "$entity_template" "$entity_yaml" \
        '${ENTITY_VERSION} ${FS_STAGE} ${ENTITY_NAME}'

    if az ml feature-store-entity show \
        --name "$ENTITY_NAME" \
        --version "$ENTITY_VERSION" \
        --resource-group "$RG" \
        --feature-store-name "$FEATURE_STORE_WS" \
        >/dev/null 2>&1; then

        info "Entity already exists: ${ENTITY_NAME}:${ENTITY_VERSION}"
    else
        az ml feature-store-entity create \
            --file "$entity_yaml" \
            --resource-group "$RG" \
            --feature-store-name "$FEATURE_STORE_WS"

        info "Entity created: ${ENTITY_NAME}:${ENTITY_VERSION}"
    fi


    # ------------------------------------------------------------------
    # Configure the common materialization and caller permissions.
    # ------------------------------------------------------------------

    configure_storage_access


    # ------------------------------------------------------------------
    # Create feature set
    # ------------------------------------------------------------------

    mkdir -p \
        "$(dirname "$feature_set_yaml")" \
        "$(dirname "$feature_set_spec_yaml")" \
        "$(dirname "$transformation_code")"


    info "Feature source ($STORAGE_TYPE): $SOURCE_DATA_URI"


    export FEATURE_SET_VERSION FS_STAGE FEATURE_SET_NAME SOURCE_DATA_URI ENTITY_VERSION ENTITY_NAME
    render_template "$feature_set_template" "$feature_set_yaml" \
        '${FS_STAGE} ${FEATURE_SET_VERSION} ${FEATURE_SET_NAME} ${ENTITY_VERSION} ${ENTITY_NAME}'
    render_template "$feature_set_spec_template" "$feature_set_spec_yaml" \
        '${SOURCE_DATA_URI}'
    cp "$transformation_template" "$transformation_code"

    if az ml feature-set show \
        --name "$FEATURE_SET_NAME" \
        --version "$FEATURE_SET_VERSION" \
        --resource-group "$RG" \
        --feature-store-name "$FEATURE_STORE_WS" \
        >/dev/null 2>&1; then

        info "Feature set already exists: ${FEATURE_SET_NAME}:${FEATURE_SET_VERSION}"
    else
        az ml feature-set create \
            --file "$feature_set_yaml" \
            --resource-group "$RG" \
            --feature-store-name "$FEATURE_STORE_WS"

        info "Feature set created: ${FEATURE_SET_NAME}:${FEATURE_SET_VERSION}"
    fi


    # ------------------------------------------------------------------
    # Run backfill and wait for all materialization jobs
    # ------------------------------------------------------------------
    run_backfill

}

run_backfill(){
    local backfill_end_time="2012-11-01T00:00:00Z"
    local backfill_result
    local backfill_start_time="2010-02-05T00:00:00Z"
    local job_id
    local job_status
    local -a backfill_job_ids=()

    backfill_result="$(
        az ml feature-set backfill \
            --name "$FEATURE_SET_NAME" \
            --version "$FEATURE_SET_VERSION" \
            --feature-store-name "$FEATURE_STORE_WS" \
            --resource-group "$RG" \
            --by-data-status '["None", "Incomplete"]' \
            --feature-window-start-time "$backfill_start_time" \
            --feature-window-end-time "$backfill_end_time" \
            --output json
    )"

    jq . <<<"$backfill_result"

    while IFS= read -r job_id; do
        [[ -n "$job_id" ]] && backfill_job_ids+=("$job_id")
    done < <(
        echo "$backfill_result" |
            jq -r '(.jobIds // .job_ids // [])[]'
    )

    if [[ "${#backfill_job_ids[@]}" -eq 0 ]]; then
        info "No new backfill jobs were submitted."
        info "The requested intervals may already be materialized."

        info "Existing feature-store materialization jobs:"

        az ml job list \
            --resource-group "$RG" \
            --workspace-name "$FEATURE_STORE_WS" \
            --query "[?starts_with(name, 'fs-mat-')].{
                Job:name,
                Status:status,
                Created:creation_context.created_at
            }" \
            --output table

    else
        info "Submitted ${#backfill_job_ids[@]} backfill job(s)."

        for job_id in "${backfill_job_ids[@]}"; do
            info "Waiting for materialization job: $job_id"

            while true; do
                job_status="$(
                    az ml job show \
                        --name "$job_id" \
                        --resource-group "$RG" \
                        --workspace-name "$FEATURE_STORE_WS" \
                        --query status \
                        --output tsv
                )"

                info "Backfill job $job_id status: $job_status"

                case "$job_status" in
                    Completed)
                        info "Backfill job completed successfully: $job_id"
                        break
                        ;;

                    Failed|Canceled|Cancelled)
                        echo \
                            "Backfill job did not complete successfully: $job_id ($job_status)" \
                            >&2

                        az ml job show \
                            --name "$job_id" \
                            --resource-group "$RG" \
                            --workspace-name "$FEATURE_STORE_WS" \
                            --output yaml

                        echo "================ JOB LOGS ================" >&2

                        az ml job stream \
                            --name "$job_id" \
                            --resource-group "$RG" \
                            --workspace-name "$FEATURE_STORE_WS" \
                            2>&1 || true

                        echo "============== END JOB LOGS ==============" >&2

                        exit 1
                        ;;

                    *)
                        sleep 30
                        ;;
                esac
            done
        done
    fi
}

create_engineered_data(){
    if [[ -z "$FEATURE_STORE_STORAGE_ID" || \
          "$FEATURE_STORE_STORAGE_ID" == "null" ]]; then
        echo "ERROR: Could not determine the feature-store storage resource ID." >&2
        exit 1
    fi

    if [[ -z "$OFFLINE_STORE_SCOPE" ||
          "$OFFLINE_STORE_SCOPE" == "null" ]]; then
        echo "ERROR: Could not determine the offline-store container." >&2
        exit 1
    fi

    # Extract the names from their Azure resource IDs.
    export OFFLINE_STORAGE_ACCOUNT="${FEATURE_STORE_STORAGE_ID##*/}"
    export OFFLINE_STORAGE_CONTAINER="${OFFLINE_STORE_SCOPE##*/}"

    info "Offline storage account: $OFFLINE_STORAGE_ACCOUNT"
    info "Offline storage container: $OFFLINE_STORAGE_CONTAINER"

    export AZURE_STORAGE_KEY="$(
        az storage account keys list \
            --resource-group "$RG" \
            --account-name "$OFFLINE_STORAGE_ACCOUNT" \
            --query "[0].value" \
            --output tsv
    )"

    if [[ -z "${AZURE_STORAGE_KEY:-}" || \
          "$AZURE_STORAGE_KEY" == "null" ]]; then
        echo "ERROR: Could not retrieve the feature-store storage key." >&2
        exit 1
    fi

    info "Feature-store storage key retrieved successfully."


    az ml job list \
            --resource-group "$RG" \
            --workspace-name "$FEATURE_STORE_WS" \
            --query "[?starts_with(name, 'fs-mat-')].{
                Job:name,
                Status:status,
                Created:creation_context.created_at
            }" \
            --output table


    export FEATURE_SET_NAME FEATURE_SET_VERSION
    PYTHONPATH=src python3 -m walmart_ml.pipeline.create_all_weeks_features
    
}

ensure_adls_directory_exists() {
    local directory="$1"
    local exists

    exists="$(
        az storage fs directory exists \
            --name "$directory" \
            --file-system "$ADLS_FILESYSTEM" \
            --account-name "$ADLS_ACCOUNT" \
            --auth-mode login \
            --query exists \
            --output tsv
    )"

    if [[ "$exists" == "true" ]]; then
        info "ADLS directory already exists: $directory"
    else
        az storage fs directory create \
            --name "$directory" \
            --file-system "$ADLS_FILESYSTEM" \
            --account-name "$ADLS_ACCOUNT" \
            --auth-mode login \
            --output none

        info "ADLS directory created: $directory"
    fi
}

create_datalake(){
    local current_exists
    local filesystem_exists
    local version_exists

    az storage account check-name \
        --name "$ADLS_ACCOUNT" \
        --query nameAvailable \
        --output tsv
    
    az storage account create \
        --name "$ADLS_ACCOUNT" \
        --resource-group "$RG" \
        --location "$LOC" \
        --sku Standard_LRS \
        --kind StorageV2 \
        --enable-hierarchical-namespace true

    az storage account show \
        --name "$ADLS_ACCOUNT" \
        --resource-group "$RG" \
        --query '{
            Name:name,
            Location:location,
            HierarchicalNamespace:isHnsEnabled
        }' \
        --output table
    
    filesystem_exists="$(
        az storage fs exists \
            --name "$ADLS_FILESYSTEM" \
            --account-name "$ADLS_ACCOUNT" \
            --auth-mode login \
            --query exists \
            --output tsv
    )"

    if [[ "$filesystem_exists" == "true" ]]; then
        info "ADLS filesystem already exists: $ADLS_FILESYSTEM"
    else
        az storage fs create \
            --name "$ADLS_FILESYSTEM" \
            --account-name "$ADLS_ACCOUNT" \
            --auth-mode login \
            --output none

        info "ADLS filesystem created: $ADLS_FILESYSTEM"
    fi
    
    ensure_adls_directory_exists "${ADLS_FILE_PATH_CURRENT%/*}"
    ensure_adls_directory_exists "${ADLS_FILE_PATH_VERSIONS%/*}"

    version_exists="$(
        az storage fs file exists \
            --account-name "$ADLS_ACCOUNT" \
            --file-system "$ADLS_FILESYSTEM" \
            --path "$ADLS_FILE_PATH_VERSIONS" \
            --auth-mode login \
            --query exists \
            --output tsv
    )"

    current_exists="$(
        az storage fs file exists \
            --account-name "$ADLS_ACCOUNT" \
            --file-system "$ADLS_FILESYSTEM" \
            --path "$ADLS_FILE_PATH_CURRENT" \
            --auth-mode login \
            --query exists \
            --output tsv
    )"

    if [[ "$version_exists" == "true" ]]; then
        info "ADLS dataset version already exists: $ADLS_VERSION"

        if [[ "$current_exists" != "true" ]]; then
            fail \
                "The archived dataset version exists, but the current file is missing: $ADLS_FILE_PATH_CURRENT"
        fi

        info "Leaving the archived version and current file unchanged."
    else
        az storage fs file upload \
            --account-name "$ADLS_ACCOUNT" \
            --file-system "$ADLS_FILESYSTEM" \
            --path "$ADLS_FILE_PATH_VERSIONS" \
            --source "$SOURCE_FILE" \
            --overwrite false \
            --auth-mode login

        az storage fs file upload \
            --account-name "$ADLS_ACCOUNT" \
            --file-system "$ADLS_FILESYSTEM" \
            --path "$ADLS_FILE_PATH_CURRENT" \
            --source "$SOURCE_FILE" \
            --overwrite true \
            --auth-mode login

        info "Uploaded ADLS dataset version: $ADLS_VERSION"
    fi
    
    az storage fs file show \
        --account-name "$ADLS_ACCOUNT" \
        --file-system "$ADLS_FILESYSTEM" \
        --path "$ADLS_FILE_PATH_VERSIONS" \
        --auth-mode login \
        --output table
    
    az storage fs file show \
        --account-name "$ADLS_ACCOUNT" \
        --file-system "$ADLS_FILESYSTEM" \
        --path "$ADLS_FILE_PATH_CURRENT" \
        --auth-mode login \
        --output table

    info "ADLS dataset version: $ADLS_VERSION"
    info "ADLS versioned path: $ADLS_FILE_PATH_VERSIONS"
    info "ADLS current path: $ADLS_FILE_PATH_CURRENT"


}

configure_datalake_datastore() {
    local compute_principal_id
    local datastore_name="walmart_datalake"
    local datastore_template="azureml/feature_store/datastores/datalake-datastore.template.yaml"
    local datastore_yaml="tmp/datalake-datastore.yaml"
    local storage_account_scope

    [[ "$STORAGE_TYPE" == "datalake" ]] || return 0

    STORAGE_ACCOUNT="$(
        az storage account list \
            --resource-group "$RG" \
            --query "[?isHnsEnabled].name | [0]" \
            --output tsv
    )"

    [[ -n "$STORAGE_ACCOUNT" && "$STORAGE_ACCOUNT" != "null" ]] ||
        fail "Could not find an HNS-enabled storage account in resource group '$RG'."

    export STORAGE_ACCOUNT
    render_template \
        "$datastore_template" \
        "$datastore_yaml" \
        '${STORAGE_ACCOUNT}'

    if az ml datastore show \
        --name "$datastore_name" \
        --resource-group "$RG" \
        --workspace-name "$WORKSPACE_NAME" \
        >/dev/null 2>&1; then

        info "Datastore already exists in workspace '$WORKSPACE_NAME': $datastore_name"
    else
        az ml datastore create \
            --file "$datastore_yaml" \
            --resource-group "$RG" \
            --workspace-name "$WORKSPACE_NAME" \
            --output none

        info "Datastore created in workspace '$WORKSPACE_NAME': $datastore_name"
    fi

    compute_principal_id="$(
        az ml compute show \
            --name "$CLUSTER_NAME" \
            --resource-group "$RG" \
            --workspace-name "$WORKSPACE_NAME" \
            --query identity.principal_id \
            --output tsv
    )"

    [[ -n "$compute_principal_id" && "$compute_principal_id" != "null" ]] ||
        fail "Compute '$CLUSTER_NAME' does not have a system-assigned managed identity."

    storage_account_scope="$(
        az storage account show \
            --name "$STORAGE_ACCOUNT" \
            --resource-group "$RG" \
            --query id \
            --output tsv
    )"

    [[ -n "$storage_account_scope" && "$storage_account_scope" != "null" ]] ||
        fail "Could not resolve the resource ID for storage account '$STORAGE_ACCOUNT'."

    ensure_role_assignment \
        "$compute_principal_id" \
        "ServicePrincipal" \
        "Storage Blob Data Reader" \
        "$storage_account_scope"
}

configure_storage_access() {
    local account_type account_type_lower caller_object_id caller_principal_type
    local source_storage_id
    local materialization_identity_id materialization_principal_id

    account_type="$AZURE_ACCOUNT_TYPE"

    account_type_lower="$(
        printf '%s' "$account_type" |
            tr '[:upper:]' '[:lower:]'
    )"

    case "$account_type_lower" in
        user)
            caller_principal_type="User"
            caller_object_id="$(az ad signed-in-user show --query id --output tsv)"
            ;;
        serviceprincipal)
            caller_principal_type="ServicePrincipal"
            caller_object_id="$(az ad sp show --id "$AZURE_ACCOUNT_USER_NAME" --query id --output tsv)"
            ;;
        *) fail "Unsupported Azure account type: $account_type" ;;
    esac

    source_storage_id="$(az storage account show --name "$SOURCE_STORAGE_ACCOUNT" --resource-group "$RG" --query id --output tsv)"
    materialization_identity_id="$(az ml feature-store show --name "$FEATURE_STORE_WS" --resource-group "$RG" --query 'materialization_identity.resource_id || materialization_identity.resource' --output tsv)"
    [[ -n "$materialization_identity_id" && "$materialization_identity_id" != "null" && "$materialization_identity_id" != "None" ]] || fail "Could not resolve the materialization identity."
    materialization_principal_id="$(az identity show --ids "$materialization_identity_id" --query principalId --output tsv)"
    [[ -n "$materialization_principal_id" && "$materialization_principal_id" != "null" ]] || fail "Could not resolve the materialization principal ID."
    ensure_role_assignment "$caller_object_id" "$caller_principal_type" "AzureML Data Scientist" "$FEATURE_STORE_ID"
    ensure_role_assignment "$caller_object_id" "$caller_principal_type" "Storage Blob Data Reader" "$source_storage_id"
    ensure_role_assignment "$caller_object_id" "$caller_principal_type" "Storage Blob Data Contributor" "$source_storage_id"
    ensure_role_assignment "$caller_object_id" "$caller_principal_type" "Storage Blob Data Reader" "$OFFLINE_STORE_SCOPE"
    ensure_role_assignment "$caller_object_id" "$caller_principal_type" "Storage Blob Data Contributor" "$FEATURE_STORE_STORAGE_ID"

    ensure_role_assignment "$materialization_principal_id" "ServicePrincipal" "AzureML Data Scientist" "$FEATURE_STORE_ID"
    ensure_role_assignment "$materialization_principal_id" "ServicePrincipal" "Storage Blob Data Reader" "$source_storage_id"
    ensure_role_assignment "$materialization_principal_id" "ServicePrincipal" "Storage Blob Data Contributor" "$OFFLINE_STORE_SCOPE"

    wait_for_selected_source_access 20 30
    info "Allowing additional time for the materialization identity..."
    sleep 60
}

wait_for_selected_source_access() {
    local max_attempts="${1:-20}"
    local delay_seconds="${2:-30}"
    local attempt

    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        if [[ "$STORAGE_TYPE" == "datalake" ]]; then
            az storage fs file show --account-name "$SOURCE_STORAGE_ACCOUNT" --file-system "$SOURCE_CONTAINER" --path "$ADLS_FILE_PATH_CURRENT" --auth-mode login >/dev/null 2>&1 && return 0
        else
            az storage blob list --account-name "$SOURCE_STORAGE_ACCOUNT" --container-name "$SOURCE_CONTAINER" --auth-mode login --num-results 1 --output none >/dev/null 2>&1 && return 0
        fi
        info "Waiting for $STORAGE_TYPE access: attempt $attempt/$max_attempts"
        sleep "$delay_seconds"
    done
    fail "$STORAGE_TYPE access did not become available."
}

ensure_role_assignment() {
    local principal_id="$1"
    local principal_type="$2"
    local role_name="$3"
    local scope="$4"

    local assignment_count

    assignment_count="$(
        az role assignment list \
            --assignee-object-id "$principal_id" \
            --scope "$scope" \
            --include-inherited \
            --query "[?roleDefinitionName=='${role_name}'] | length(@)" \
            --output tsv
    )"

    if (( assignment_count > 0 )); then
        info "'$role_name' already assigned on $scope"
        return 0
    fi

    az role assignment create \
        --assignee-object-id "$principal_id" \
        --assignee-principal-type "$principal_type" \
        --role "$role_name" \
        --scope "$scope" \
        --output none

    info "Granted '$role_name' on $scope"
}

ensure_endpoint() {
    local endpoint_type="$1"
    local endpoint_prefix="$2"
    local template_path="$3"
    local output_path="$4"
    local name_variable="$5"
    local substitutions="$6"
    local max_attempts=20
    local attempt
    local candidate_name
    local creation_json
    local dns_name
    local endpoint_command
    local endpoint_name
    local endpoint_state
    local scoring_uri

    case "$endpoint_type" in
        batch) endpoint_command="batch-endpoint" ;;
        realtime) endpoint_command="online-endpoint" ;;
        *) fail "Unsupported endpoint type: $endpoint_type" ;;
    esac

    endpoint_name="$(
        az ml "$endpoint_command" list \
            --resource-group "$RG" \
            --workspace-name "$WORKSPACE_NAME" \
            --query "[?starts_with(name, '${endpoint_prefix}')].name | [0]" \
            --output tsv
    )"

    if [[ -n "$endpoint_name" ]]; then
        info "Existing $endpoint_type endpoint found: '$endpoint_name'"
    else
        info "No $endpoint_type endpoint beginning with '$endpoint_prefix' exists."

        for ((attempt = 1; attempt <= max_attempts; attempt++)); do
            candidate_name="${endpoint_prefix}-$(openssl rand -hex 5)"
            dns_name="${candidate_name}.${LOC}.inference.ml.azure.com"
            info "Checking $endpoint_type endpoint name: '$candidate_name'"

            # Check if DNS record exists via host/nslookup
            if host "$dns_name" >/dev/null 2>&1 ||
                nslookup "$dns_name" >/dev/null 2>&1; then
                info "Name '$candidate_name' is already in use in '$LOC'."
            else
                endpoint_name="$candidate_name"
                printf -v "$name_variable" '%s' "$endpoint_name"
                export "$name_variable"
                render_template "$template_path" "$output_path" "$substitutions"
                
                creation_json="$(
                    az ml "$endpoint_command" create \
                        --file "$output_path" \
                        --resource-group "$RG" \
                        --workspace-name "$WORKSPACE_NAME" \
                        --output json
                )"
                endpoint_state="$(jq -r '.provisioning_state' <<<"$creation_json")"
                scoring_uri="$(jq -r '.scoring_uri // empty' <<<"$creation_json")"

                if ! printf '%s' "$endpoint_state" | grep -Eqi '(^|\.)succeeded$'; then
                    fail "$endpoint_type endpoint '$endpoint_name' creation failed (provisioningState: $endpoint_state)."
                fi

                info "$endpoint_type endpoint '$endpoint_name' created. Scoring URI: $scoring_uri"
                break
            fi
        done

        if (( attempt > max_attempts )); then
            fail "Could not find an available $endpoint_type endpoint name."
        fi
    fi

    printf -v "$name_variable" '%s' "$endpoint_name"
    export "$name_variable"
}

grant_realtime_endpoint_workspace_access() {
    local endpoint_name="$1"
    local endpoint_principal_id

    endpoint_principal_id="$(
        az ml online-endpoint show \
            --name "$endpoint_name" \
            --resource-group "$RG" \
            --workspace-name "$WORKSPACE_NAME" \
            --query identity.principal_id \
            --output tsv
    )"

    [[ -n "$endpoint_principal_id" && "$endpoint_principal_id" != "null" ]] ||
        fail "Realtime endpoint '$endpoint_name' has no managed identity principal ID."

    [[ -n "$WORKSPACE_STORAGE_ID" && "$WORKSPACE_STORAGE_ID" != "null" ]] ||
        fail "Could not resolve the workspace storage account resource ID."

    info "Realtime endpoint principal ID: $endpoint_principal_id"

    ensure_role_assignment "$endpoint_principal_id" "ServicePrincipal" \
        "AzureML Data Scientist" "$WORKSPACE_ID"
    ensure_role_assignment "$endpoint_principal_id" "ServicePrincipal" \
        "Storage Blob Data Reader" "$WORKSPACE_STORAGE_ID"
}

grant_batch_compute_workspace_access() {
    local compute_principal_id

    compute_principal_id="$(
        az ml compute show \
            --name "$CLUSTER_NAME" \
            --resource-group "$RG" \
            --workspace-name "$WORKSPACE_NAME" \
            --query identity.principal_id \
            --output tsv
    )"

    [[ -n "$compute_principal_id" && "$compute_principal_id" != "null" ]] ||
        fail "Batch compute '$CLUSTER_NAME' has no managed identity principal ID."

    [[ -n "$WORKSPACE_STORAGE_ID" && "$WORKSPACE_STORAGE_ID" != "null" ]] ||
        fail "Could not resolve the workspace storage account resource ID."

    info "Batch compute principal ID: $compute_principal_id"

    # Allows the batch scoring code to resolve the registered
    # walmart_materialized_features data asset in this workspace.
    ensure_role_assignment "$compute_principal_id" "ServicePrincipal" \
        "AzureML Data Scientist" "$WORKSPACE_ID"

    # Allows read-only access to the data asset files held in the workspace's
    # default Blob Storage account.
    ensure_role_assignment "$compute_principal_id" "ServicePrincipal" \
        "Storage Blob Data Reader" "$WORKSPACE_STORAGE_ID"
}

create_batch_endpoint() {
    local BATCH_ENDPOINT_DESC="XGBoost Walmart Sales Prediction"
    export BATCH_ENDPOINT_DESC

    ensure_endpoint \
        batch \
        "batch-walmart-sales" \
        "azureml/batch-endpoint.template.yaml" \
        "azureml/batch-endpoint.yaml" \
        BATCH_ENDPOINT_NAME \
        '${BATCH_ENDPOINT_NAME} ${BATCH_ENDPOINT_DESC}'

    grant_batch_compute_workspace_access
}

create_rt_endpoint() {
    ensure_endpoint \
        realtime \
        "rt-walmart-sales" \
        "azureml/rt-endpoint.template.yaml" \
        "azureml/rt-endpoint.yaml" \
        RT_ENDPOINT_NAME \
        '${RT_ENDPOINT_NAME}'

    grant_realtime_endpoint_workspace_access "$RT_ENDPOINT_NAME"
}

env_type_lower=$(echo "$ENV_TYPE" | tr '[:upper:]' '[:lower:]')
if [ "$env_type_lower" = "prod" ]; then
    STORAGE_TYPE="na"
fi

check_prereq
create_resource_group
create_container_registry
create_workspace
create_endpoint_environment


if [ "$env_type_lower" = "prod" ]; then
    info "Environment is production. Skipping training environment creation."
    create_compute 1
else
    info "Environment is NOT production. Creating training environment."
    create_training_environment
    #create_compute 2
    create_compute 1

    create_data_asset "test_data" "./data/test_data.csv" TEST_DATA_VERSION TEST_DATA_PATH

    if [[ "$STORAGE_TYPE" == "datalake" ]]; then
        create_datalake
        configure_datalake_datastore
        SOURCE_DATA_VERSION="$ADLS_VERSION"
    else    
        create_data_asset "$DATA_ASSET_NAME" "$SOURCE_FILE" WALMART_SALES_DATASET_VERSION WALMART_SALES_DATASET_PATH
        SOURCE_DATA_VERSION="$WALMART_SALES_DATASET_VERSION"
    fi

    [[ -n "$SOURCE_DATA_VERSION" ]] ||
    fail "Could not determine the source dataset version."

    create_feature_store
    create_engineered_data
    create_data_asset \
        "walmart_materialized_features" \
        "$ENGINEERED_DATA_PATH" \
        MATERIALIZED_WALMART_SALES_DATASET_VERSION \
        MATERIALIZED_WALMART_SALES_DATASET_PATH \
        "$SOURCE_DATA_VERSION"
fi



create_rt_endpoint
create_batch_endpoint
create_service_principal


info "CI/CD environment secrets:"
info "TEST/PROD_AZURE_SECRET: $AZURE_SECRET"


info "CI/CD environment variables:"
info "TEST/PROD_PREFIX: $RESOURCE_PREFIX"
if [ "$env_type_lower" = "test" ]; then
    info "TEST_STORAGE_TYPE: $STORAGE_TYPE"
    info "TEST_TRAINING_ENV_VERSION: $TRAINING_ENV_VERSION"
fi
info "TEST/PROD_ENDPOINT_ENV_VERSION: $ENDPOINT_ENV_VERSION"










