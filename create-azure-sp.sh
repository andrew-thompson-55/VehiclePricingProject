#!/bin/bash

# This script creates an Azure Service Principal with Contributor role scoped to a specific subscription.
# HOW TO USE:
# 1. Ensure you have the Azure CLI and GitHub CLI installed and authenticated.
# azure-cli: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
#   winget install --exact --id Microsoft.AzureCLI
# github-cli: https://cli.github.com/manual/installation
#   winget install --id GitHub.cli
# 1. Create a .env file in the same directory as this script.
# 2. Add your Azure subscription ID in the .env file as follows:
#    AZURE_SUBSCRIPTION_ID="your-subscription-id"
# 3. Make this script executable:
#    chmod +x create-azure-sp.sh
# 3. Run this script: ./create-azure-sp.sh  

# --- CONFIGURATION: UPDATE THIS VALUE ---
# The name of the secret to be created in your GitHub repository.
githubSecretName="AZURE_CREDENTIALS"


# --- SCRIPT LOGIC: DO NOT CHANGE BELOW THIS LINE ---

# Load environment variables from the .env file
if [ -f .env ]; then
  echo "INFO: Loading subscription ID from .env file..."
  source .env
else
  echo "❌ ERROR: .env file not found. Please create it and add your AZURE_SUBSCRIPTION_ID."
  exit 1
fi

# Check if the subscription ID was loaded successfully
if [ -z "$AZURE_SUBSCRIPTION_ID" ]; then
  echo "❌ ERROR: AZURE_SUBSCRIPTION_ID is not set in your .env file."
  exit 1
fi

# --- Automatic Project Name Detection ---
# Get the project name from the current directory's name.
projectName=$(basename "$(pwd)")
echo "INFO: Project name detected from folder: $projectName"


echo "INFO: Preparing to create Azure Service Principal..."

# Construct a unique and descriptive name for the Service Principal.
servicePrincipalName="Azure-ARM-${projectName}-SP"

echo "INFO: Service Principal name will be: $servicePrincipalName"
echo "INFO: Scoped to subscription ID: $AZURE_SUBSCRIPTION_ID"
echo "INFO: GitHub secret will be named: $githubSecretName"

# The core command now uses the variable loaded from the .env file.
az ad sp create-for-rbac --name "$servicePrincipalName" --role "Contributor" --scopes "/subscriptions/$AZURE_SUBSCRIPTION_ID" --json-auth | gh secret set "$githubSecretName"

# Check the exit code of the `gh` command to confirm success.
if [ $? -eq 0 ]; then
  echo ""
  echo "✅ SUCCESS: GitHub secret '$githubSecretName' was created successfully in your repository."
  echo "You can now use this secret in your GitHub Actions workflows."
else
  echo ""
  echo "❌ ERROR: Failed to create GitHub secret. Please check your permissions and ensure the GitHub CLI is authenticated."
fi