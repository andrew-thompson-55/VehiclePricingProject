# This script creates an Azure Service Principal with Contributor role scoped to a specific subscription.
# HOW TO USE:
# 1. Ensure you have the Azure CLI and GitHub CLI installed and authenticated.
# 2. Create a .env file in the same directory as this script with your Azure subscription ID:
#    AZURE_SUBSCRIPTION_ID="your-subscription-id"
# 3. Make this script executable: chmod +x create-azure-sp.sh
# 4. Run this script: ./create-azure-sp.sh
#!/bin/bash

# --- CONFIGURATION ---
githubSecretName="AZURE_CREDENTIALS"

# --- Environment and Project Setup ---
if [ -f .env ]; then
    echo "INFO: Loading environment from .env file..."
    source .env
fi

# --- HARDENED DEBUGGING AND VALIDATION ---
echo "DEBUG: Validating loaded subscription ID..."
# The next line is the most important for debugging. It shows us exactly what the script sees.
echo "DEBUG: Value is: [${AZURE_SUBSCRIPTION_ID}]"

if [ -z "$AZURE_SUBSCRIPTION_ID" ]; then
    echo "❌ FATAL ERROR: The AZURE_SUBSCRIPTION_ID variable is empty or was not loaded correctly."
    echo "   Please check your .env file for typos, extra spaces, or incorrect formatting."
    exit 1
fi
echo "INFO: Subscription ID validated."

# --- SCRIPT LOGIC ---
projectName=$(basename "$(pwd)")
servicePrincipalName="Azure-ARM-${projectName}-SP"
scope="/subscriptions/$AZURE_SUBSCRIPTION_ID"

echo "INFO: Preparing to create Service Principal '$servicePrincipalName' for scope '$scope'"

# --- Step 1: Create the Application Registration ---
echo "INFO: Step 1/6 - Creating App Registration..."
app_json=$(az ad app create --display-name "$servicePrincipalName")
if [ $? -ne 0 ]; then echo "❌ ERROR: Failed to create App Registration."; exit 1; fi
appId=$(echo "$app_json" | grep '"appId":' | cut -d '"' -f 4)
echo "✅ App Registration created with AppID: $appId"

# --- Step 2: Create the Service Principal from the App ---
echo "INFO: Step 2/6 - Creating Service Principal..."
sp_json=$(az ad sp create --id "$appId")
if [ $? -ne 0 ]; then echo "❌ ERROR: Failed to create Service Principal."; exit 1; fi
spObjectId=$(echo "$sp_json" | grep '"id":' | cut -d '"' -f 4)
echo "✅ Service Principal created with Object ID: $spObjectId"

# --- Step 3: Wait for Azure AD Replication ---
echo "INFO: Step 3/6 - Waiting 15 seconds for Azure AD replication..."
sleep 15

# --- Step 4: Assign the Contributor Role (with fix for the warning) ---
echo "INFO: Step 4/6 - Assigning 'Contributor' role..."
# Added --assignee-principal-type to make the script more future-proof
az role assignment create --assignee-object-id "$spObjectId" --role "Contributor" --scope "$scope" --assignee-principal-type "ServicePrincipal"
if [ $? -ne 0 ]; then echo "❌ ERROR: Failed to assign role. Check your permissions."; exit 1; fi
echo "✅ Role assigned successfully."

# --- Step 5: Generate the Client Secret ---
echo "INFO: Step 5/6 - Generating client secret..."
credential_json=$(az ad app credential reset --id "$appId")
if [ $? -ne 0 ]; then echo "❌ ERROR: Failed to generate client secret."; exit 1; fi
clientSecret=$(echo "$credential_json" | grep '"password":' | cut -d '"' -f 4)
tenantId=$(echo "$credential_json" | grep '"tenant":' | cut -d '"' -f 4)
echo "✅ Client secret generated."

# --- Step 6: Construct and Set the GitHub Secret ---
echo "INFO: Step 6/6 - Creating GitHub secret '$githubSecretName'..."
github_secret_json="{\"clientId\": \"$appId\", \"clientSecret\": \"$clientSecret\", \"subscriptionId\": \"$AZURE_SUBSCRIPTION_ID\", \"tenantId\": \"$tenantId\"}"

echo "$github_secret_json" | gh secret set "$githubSecretName"
if [ $? -ne 0 ]; then echo "❌ ERROR: Failed to set GitHub secret."; exit 1; fi

echo ""
echo "✅✅✅ ALL DONE: Service Principal and GitHub secret created successfully."