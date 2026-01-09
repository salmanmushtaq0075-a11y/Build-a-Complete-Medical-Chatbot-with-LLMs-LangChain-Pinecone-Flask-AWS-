# Medical Chatbot using Gemini, LangChain, Pinecone & Flask

A secure, enterprise-grade AI medical assistant capable of answering health queries using RAG (Retrieval Augmented Generation) and maintaining conversation history.

## 🛠️ Local Setup & Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/salmanmushtaq0075-a11y/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS-.git

```

### Step 2: Create a Conda Environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot

```

### Step 3: Install Requirements

```bash
pip install -r requirements.txt

```

### Step 4: Setup Environment Variables

Create a `.env` file in the root directory and add your keys:

```ini
PINECONE_API_KEY="pcsk_5mUi4d_TyC5CUHWqJEakV9Gix7oiWgXx5cL2rGZxr4r98wpVaZLXKvjbH4hRqXgVmxJg2j"
GOOGLE_API_KEY="AIzaSyD5jayqxAH_f-_ar1WIxmh6TdyHARa0dT4"

```
#Run the following command to store embeddings to pinecone
python store_index.py

#Finally run the following command 
python app.py

### Step 5: Run the Application

```bash
python app.py

```

*The app will start at `http://localhost:5000*`

---

## 🚀 AWS CI/CD Deployment with GitHub Actions

### 1. Login to AWS Console & Create IAM User

Create a new IAM user for deployment with **Programmatic Access**. Attach the following policies directly:

1. `AmazonEC2ContainerRegistryFullAccess`
2. `AmazonEC2FullAccess`

### 2. Create ECR Repository

1. Go to **Elastic Container Registry (ECR)**.
2. Create a new repository (e.g., `medibot-repo`).
3. Note the **URI** (you will need this for the GitHub secret).

### 3. Launch EC2 Instance

1. Launch a new EC2 instance (Ubuntu 22.04 LTS recommended).
2. Ensure the Security Group allows **Inbound Traffic** on port `5000` (Custom TCP) and `22` (SSH).

### 4. Install Docker on EC2

Connect to your EC2 instance via SSH and run the following commands:

```bash
# Update packages
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (avoids using sudo for docker commands)
sudo usermod -aG docker ubuntu
newgrp docker

```

### 5. Configure Self-Hosted Runner

1. Go to your GitHub Repository > **Settings** > **Actions** > **Runners**.
2. Click **New self-hosted runner**.
3. Select **Linux** and follow the commands shown to install the runner agent on your EC2 machine.

### 6. Setup GitHub Secrets

Go to **Settings** > **Secrets and variables** > **Actions** > **New repository secret**. Add these:

| Secret Name | Description |
| --- | --- |
| `AWS_ACCESS_KEY_ID` | Your IAM User Access Key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM User Secret Key |
| `AWS_DEFAULT_REGION` | e.g., `us-east-1` |
| `ECR_REPO` | The URI of your ECR repository |
| `PINECONE_API_KEY` | Your Pinecone API Key |
| `GOOGLE_API_KEY` | Your Gemini API Key (Note: Must match `app.py`) |

