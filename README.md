# Agentic RAG Chatbot

This project is a Gemini-powered agentic RAG chatbot that helps people learn about my professional background, resume, skills, and experience.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment Variables:**

    Create a `.env` file in the project root and add the following variables:

    ```
    GOOGLE_API_KEY="your_gemini_api_key"

    # Database Configuration (choose one)
    DB_TYPE="sqlite"  # or "postgresql"

    # For SQLite
    DB_NAME="feedback.db"

    # For PostgreSQL
    # DB_USER="your_db_user"
    # DB_PASSWORD="your_db_password"
    # DB_HOST="your_db_host"
    # DB_PORT="your_db_port"
    # DB_NAME="your_db_name"

    # Google Drive Configuration
    GOOGLE_DRIVE_FOLDER_ID="your_google_drive_folder_id"
    GOOGLE_API_CREDENTIALS_PATH="path/to/your/credentials.json"
    GOOGLE_CHAT_WEBHOOK_URL="your_google_chat_webhook_url"
    ```

3.  **Google Drive API Setup:**

    *   Enable the Google Drive API in your Google Cloud Platform project.
    *   Create an OAuth 2.0 Client ID and download the `credentials.json` file.
    *   Move the `credentials.json` file to the path specified in the `GOOGLE_API_CREDENTIALS_PATH` environment variable.
    *   Share your Google Drive folder with the client email found in your `credentials.json` file.

## Running the Application with Docker

To run the application with Docker, you can use the provided `docker-compose.yml` file.

1.  **Build the Docker image:**

    ```bash
    docker-compose build
    ```

2.  **Run the Docker container:**

    ```bash
    docker-compose up -d
    ```

This will build the Docker image and run the container in the background. The cron job will be set up automatically and the vector store will be updated every day at midnight. An alert will be sent to the Google Chat webhook with the changes.

## Running the Application Locally

To run the FastAPI server, use the following command from the project root directory:

```bash
uvicorn app.main:app --reload
```

Once running, the API documentation will be available at `http://127.0.0.1:8000/docs`.

## Updating the Vectorstore

To sync your Google Drive folder and update the vectorstore, run the following command:

```bash
python pipeline/sync.py
```

This will sync the Google Drive folder with the local `data` folder and update the vectorstore only if there are changes.

### Automating with Cron Job

To automate the synchronization process, you can use the provided `setup_cron.sh` script to create a cron job.

1.  **Make the script executable:**

    ```bash
    chmod +x setup_cron.sh
    ```

2.  **Run the script:**

    ```bash
    ./setup_cron.sh
    ```

    The script will prompt you to enter the desired cron schedule.

## Deploying to Google Cloud

To deploy the main FastAPI application, we will use Google Cloud Run.

**1. Build the Docker Image:**

```bash
gcloud builds submit --tag gcr.io/<your-gcp-project-id>/agentic-rag-chatbot
```

*   Replace `<your-gcp-project-id>` with your Google Cloud project ID.

**2. Deploy to Cloud Run:**

```bash
gcloud run deploy agentic-rag-chatbot \ 
    --image gcr.io/<your-gcp-project-id>/agentic-rag-chatbot \ 
    --platform managed \ 
    --region <your-gcp-region> \ 
    --allow-unauthenticated
```

*   Replace `<your-gcp-project-id>` with your Google Cloud project ID.
*   Replace `<your-gcp-region>` with the Google Cloud region where you want to deploy the application.
