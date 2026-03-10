# Devcontainers are essential for Vibecoding

Containers isolate the environment and make it safer to run the agents even in YOLO mode.


This devcontainer.json will install Gemini CLI on top of the Playwright container image provided by Microsoft.

```json
{
  "name": "Gemini Playwright Sandbox",
  // Use the official Playwright image. It includes Node.js and all browser binaries.
  "image": "mcr.microsoft.com/playwright:v1.58.2-jammy",

  "customizations": {
    "vscode": {
      "settings": {
        "terminal.integrated.defaultProfile.linux": "bash"
      },
      "extensions": [
        "google.geminicode" // Optional: Google's AI assistant extension
      ]
    }
  },

  // Install the Gemini CLI automatically when the container builds
  "postCreateCommand": "npm install -g @google/gemini-cli && echo 'Gemini CLI installed!'"
}
```

Trying to Add quarto and R and other dependencies into the container. This will break the Gemini CLI

```json
{
  "name": "Quarto R Python DevContainer",
  "image": "mcr.microsoft.com/devcontainers/base:ubuntu-22.04",
  "features": {
    "ghcr.io/rocker-org/devcontainer-features/r-apt:latest": {
      "installRStudio": false
    },
    "ghcr.io/devcontainers/features/python:1": {
      "version": "latest"
    },
    "ghcr.io/rocker-org/devcontainer-features/quarto-cli:1": {
      "version": "latest"
    }
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "REditorSupport.r",
        "ms-python.python",
        "quarto.quarto",
        "ms-toolsai.jupyter"
      ],
      "settings": {
        "r.lsp.enabled": true,
        "python.defaultInterpreterPath": "/usr/local/bin/python"
      }
    }
  },
  "postCreateCommand": "npm install -g @google/gemini-cli && Rscript -e 'install.packages(c(\"languageserver\", \"renv\"), repos=\"https://cloud.r-project.org\")' && if [ -f renv.lock ]; then Rscript -e 'renv::restore()'; fi && pip install jupyter ipykernel"
}
```

Attempt to install everything together

```json
{
  "name": "Gemini Playwright Sandbox",
  // Use the official Playwright image. It includes Node.js and all browser binaries.
  "image": "mcr.microsoft.com/playwright:v1.58.2-jammy",
  "features": {
    "ghcr.io/rocker-org/devcontainer-features/r-apt:latest": {
      "installRStudio": false
    },
    "ghcr.io/devcontainers/features/python:1": {
      "version": "latest"
    },
    "ghcr.io/rocker-org/devcontainer-features/quarto-cli:1": {
      "version": "latest"
    }
  },
  "customizations": {
    "vscode": {
      "settings": {
        "terminal.integrated.defaultProfile.linux": "bash",
        "r.lsp.enabled": true,
        "python.defaultInterpreterPath": "/usr/local/bin/python"
      },
      "extensions": [
        "google.geminicode", // Optional: Google's AI assistant extension
        "REditorSupport.r",
        "ms-python.python",
        "quarto.quarto",
        "ms-toolsai.jupyter"
      ]
    }
  },

  // Install the Gemini CLI automatically when the container builds
  "postCreateCommand": "npm install -g @google/gemini-cli && echo 'Gemini CLI installed!' && Rscript -e 'install.packages(c(\"languageserver\", \"renv\"), repos=\"https://cloud.r-project.org\")' && if [ -f renv.lock ]; then Rscript -e 'renv::restore()'; fi && pip install jupyter ipykernel"
}
```