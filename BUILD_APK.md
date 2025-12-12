# How to Build the Android APK

Since the Android SDK is required to compile APKs and it is very large (2GB+), the best way to generate an APK for this project is using **GitHub Actions** (Cloud Build).

## 1. Cloud Build (Recommended)

I have included a GitHub Action workflow in `.github/workflows/build_apk.yml`.

### Steps:

1.  **Push to GitHub**: Push this code to a GitHub repository.
2.  **Edit Manifest**: ensure `static/manifest.json` points to your REAL server IP, not `localhost`.
    _Note: Since your IP changes, you might want to use a service like `ngrok` to get a stable public URL, and put THAT in the manifest._
3.  **Wait**: Go to the "Actions" tab in your repository.
4.  **Download**: Click on the latest workflow run. The `.apk` file will be available under "Artifacts" at the bottom.

## 2. Local Build (Advanced)

If you have the Android SDK installed:

1.  Install Bubblewrap:
    ```bash
    npm install -g @bubblewrap/cli
    ```
2.  Initialize:
    ```bash
    bubblewrap init --manifest static/manifest.json
    ```
3.  Build:
    ```bash
    bubblewrap build
    ```

## Critical Note on IP Addresses

An APK installed on your phone cannot access `http://localhost`.
You MUST ensure your phone can reach the server.

- **Local Wi-Fi**: Use your PC's IP (e.g., `http://192.168.1.15:5000`).
- **Internet**: Use a tunnel like `ngrok http 5000`.
