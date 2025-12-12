const video = document.getElementById("videoElement");
const captureBtn = document.getElementById("captureBtn");
const usernameInput = document.getElementById("username");
const statusArea = document.getElementById("statusArea");
const captureCanvas = document.getElementById("captureCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const ctxCapture = captureCanvas.getContext("2d");
const ctxOverlay = overlayCanvas.getContext("2d");
const videoContainer = document.querySelector(".fullscreen-video-container");

// Add fullscreen-mode class if mobile
if (window.innerWidth <= 768) {
    document.body.classList.add("fullscreen-mode");
    document.querySelector(".mobile-only").style.display = "block";
    document.querySelector(".desktop-header").style.display = "none";
}

let isProcessing = false;

// Start Webcam
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 },
            },
        });
        video.srcObject = stream;
        startFeedbackLoop();
    } catch (err) {
        console.error("Camera Error:", err);
        statusArea.innerHTML = `<span class="status-error">Camera Access Denied</span>`;
    }
}

startCamera();

function startFeedbackLoop() {
    setInterval(async () => {
        if (isProcessing) return;
        isProcessing = true;

        // Sync Canvas to Container Size (works for both Desktop & Mobile)
        const rect = videoContainer.getBoundingClientRect();
        if (
            overlayCanvas.width !== rect.width ||
            overlayCanvas.height !== rect.height
        ) {
            overlayCanvas.width = rect.width;
            overlayCanvas.height = rect.height;
        }

        // Capture logic
        const sendWidth = 640;
        const scale = sendWidth / video.videoWidth;
        const sendHeight = video.videoHeight * scale;

        captureCanvas.width = sendWidth;
        captureCanvas.height = sendHeight;
        ctxCapture.drawImage(
            video,
            0,
            0,
            captureCanvas.width,
            captureCanvas.height
        );

        const imageData = captureCanvas.toDataURL("image/jpeg", 0.7);

        try {
            const response = await fetch("/api/recognize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: imageData }),
            });
            const data = await response.json();

            if (data.success) {
                const matches = data.matches.map((m) => {
                    return {
                        ...m,
                        box: m.box.map((coord) => coord / scale),
                    };
                });
                drawBoxes(matches);
            }
        } catch (err) {
            // Ignore
        } finally {
            isProcessing = false;
        }
    }, 300);
}

function drawBoxes(matches) {
    ctxOverlay.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Determine Fit Mode: Mobile = Cover, Desktop = Contain
    // We can check the computed style or logic
    const isMobile = window.innerWidth <= 768;
    const fitMode = isMobile ? "cover" : "contain";

    const videoRatio = video.videoWidth / video.videoHeight;
    const containerRatio = overlayCanvas.width / overlayCanvas.height;

    let drawWidth, drawHeight, startX, startY;

    if (fitMode === "cover") {
        if (containerRatio > videoRatio) {
            drawWidth = overlayCanvas.width;
            drawHeight = drawWidth / videoRatio;
            startX = 0;
            startY = (overlayCanvas.height - drawHeight) / 2;
        } else {
            drawHeight = overlayCanvas.height;
            drawWidth = drawHeight * videoRatio;
            startX = (overlayCanvas.width - drawWidth) / 2;
            startY = 0;
        }
    } else {
        // Contain
        if (containerRatio > videoRatio) {
            drawHeight = overlayCanvas.height;
            drawWidth = drawHeight * videoRatio;
            startX = (overlayCanvas.width - drawWidth) / 2;
            startY = 0;
        } else {
            drawWidth = overlayCanvas.width;
            drawHeight = drawWidth / videoRatio;
            startX = 0;
            startY = (overlayCanvas.height - drawHeight) / 2;
        }
    }

    const displayScale = drawWidth / video.videoWidth;

    matches.forEach((match) => {
        const [x1, y1, x2, y2] = match.box;

        const dx = startX + x1 * displayScale;
        const dy = startY + y1 * displayScale;
        const dw = (x2 - x1) * displayScale;
        const dh = (y2 - y1) * displayScale;

        ctxOverlay.strokeStyle = "#00FF00";
        ctxOverlay.lineWidth = 4;
        ctxOverlay.lineJoin = "round";
        ctxOverlay.strokeRect(dx, dy, dw, dh);
    });
}

captureBtn.addEventListener("click", async () => {
    const name = usernameInput.value.trim();
    if (!name) {
        alert("Please enter a name first.");
        return;
    }

    captureBtn.disabled = true;
    captureBtn.innerText = "Processing...";

    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    ctxCapture.drawImage(
        video,
        0,
        0,
        captureCanvas.width,
        captureCanvas.height
    );
    const imageData = captureCanvas.toDataURL("image/jpeg", 0.9);

    try {
        const response = await fetch("/api/register_capture", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: name, image: imageData }),
        });
        const data = await response.json();

        if (data.success) {
            statusArea.innerHTML = `<span class="status-success" style="display:block; text-align:center;">${data.message}</span>`;
            usernameInput.value = "";
        } else {
            statusArea.innerHTML = `<span class="status-error" style="display:block; text-align:center;">${data.message}</span>`;
        }
    } catch (e) {
        console.error(e);
        statusArea.innerHTML = `<span class="status-error">Network Error</span>`;
    } finally {
        captureBtn.disabled = false;
        captureBtn.innerText = "Capture & Register";
    }
});

// Handle Resize
window.addEventListener("resize", () => {
    // Just reload layout logic next frame
    if (window.innerWidth <= 768) {
        document.body.classList.add("fullscreen-mode");
    } else {
        document.body.classList.remove("fullscreen-mode");
    }
});