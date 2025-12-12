/**
 * recognize.js
 * Handles the webcam feed, frame capture, and communication with the backend
 * for face recognition and attendance marking.
 */

const videoElement = document.getElementById("videoElement");
const overlayCanvas = document.getElementById("overlayCanvas");
const captureCanvas = document.getElementById("captureCanvas");
const overlayContext = overlayCanvas.getContext("2d");
const captureContext = captureCanvas.getContext("2d");

const statusLabel = document.getElementById("systemStatus");
const successModal = document.getElementById("successModal");
const modalNameElement = document.getElementById("modalName");
const videoContainer = document.querySelector(".fullscreen-video-container");

// Configuration
const CONFIG = {
    REFRESH_INTERVAL_MS: 500,
    IMAGE_QUALITY: 0.7,
    SEND_WIDTH: 640
};

// Application State
let isProcessingFrame = false;
let isSuccessModalOpen = false;

// --- Initialization ---

// Mobile responsive adjustment
if (window.innerWidth <= 768) {
    document.body.classList.add("fullscreen-mode");
    const desktopHeader = document.querySelector(".desktop-header");
    if (desktopHeader) desktopHeader.style.display = "none";
}

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
        videoElement.srcObject = stream;
        statusLabel.innerText = "● System Active";
        startRecognitionLoop();
    } catch (err) {
        console.error("Camera Error:", err);
        statusLabel.innerText = "● Camera Error";
        statusLabel.style.color = "var(--danger)";
    }
}

startCamera();

// --- UI Helpers ---

function openSuccessModal(name) {
    isSuccessModalOpen = true;
    modalNameElement.innerText = name;
    successModal.style.display = "flex";
}

function closeSuccessModal() {
    successModal.style.display = "none";
    isSuccessModalOpen = false;
}

// Make globally valid for the onclick in HTML
window.closeModal = closeSuccessModal;

/**
 * Draws bounding boxes and labels on the overlay canvas.
 * Handles responsive scaling to ensure boxes align with the video feed.
 * @param {Array} matches - Array of recognition matches.
 */
function drawRecognitionBoxes(matches) {
    overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Responsive Fit Logic: Calculate how the video is displayed
    const isMobile = window.innerWidth <= 768;
    const fitMode = isMobile ? "cover" : "contain";

    const videoRatio = videoElement.videoWidth / videoElement.videoHeight;
    const containerRatio = overlayCanvas.width / overlayCanvas.height;

    let drawWidth, drawHeight, startX, startY;

    if (fitMode === "cover") {
        if (containerRatio > videoRatio) {
            // Container is wider than video -> Crop height
            drawWidth = overlayCanvas.width;
            drawHeight = drawWidth / videoRatio;
            startX = 0;
            startY = (overlayCanvas.height - drawHeight) / 2;
        } else {
            // Container is taller than video -> Crop width
            drawHeight = overlayCanvas.height;
            drawWidth = drawHeight * videoRatio;
            startX = (overlayCanvas.width - drawWidth) / 2;
            startY = 0;
        }
    } else {
        // Contain Mode
        if (containerRatio > videoRatio) {
            // Container is wider -> Pillarbox
            drawHeight = overlayCanvas.height;
            drawWidth = drawHeight * videoRatio;
            startX = (overlayCanvas.width - drawWidth) / 2;
            startY = 0;
        } else {
            // Container is taller -> Letterbox
            drawWidth = overlayCanvas.width;
            drawHeight = drawWidth / videoRatio;
            startX = 0;
            startY = (overlayCanvas.height - drawHeight) / 2;
        }
    }

    const displayScale = drawWidth / videoElement.videoWidth;

    matches.forEach((match) => {
        const [x1, y1, x2, y2] = match.box;
        const name = match.name;
        // const similarity = match.similarity; // Unused for display but available

        // Trigger Modal if newly marked
        if (match.newly_marked && !isSuccessModalOpen) {
            openSuccessModal(name);
        }

        // Transform coordinates to display space
        const dx = startX + x1 * displayScale;
        const dy = startY + y1 * displayScale;
        const dw = (x2 - x1) * displayScale;
        const dh = (y2 - y1) * displayScale;

        // Styling
        let color = "var(--danger)"; // Default Unknown Red
        if (name !== "Unknown" && !name.startsWith("Unknown")) {
            color = "var(--success)"; // Known Green
        }
        // Force hex for canvas if needed, or get computed style. 
        // For simplicity using hardcoded hex fallback if var fails parsing by canvas (canvas needs standard colors)
        const uiColor = name !== "Unknown" && !name.startsWith("Unknown") ? "#00FF7F" : "#FF4D4D";

        overlayContext.strokeStyle = uiColor;
        overlayContext.lineWidth = 4;
        overlayContext.lineJoin = "round";
        overlayContext.strokeRect(dx, dy, dw, dh);

        overlayContext.fillStyle = uiColor;
        overlayContext.font = "bold 18px Inter, sans-serif";
        overlayContext.shadowColor = "black";
        overlayContext.shadowBlur = 4;

        // Label positioning
        const textY = dy > 30 ? dy - 10 : dy + 30;
        overlayContext.fillText(name, dx, textY);
    });
}

// --- Main Loop ---

function startRecognitionLoop() {
    setInterval(async () => {
        // Skip if busy or modal is blocking view
        if (isProcessingFrame || isSuccessModalOpen) return;
        isProcessingFrame = true;

        try {
            // 1. Sync Overlay Canvas Size with Container
            const rect = videoContainer.getBoundingClientRect();
            if (overlayCanvas.width !== rect.width || overlayCanvas.height !== rect.height) {
                overlayCanvas.width = rect.width;
                overlayCanvas.height = rect.height;
            }

            // 2. Capture Frame (Downscaled for performance)
            const scale = CONFIG.SEND_WIDTH / videoElement.videoWidth;
            const sendHeight = videoElement.videoHeight * scale;

            captureCanvas.width = CONFIG.SEND_WIDTH;
            captureCanvas.height = sendHeight;

            if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
                captureContext.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
            } else {
                return; // Video not ready
            }

            const imageData = captureCanvas.toDataURL("image/jpeg", CONFIG.IMAGE_QUALITY);

            // 3. Send to API
            const response = await fetch("/api/recognize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: imageData }),
            });

            const data = await response.json();

            // 4. Update UI based on response
            if (data.attendance_error) {
                statusLabel.innerText = `● ${data.attendance_error}`;
                statusLabel.style.color = "var(--danger)";
            } else {
                statusLabel.innerText = "● System Active";
                statusLabel.style.color = "var(--success)";
            }

            if (data.success && data.matches) {
                // Upscale boxes back to video resolution for drawing logic
                const matches = data.matches.map((m) => {
                    return {
                        ...m,
                        box: m.box.map((coord) => coord / scale),
                    };
                });
                drawRecognitionBoxes(matches);
            }

        } catch (err) {
            console.error("Recognition Loop Error:", err);
            statusLabel.innerText = "● Recognition Error";
            statusLabel.style.color = "var(--danger)";
        } finally {
            isProcessingFrame = false;
        }
    }, CONFIG.REFRESH_INTERVAL_MS);
}

// --- Event Listeners ---

window.addEventListener("resize", () => {
    if (window.innerWidth <= 768) {
        document.body.classList.add("fullscreen-mode");
    } else {
        document.body.classList.remove("fullscreen-mode");
    }
});