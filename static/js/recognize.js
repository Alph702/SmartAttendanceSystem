const video = document.getElementById("videoElement");
const overlayCanvas = document.getElementById("overlayCanvas");
const captureCanvas = document.getElementById("captureCanvas");
const ctxOverlay = overlayCanvas.getContext("2d");
const ctxCapture = captureCanvas.getContext("2d");
const statusLabel = document.getElementById("systemStatus");
const successModal = document.getElementById("successModal");
const modalName = document.getElementById("modalName");
const videoContainer = document.querySelector(".fullscreen-video-container");

// Add fullscreen-mode class if mobile
if (window.innerWidth <= 768) {
    document.body.classList.add("fullscreen-mode");
    document.querySelector(".desktop-header").style.display = "none";
}

let isProcessing = false;
let isModalOpen = false;

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
        statusLabel.innerText = "● System Active";
        startRecognitionLoop();
    } catch (err) {
        console.error("Camera Error:", err);
        statusLabel.innerText = "● Camera Error";
        statusLabel.style.color = "var(--danger)";
    }
}

startCamera();

function showModal(name) {
    isModalOpen = true;
    modalName.innerText = name;
    successModal.style.display = "flex";
}

function closeModal() {
    successModal.style.display = "none";
    isModalOpen = false;
}

function drawBoxes(matches) {
    ctxOverlay.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Responsive Fit Logic
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
        const name = match.name;
        const sim = match.similarity;

        if (match.newly_marked && !isModalOpen) {
            showModal(name);
        }

        const dx = startX + x1 * displayScale;
        const dy = startY + y1 * displayScale;
        const dw = (x2 - x1) * displayScale;
        const dh = (y2 - y1) * displayScale;

        let color = "#FF0000";
        if (name !== "Unknown" && !name.startsWith("Unknown")) {
            color = "#00FF00";
        }

        ctxOverlay.strokeStyle = color;
        ctxOverlay.lineWidth = 4;
        ctxOverlay.lineJoin = "round";
        ctxOverlay.strokeRect(dx, dy, dw, dh);

        ctxOverlay.fillStyle = color;
        ctxOverlay.font = "bold 18px Inter";
        ctxOverlay.shadowColor = "black";
        ctxOverlay.shadowBlur = 4;
        ctxOverlay.fillText(`${name}`, dx, dy > 30 ? dy - 10 : dy + 30);
    });
}

function startRecognitionLoop() {
    setInterval(async () => {
        if (isProcessing || isModalOpen) return;
        isProcessing = true;

        // Sync Canvas to Container Size
        const rect = videoContainer.getBoundingClientRect();
        if (
            overlayCanvas.width !== rect.width ||
            overlayCanvas.height !== rect.height
        ) {
            overlayCanvas.width = rect.width;
            overlayCanvas.height = rect.height;
        }

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
            statusLabel.innerText = "● System Active";
            statusLabel.style.color = "var(--success)";
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
            console.error("Recognition Error:", err);
            statusLabel.innerText = "● Recognition Error";
            statusLabel.style.color = "var(--danger)";
        } finally {
            isProcessing = false;
        }
    }, 500);
}

window.addEventListener("resize", () => {
    if (window.innerWidth <= 768) {
        document.body.classList.add("fullscreen-mode");
    } else {
        document.body.classList.remove("fullscreen-mode");
    }
});