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
const modalPanel = document.getElementById("modalPanel");
const modalIcon = document.getElementById("modalIcon");
const modalTitle = document.getElementById("modalTitle");
const modalMessage = document.getElementById("modalMessage");
const modalButton = document.getElementById("modalButton");
const videoContainer = document.querySelector(".fullscreen-video-container");
const geofenceMapContainer = document.getElementById("geofenceMapContainer");

// Configuration
const CONFIG = {
    REFRESH_INTERVAL_MS: 500,
    IMAGE_QUALITY: 0.7,
    SEND_WIDTH: 640
};

// Application State
let isProcessingFrame = false;
let isSuccessModalOpen = false;
let currentModalUser = null;
let framesWithoutUser = 0;
const MAX_MISSING_FRAMES = 4; // Approx 2 seconds grace period
let acknowledgedUsers = new Map(); // Name -> framesMissingCount

// Geolocation tracking
let currentCoords = { latitude: null, longitude: null, accuracy: null };

if (navigator.geolocation) {
    navigator.geolocation.watchPosition(
        (position) => {
            currentCoords.latitude = position.coords.latitude;
            currentCoords.longitude = position.coords.longitude;
            currentCoords.accuracy = position.coords.accuracy;
            console.log("Location updated:", currentCoords);
        },
        (error) => {
            console.warn("Geolocation Error:", error.message);
            statusLabel.innerText = "● Location Error: " + error.message;
            statusLabel.style.color = "var(--danger)";
        },
        { enableHighAccuracy: true }
    );
} else {
    statusLabel.innerText = "● Geolocation not supported";
    statusLabel.style.color = "var(--danger)";
}

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

/**
 * Renders a mini SVG map showing the school and user's relative position.
 * Includes dynamic scaling to ensure both are always in view.
 */
function renderGeofenceMap(userLat, userLon, userAcc, schoolLat, schoolLon, schoolRadius) {
    if (!userLat || !userLon || !schoolLat || !schoolLon) return;

    const size = 220; // Slightly larger
    const center = size / 2;
    const padding = 20;

    // Haversine distance
    const R = 6371000;
    const phi1 = userLat * Math.PI / 180;
    const phi2 = schoolLat * Math.PI / 180;
    const dphi = (schoolLat - userLat) * Math.PI / 180;
    const dlambda = (schoolLon - userLon) * Math.PI / 180;
    const a = Math.sin(dphi / 2) ** 2 + Math.cos(phi1) * Math.cos(phi2) * Math.sin(dlambda / 2) ** 2;
    const distanceBody = 2 * R * Math.asin(Math.sqrt(a));

    // Scaling logic
    const maxMeters = Math.max(schoolRadius, distanceBody + (userAcc || 0)) * 1.3;
    const scale = (size / 2 - padding) / maxMeters;

    // Relative offsets
    const dx = (userLon - schoolLon) * 111320 * Math.cos(schoolLat * Math.PI / 180) * scale;
    const dy = (schoolLat - userLat) * 110540 * scale;

    const svg = `
        <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" style="background: #0a0a0a; border-radius: 50%; border: 2px solid rgba(255,255,255,0.1); box-shadow: inset 0 0 20px rgba(0,255,127,0.1);">
            <defs>
                <radialGradient id="gradSchool" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#00FF7F" stop-opacity="0.2" />
                    <stop offset="100%" stop-color="#00FF7F" stop-opacity="0" />
                </radialGradient>
                <radialGradient id="gradUser" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#FF4D4D" stop-opacity="0.3" />
                    <stop offset="100%" stop-color="#FF4D4D" stop-opacity="0" />
                </radialGradient>
            </defs>

            <!-- Radar Grid -->
            <circle cx="${center}" cy="${center}" r="${(size / 2 - padding) * 0.33}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1" />
            <circle cx="${center}" cy="${center}" r="${(size / 2 - padding) * 0.66}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1" />
            <line x1="${center}" y1="${padding}" x2="${center}" y2="${size - padding}" stroke="rgba(255,255,255,0.05)" stroke-width="1" />
            <line x1="${padding}" y1="${center}" x2="${size - padding}" y2="${center}" stroke="rgba(255,255,255,0.05)" stroke-width="1" />

            <!-- SCHOOL: Geofence Zone -->
            <circle cx="${center}" cy="${center}" r="${schoolRadius * scale}" fill="url(#gradSchool)" stroke="#00FF7F" stroke-width="1.5" stroke-dasharray="3,3" />
            
            <!-- SCHOOL: Marker -->
            <circle cx="${center}" cy="${center}" r="4" fill="#00FF7F">
                <animate attributeName="opacity" values="1;0.5;1" dur="3s" repeatCount="indefinite" />
            </circle>
            <text x="${center}" y="${center - 12}" fill="#00FF7F" font-size="10" font-weight="900" text-anchor="middle" style="text-shadow: 0 0 5px rgba(0,255,127,0.5);">SCHOOL</text>

            <!-- USER: Accuracy Zone -->
            <circle cx="${center + dx}" cy="${center + dy}" r="${userAcc * scale}" fill="url(#gradUser)" stroke="#FF4D4D" stroke-width="1" stroke-dasharray="2,2" />
            
            <!-- USER: Marker -->
            <circle cx="${center + dx}" cy="${center + dy}" r="6" fill="#FF4D4D">
                <animate attributeName="r" values="5;7;5" dur="1.5s" repeatCount="indefinite" />
            </circle>
            
            <!-- Dynamic Label Positioning to prevent clipping -->
            <text x="${center + dx}" y="${dy < 0 ? center + dy + 22 : center + dy - 18}" 
                  fill="#FF4D4D" font-size="10" font-weight="900" text-anchor="middle" 
                  style="text-shadow: 0 0 5px rgba(255,77,77,0.5);">
                YOU
            </text>
            
            <!-- Stats Footer -->
            <rect x="0" y="${size - 25}" width="${size}" height="25" fill="rgba(0,0,0,0.6)" />
            <text x="${center}" y="${size - 10}" fill="rgba(255,255,255,0.6)" font-size="9" font-weight="600" text-anchor="middle">
                ${Math.round(distanceBody)}m Away
            </text>


        </svg>
    `;

    geofenceMapContainer.innerHTML = svg;
    geofenceMapContainer.style.display = "block";
}

/**
 * Renders an accuracy visualization comparing current vs required accuracy.
 */
function renderAccuracyGauge(current, required) {
    const size = 200;
    const center = size / 2;
    const radius = 60;

    // Calculate percentage (clamped 0-1) 
    const ratio = Math.min(required / current, 1.0);
    const angle = ratio * Math.PI; // Half circle

    const svg = `
        <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
            <path d="M 40 ${center} A ${radius} ${radius} 0 0 1 160 ${center}" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="12" stroke-linecap="round" />
            <path d="M 40 ${center} A ${radius} ${radius} 0 0 1 ${center + radius * Math.cos(Math.PI + angle)} ${center + radius * Math.sin(Math.PI + angle)}" 
                  fill="none" stroke="${ratio > 0.8 ? '#00FF7F' : '#FFD700'}" stroke-width="12" stroke-linecap="round" />
            <text x="${center}" y="${center + 25}" fill="white" font-size="14" font-weight="bold" text-anchor="middle">${Math.round(current)}m</text>
            <text x="${center}" y="${center + 45}" fill="rgba(255,255,255,0.5)" font-size="10" text-anchor="middle">Current Accuracy</text>
            <text x="${center}" y="${center - 15}" fill="rgba(255,255,255,0.4)" font-size="9" text-anchor="middle">Required: < ${required}m</text>
            <g transform="translate(${center - 15}, ${center - 50})">
                <rect x="0" y="10" width="4" height="4" fill="${ratio > 0.2 ? '#FFD700' : 'rgba(255,255,255,0.1)'}" />
                <rect x="6" y="7" width="4" height="7" fill="${ratio > 0.5 ? '#FFD700' : 'rgba(255,255,255,0.1)'}" />
                <rect x="12" y="4" width="4" height="10" fill="${ratio > 0.8 ? '#00FF7F' : 'rgba(255,255,255,0.1)'}" />
                <rect x="18" y="0" width="4" height="14" fill="${ratio >= 1.0 ? '#00FF7F' : 'rgba(255,255,255,0.1)'}" />
            </g>
        </svg>
    `;

    geofenceMapContainer.innerHTML = svg;
    geofenceMapContainer.style.display = "block";
}

function openModal(title, message, colorVar, relatedUser = null, visualData = null) {
    // If modal is already open for THIS user, just return (avoid flicker)
    if (isSuccessModalOpen && currentModalUser === relatedUser && relatedUser !== null) {
        return;
    }

    // If user has acknowledged this recently, do not show
    if (relatedUser && acknowledgedUsers.has(relatedUser)) {
        return;
    }

    isSuccessModalOpen = true;
    currentModalUser = relatedUser;
    framesWithoutUser = 0;

    // Update Content
    modalTitle.innerText = title;
    modalMessage.innerText = message;

    // Handle Visuals
    if (visualData) {
        if (visualData.type === 'geofence') {
            renderGeofenceMap(
                visualData.userLat,
                visualData.userLon,
                visualData.userAcc,
                visualData.schoolLat,
                visualData.schoolLon,
                visualData.schoolRadius
            );
        } else if (visualData.type === 'accuracy') {
            renderAccuracyGauge(visualData.current, visualData.required);
        }
    } else {
        geofenceMapContainer.style.display = "none";
        geofenceMapContainer.innerHTML = "";
    }

    // Update Colors
    // colorVar should be like "var(--success)" or "var(--warning)" or hex
    modalPanel.style.borderColor = colorVar;
    modalPanel.style.boxShadow = `0 0 50px ${colorVar === 'var(--success)' ? 'rgba(0, 255, 127, 0.2)' : 'rgba(255, 165, 0, 0.2)'}`; // Approximate
    modalIcon.style.backgroundColor = colorVar;
    modalTitle.style.color = colorVar;
    modalButton.style.backgroundColor = colorVar;

    // Ensure high contrast for text/icon inside colored elements
    modalButton.style.color = "#FFD700";
    modalButton.style.fontWeight = "bold";
    const iconSvg = modalIcon.querySelector('svg');
    if (iconSvg) iconSvg.style.stroke = "#FFD700";

    successModal.style.display = "flex";
}

function closeSuccessModal(isManual = false) {
    if (isManual && currentModalUser) {
        acknowledgedUsers.set(currentModalUser, 0); // Start tracking absence for this user
    }
    successModal.style.display = "none";
    isSuccessModalOpen = false;
    currentModalUser = null;
    framesWithoutUser = 0;
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

        // NOTE: Client should not be the source of truth for whether a user was
        // "marked" — the server returns `newly_marked` in the `matches` payload
        // and we open the modal centrally from the recognition loop to ensure
        // consistent messaging. We no longer open the modal from here.

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
        // Skip if busy
        if (isProcessingFrame) return;
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
                body: JSON.stringify({
                    image: imageData,
                    latitude: currentCoords.latitude,
                    longitude: currentCoords.longitude,
                    accuracy: currentCoords.accuracy
                }),
            });

            const data = await response.json();

            console.log(data)

            // 4. Update UI based on response
            if (data.attendance_error) {
                statusLabel.innerText = `● ${data.attendance_error}`;
                statusLabel.style.color = "var(--primary)"; // Info style

                // Check for specific "Already marked" error to show modal
                if (data.attendance_error.includes("already marked")) {
                    // Try to identify the user who is already marked to track them
                    // We assume the first detected known user is the one triggered this
                    let likelyUser = null;
                    if (data.matches && data.matches.length > 0) {
                        const known = data.matches.find(m => m.name !== "Unknown" && !m.name.startsWith("Unknown"));
                        if (known) likelyUser = known.name;
                    }

                    openModal("Notice", "Already marked present today.", "#FFD700", likelyUser); // Gold/Yellow
                    // Or use a hex that matches a warning style. 
                } else if (data.attendance_error.includes("outside school")) {
                    let likelyUser = null;
                    if (data.matches && data.matches.length > 0) {
                        const known = data.matches.find(m => m.name !== "Unknown" && !m.name.startsWith("Unknown"));
                        if (known) likelyUser = known.name;
                    }

                    openModal("Outside Boundary", "You must be inside school to mark attendance.", "var(--danger)", likelyUser, {
                        type: 'geofence',
                        userLat: currentCoords.latitude,
                        userLon: currentCoords.longitude,
                        userAcc: currentCoords.accuracy,
                        schoolLat: data.school_lat,
                        schoolLon: data.school_lon,
                        schoolRadius: data.school_radius
                    });
                } else if (data.attendance_error.includes("accuracy too low")) {
                    openModal("Low GPS Accuracy", "Your GPS signal is too weak. Try moving to an open area.", "var(--warning)", null, {
                        type: 'accuracy',
                        current: data.current_accuracy,
                        required: data.max_accuracy
                    });
                } else if (data.attendance_error.includes("Success")) {
                    let likelyUser = null;
                    if (data.matches && data.matches.length > 0) {
                        const known = data.matches.find(m => m.name !== "Unknown" && !m.name.startsWith("Unknown"));
                        if (known) likelyUser = known.name;
                    }
                    if (likelyUser) {
                        openModal("Marked!", "You are marked present today.", "var(--success)", likelyUser);
                    }
                }
            } else {
                // Show a "Marked" modal only if the server explicitly flagged
                // the match as newly marked (m.newly_marked === true).
                // let newlyMarked = null;
                // if (data.matches && data.matches.length > 0) {
                //     newlyMarked = data.matches.find(m => m.newly_marked === true);
                //     console.log(newlyMarked)
                // }
                // if (newlyMarked) {
                //     // Use a consistent user-facing message and include the name
                //     // for clarity in multi-face scenarios.
                //     console.log("Marked!", `${newlyMarked.name} is marked present today.`, "var(--success)", newlyMarked.name)
                //     openModal("Marked!", `${newlyMarked.name} is marked present today.`, "var(--success)", newlyMarked.name);
                // }
                let likelyUser = null;
                if (data.matches && data.matches.length > 0) {
                    const known = data.matches.find(m => m.name !== "Unknown" && !m.name.startsWith("Unknown"));
                    if (known) likelyUser = known.name;
                }
                if (likelyUser) {
                    openModal("Marked!", "You are marked present today.", "var(--success)", likelyUser);
                }
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

                // --- Auto Disappear Logic ---
                if (isSuccessModalOpen && currentModalUser) {
                    // Check if currentModalUser is still in frame (matches)
                    const isUserPresent = matches.some(m => m.name === currentModalUser);

                    if (isUserPresent) {
                        framesWithoutUser = 0; // Reset counter if seen
                    } else {
                        framesWithoutUser++;
                        // If user is missing for too many consecutive frames, close modal
                        if (framesWithoutUser > MAX_MISSING_FRAMES) {
                            closeSuccessModal(false); // Auto-close is NOT manual
                        }
                    }
                }

                // --- Acknowledged User Cleanup Logic ---
                // For anyone in the acknowledged Set, if they are NOT in matches, increment their missing count.
                // If missing count > MAX, remove them (so they can be warned again if they return).
                acknowledgedUsers.forEach((missingCount, user) => {
                    const isUserPresent = matches.some(m => m.name === user);
                    if (isUserPresent) {
                        acknowledgedUsers.set(user, 0); // Reset if seen
                    } else {
                        const newCount = missingCount + 1;
                        if (newCount > MAX_MISSING_FRAMES) {
                            acknowledgedUsers.delete(user);
                        } else {
                            acknowledgedUsers.set(user, newCount);
                        }
                    }
                });

            } else if (isSuccessModalOpen && currentModalUser) {
                // No matches found at all (or success=false), but modal is open with a user
                framesWithoutUser++;
                if (framesWithoutUser > MAX_MISSING_FRAMES) {
                    closeSuccessModal(false);
                }

                // Also increment for all acknowledged users since no one is matched
                acknowledgedUsers.forEach((missingCount, user) => {
                    const newCount = missingCount + 1;
                    if (newCount > MAX_MISSING_FRAMES) {
                        acknowledgedUsers.delete(user);
                    } else {
                        acknowledgedUsers.set(user, newCount);
                    }
                });
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