const API_BASE = "http://127.0.0.1:8000";
window.__YOLO_FE_APP_LOADED__ = true;
const SUMMARY_THRESHOLD = 20;
const CAMERA_INTERVAL_MS = 700;

let activeMode = "image";
let selectedImageFile = null;
let imageObjectUrl = null;

let overlayAnimationId = null;

let cameraStream = null;
let cameraTimer = null;
let cameraRequestInFlight = false;
let cameraFrameCount = 0;
let cameraClassCounts = {};
let currentCameraObjects = [];

document.addEventListener("DOMContentLoaded", init);
document.addEventListener("submit", event => event.preventDefault());

function init() {
    byId("imageInput").addEventListener("change", onImageSelected);
    byId("detectBtn").addEventListener("click", runDetection);

    document.querySelectorAll("input[name='sourceMode']").forEach(input => {
        input.addEventListener("change", () => switchMode(input.value));
    });

    switchMode("image");
    loadModelStatus();
}

function byId(id) {
    return document.getElementById(id);
}

function setStatus(text) {
    byId("runStatus").textContent = text;
}

async function loadModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/model/status`);
        renderModelStatus(await response.json());
    } catch (error) {
        byId("modelStatus").textContent = `Không kết nối được backend: ${error.message}`;
    }
}

function renderModelStatus(data) {
    const target = byId("modelStatus");
    if (!data || data.success === false) {
        target.textContent = `Backend/model lỗi: ${data?.error || "unknown"}`;
        return;
    }

    const detector = data.detector || {};
    const pose = data.pose || {};
    target.innerHTML = [
        `<strong>Detection:</strong> ${escapeHtml(detector.type || "unknown")} YOLO`,
        `<span>${escapeHtml(detector.path || "")}</span>`,
        `<strong>Pose:</strong> ${pose.enabled ? "YOLO pose enabled" : "pose unavailable"}`,
        `<span>${escapeHtml(pose.note || "")}</span>`,
    ].join(" ");
}

function switchMode(mode) {
    activeMode = mode;
    stopAllRealtimeWork();
    clearCanvas();
    renderAppearanceSummary([]);
    renderObjectDetails([]);
    setStatus("Sẵn sàng");

    const imageInput = byId("imageInput");
    const image = byId("previewImage");
    const video = byId("previewVideo");

    imageInput.hidden = mode !== "image";
    image.hidden = mode !== "image" || !imageObjectUrl;
    video.hidden = mode !== "camera";
    video.controls = false;
    byId("mediaTitle").textContent = mode === "image" ? "Ảnh phát hiện" : "Camera phát hiện";

    if (mode === "image") {
        stopCamera();
        video.pause();
        video.removeAttribute("src");
        video.load();
        if (imageObjectUrl) {
            image.src = imageObjectUrl;
            image.hidden = false;
        }
    }

    if (mode === "camera") {
        image.hidden = true;
        startCamera();
    }
}

function onImageSelected() {
    const input = byId("imageInput");
    const image = byId("previewImage");

    stopAllRealtimeWork();
    clearCanvas();
    renderAppearanceSummary([]);
    renderObjectDetails([]);

    selectedImageFile = input.files?.[0] || null;
    if (!selectedImageFile) {
        image.removeAttribute("src");
        setStatus("Chưa chọn ảnh");
        return;
    }

    if (imageObjectUrl) URL.revokeObjectURL(imageObjectUrl);
    imageObjectUrl = URL.createObjectURL(selectedImageFile);
    image.src = imageObjectUrl;
    image.hidden = false;
    setStatus("Đã chọn ảnh");
}

async function runDetection(event) {
    event?.preventDefault();
    if (activeMode === "image") await detectImage();
    if (activeMode === "camera") startCameraDetection();
}

async function detectImage() {
    if (!selectedImageFile || !imageObjectUrl) {
        alert("Vui lòng chọn ảnh");
        return;
    }

    const button = byId("detectBtn");
    const image = byId("previewImage");
    const canvas = byId("detectionCanvas");

    stopAllRealtimeWork();
    clearCanvas();
    image.hidden = false;
    image.src = imageObjectUrl;
    setBusy(button, true);
    setStatus("Đang detect ảnh...");

    try {
        await waitForImageReady(image, 5000);
        const data = await postImageFile(selectedImageFile);
        const objects = normalizeObjects(data.objects || data.detections || []);
        drawOverlay(objects, image, canvas);
        renderAppearanceSummary(summaryFromDetections(objects));
        renderObjectDetails(objects);
        applyModelStatus(data);
        setStatus(`Detect xong: ${objects.length} vật thể`);
    } catch (error) {
        clearCanvas();
        setStatus("Detect ảnh lỗi");
        alert(`Lỗi detection: ${error.message}`);
    } finally {
        setBusy(button, false);
    }
}

async function startCamera() {
    const video = byId("previewVideo");
    stopCamera();
    clearCanvas();

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false,
        });
        video.srcObject = cameraStream;
        video.hidden = false;
        video.controls = false;
        video.muted = true;
        await video.play();
        setStatus("Camera đang mở");
    } catch (error) {
        setStatus("Không mở được camera");
        alert(`Không mở được camera: ${error.message}`);
    }
}

function startCameraDetection() {
    if (!cameraStream) {
        startCamera();
        return;
    }

    stopOverlayLoop();
    clearCanvas();
    cameraClassCounts = {};
    cameraFrameCount = 0;
    currentCameraObjects = [];
    startCameraOverlay(byId("previewVideo"), byId("detectionCanvas"));

    if (cameraTimer) clearInterval(cameraTimer);
    detectCameraFrame();
    cameraTimer = setInterval(detectCameraFrame, CAMERA_INTERVAL_MS);
    setStatus("Đang detect camera...");
}

async function detectCameraFrame() {
    const video = byId("previewVideo");
    if (cameraRequestInFlight || !cameraStream || video.videoWidth <= 0 || video.videoHeight <= 0) return;

    cameraRequestInFlight = true;
    try {
        const frameCanvas = document.createElement("canvas");
        frameCanvas.width = video.videoWidth;
        frameCanvas.height = video.videoHeight;
        frameCanvas.getContext("2d").drawImage(video, 0, 0);
        const blob = await new Promise(resolve => frameCanvas.toBlob(resolve, "image/jpeg", 0.86));
        if (!blob) return;

        const data = await postImageFile(blob, "camera-frame.jpg");
        currentCameraObjects = normalizeObjects(data.objects || data.detections || []);
        cameraFrameCount += 1;

        new Set(currentCameraObjects.map(item => item.class_name).filter(Boolean)).forEach(className => {
            cameraClassCounts[className] = (cameraClassCounts[className] || 0) + 1;
        });

        renderAppearanceSummary(summaryFromCounts(cameraClassCounts, cameraFrameCount));
        renderObjectDetails(currentCameraObjects);
        applyModelStatus(data);
    } catch (error) {
        console.error("Camera detection error", error);
    } finally {
        cameraRequestInFlight = false;
    }
}

async function postImageFile(file, filename) {
    const formData = new FormData();
    formData.append("file", file, filename || file.name || "image.jpg");
    const response = await fetch(`${API_BASE}/detect/image`, {
        method: "POST",
        body: formData,
    });
    const data = await response.json();
    if (!response.ok || data.success === false || data.error) {
        throw new Error(data.error || `Backend lỗi ${response.status}`);
    }
    return data;
}

function startCameraOverlay(video, canvas) {
    stopOverlayLoop();
    const ctx = canvas.getContext("2d");

    function draw() {
        if (!cameraStream) return;
        overlayAnimationId = requestAnimationFrame(draw);
        resizeCanvasToStage(canvas);
        drawOverlay(currentCameraObjects, video, canvas, ctx);
    }

    draw();
}

function drawOverlay(objects, media, canvas, ctx) {
    ctx = ctx || canvas.getContext("2d");
    resizeCanvasToStage(canvas);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const naturalWidth = media.videoWidth || media.naturalWidth;
    const naturalHeight = media.videoHeight || media.naturalHeight;
    const drawable = (objects || []).filter(item => item.bbox_2d || bboxDictToArray(item.bbox));

    if (!drawable.length || !naturalWidth || !naturalHeight) {
        canvas.style.display = "none";
        return;
    }

    canvas.style.display = "block";
    const placement = fitContain(naturalWidth, naturalHeight, canvas.width, canvas.height);
    const colors = ["#00ff66", "#ffcc00", "#00b7ff", "#ff4d4d", "#bbff00", "#ff9f1c"];

    drawable.forEach((object, index) => {
        const bbox = object.bbox_2d || bboxDictToArray(object.bbox);
        if (!bbox) return;

        const x = placement.offsetX + bbox[0] * placement.scaleX;
        const y = placement.offsetY + bbox[1] * placement.scaleY;
        const width = (bbox[2] - bbox[0]) * placement.scaleX;
        const height = (bbox[3] - bbox[1]) * placement.scaleY;
        if (width <= 0 || height <= 0) return;

        const color = object.source === "pose" ? "#ffb000" : colors[index % colors.length];
        const confidence = Number(object.confidence);
        const label = `${object.object_id ? `#${object.object_id} ` : ""}${object.class_name || "object"} ${Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : ""}`;

        ctx.lineWidth = 3;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.font = "bold 15px Arial";
        ctx.strokeRect(x, y, width, height);

        const labelHeight = 23;
        const labelWidth = Math.min(ctx.measureText(label).width + 12, canvas.width - x);
        const labelY = y - labelHeight < 0 ? y : y - labelHeight;
        ctx.fillRect(x, labelY, labelWidth, labelHeight);
        ctx.fillStyle = "#000";
        ctx.fillText(label, x + 6, labelY + 16);

        drawKeypoints(ctx, object, placement);
    });
}

function drawKeypoints(ctx, object, placement) {
    const keypoints = object.keypoints_2d || [];
    const skeleton = object.skeleton || [];
    if (!keypoints.length) return;

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#00e5ff";
    skeleton.forEach(([start, end]) => {
        const a = keypoints[start];
        const b = keypoints[end];
        if (!a || !b || a[2] <= 0.2 || b[2] <= 0.2) return;
        ctx.beginPath();
        ctx.moveTo(placement.offsetX + a[0] * placement.scaleX, placement.offsetY + a[1] * placement.scaleY);
        ctx.lineTo(placement.offsetX + b[0] * placement.scaleX, placement.offsetY + b[1] * placement.scaleY);
        ctx.stroke();
    });

    ctx.fillStyle = "#00e5ff";
    keypoints.forEach(point => {
        if (!point || point[2] <= 0.2) return;
        ctx.beginPath();
        ctx.arc(placement.offsetX + point[0] * placement.scaleX, placement.offsetY + point[1] * placement.scaleY, 3, 0, Math.PI * 2);
        ctx.fill();
    });
}

function resizeCanvasToStage(canvas) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
}

function fitContain(naturalWidth, naturalHeight, boxWidth, boxHeight) {
    const mediaRatio = naturalWidth / naturalHeight;
    const boxRatio = boxWidth / boxHeight;
    let drawWidth;
    let drawHeight;
    let offsetX = 0;
    let offsetY = 0;

    if (boxRatio > mediaRatio) {
        drawHeight = boxHeight;
        drawWidth = drawHeight * mediaRatio;
        offsetX = (boxWidth - drawWidth) / 2;
    } else {
        drawWidth = boxWidth;
        drawHeight = drawWidth / mediaRatio;
        offsetY = (boxHeight - drawHeight) / 2;
    }

    return {
        offsetX,
        offsetY,
        scaleX: drawWidth / naturalWidth,
        scaleY: drawHeight / naturalHeight,
    };
}

function stopAllRealtimeWork() {
    stopOverlayLoop();
    stopCameraDetection();
}

function stopOverlayLoop() {
    if (overlayAnimationId) {
        cancelAnimationFrame(overlayAnimationId);
        overlayAnimationId = null;
    }
}

function stopCameraDetection() {
    if (cameraTimer) {
        clearInterval(cameraTimer);
        cameraTimer = null;
    }
    cameraRequestInFlight = false;
}

function stopCamera() {
    stopCameraDetection();
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    const video = byId("previewVideo");
    if (video.srcObject) video.srcObject = null;
}

function clearCanvas() {
    const canvas = byId("detectionCanvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.display = "none";
}

function setBusy(button, busy) {
    button.disabled = busy;
    button.textContent = busy ? "Detecting..." : "Detection";
}

function applyModelStatus(payload) {
    if (!payload?.model_status) return;
    renderModelStatus({
        success: true,
        detector: payload.model_status.detector,
        pose: payload.model_status.pose,
    });
}

function normalizeObjects(items) {
    return (items || []).map(item => ({
        ...item,
        bbox_2d: item.bbox_2d || bboxDictToArray(item.bbox),
    })).filter(item => item.bbox_2d);
}

function bboxDictToArray(bbox) {
    if (!bbox) return null;
    return [bbox.x1, bbox.y1, bbox.x2, bbox.y2];
}

function summaryFromDetections(objects) {
    return [...new Set((objects || []).map(item => item.class_name).filter(Boolean))]
        .map(className => ({ class_name: className, appearance_percent: 100 }));
}

function summaryFromCounts(counts, total) {
    if (total <= 0) return [];
    return Object.entries(counts)
        .map(([className, count]) => ({
            class_name: className,
            appearance_percent: Math.round(count * 10000 / total) / 100,
        }))
        .sort((a, b) => b.appearance_percent - a.appearance_percent);
}

function renderAppearanceSummary(summary) {
    const rows = (summary || []).filter(item => Number(item.appearance_percent) > SUMMARY_THRESHOLD);
    byId("summaryTotal").textContent = `Tổng số vật thể: ${rows.length}`;

    const tbody = byId("objectTableBody");
    tbody.innerHTML = "";
    if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="3">Chưa có vật thể nào trên ${SUMMARY_THRESHOLD}%</td></tr>`;
        return;
    }

    rows.forEach((item, index) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${index + 1}</td>
            <td>${escapeHtml(item.class_name || "object")}</td>
            <td>${formatNumber(item.appearance_percent)}%</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderObjectDetails(objects) {
    const target = byId("objectDetails");
    if (!objects || !objects.length) {
        target.innerHTML = "<p>Chưa có chi tiết vật thể.</p>";
        return;
    }

    target.innerHTML = objects.slice(0, 10).map(object => {
        const position = object.position_3d || {};
        const keypointCount = (object.keypoints_2d || []).filter(point => point?.[2] > 0.2).length;
        return `
            <div class="object-detail">
                <strong>${object.object_id ? `#${escapeHtml(object.object_id)} ` : ""}${escapeHtml(object.class_name || "object")}</strong>
                <span>conf: ${escapeHtml(object.confidence ?? "n/a")}</span>
                <span>3D: x=${formatNumber(position.x)}, y=${formatNumber(position.y)}, z=${formatNumber(position.z)}</span>
                <span>keypoints: ${keypointCount}</span>
            </div>
        `;
    }).join("");
}

function formatNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(2) : "n/a";
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
    }[char]));
}

function waitForImageReady(image, timeoutMs) {
    if (image.complete && image.naturalWidth > 0) return Promise.resolve();
    return waitForEventOrTimeout(image, "load", timeoutMs);
}

function waitForEventOrTimeout(target, eventName, timeoutMs) {
    return new Promise(resolve => {
        const timer = setTimeout(cleanup, timeoutMs || 1000);
        function cleanup() {
            clearTimeout(timer);
            target.removeEventListener(eventName, cleanup);
            target.removeEventListener("error", cleanup);
            resolve();
        }
        target.addEventListener(eventName, cleanup, { once: true });
        target.addEventListener("error", cleanup, { once: true });
    });
}
