let scrfdSession;
let canvas, ctx;
let faceBox = null;

// --- Load Model ---
async function initSCRFD() {
          scrfdSession = await ort.InferenceSession.create('models/det_500m.onnx');
    //   arcfaceSession = await ort.InferenceSession.create('models/w600k_mbf.onnx');
    console.log('SCRFD model loaded');
}

// --- Preprocess Image to Tensor ---
function preprocessSCRFD(imageData) {
    const { width, height, data } = imageData;
    const inputWidth = 640;
    const inputHeight = 640;

    // Resize to 640x640
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = inputWidth;
    resizedCanvas.height = inputHeight;
    const resizedCtx = resizedCanvas.getContext('2d');
    resizedCtx.drawImage(canvas, 0, 0, inputWidth, inputHeight);

    const resized = resizedCtx.getImageData(0, 0, inputWidth, inputHeight);
    const float32Data = new Float32Array(inputWidth * inputHeight * 3);

    for (let i = 0; i < inputWidth * inputHeight; i++) {
        float32Data[i] = resized.data[i * 4 + 2] / 255.0; // R
        float32Data[i + inputWidth * inputHeight] = resized.data[i * 4 + 1] / 255.0; // G
        float32Data[i + inputWidth * inputHeight * 2] = resized.data[i * 4] / 255.0; // B
    }

    return new ort.Tensor('float32', float32Data, [1, 3, inputHeight, inputWidth]);
}

// --- Postprocess Output ---
function postprocessSCRFD(outputs, origWidth, origHeight, threshold = 0.5) {
    const strides = [8, 16, 32];
    let bboxes = [];
    let scores = [];

    for (const stride of strides) {
        const scoreBlob = outputs[`score_${stride}`];
        const bboxBlob = outputs[`bbox_${stride}`];
        const kpsBlob = outputs[`kps_${stride}`];

        if (!scoreBlob || !bboxBlob) continue;

        const scoresData = scoreBlob.data;
        const bboxData = bboxBlob.data;
        const feature_h = 640 / stride;
        const feature_w = 640 / stride;

        let idx = 0;
        for (let h = 0; h < feature_h; h++) {
            for (let w = 0; w < feature_w; w++, idx++) {
                const score = scoresData[idx];
                if (score < threshold) continue;

                const x1 = (w + 0.5 - bboxData[idx * 4]) * stride;
                const y1 = (h + 0.5 - bboxData[idx * 4 + 1]) * stride;
                const x2 = (w + 0.5 + bboxData[idx * 4 + 2]) * stride;
                const y2 = (h + 0.5 + bboxData[idx * 4 + 3]) * stride;

                bboxes.push([x1, y1, x2, y2]);
                scores.push(score);
            }
        }
    }

    if (bboxes.length === 0) return null;

    // Pick the box with the highest score (simple version)
    let bestIdx = scores.indexOf(Math.max(...scores));
    let [x1, y1, x2, y2] = bboxes[bestIdx];

    // Scale back to original image size
    const scaleX = origWidth / 640;
    const scaleY = origHeight / 640;
    return {
        x: x1 * scaleX,
        y: y1 * scaleY,
        width: (x2 - x1) * scaleX,
        height: (y2 - y1) * scaleY
    };
}

// --- Draw Box ---
function drawBox(box) {
    if (!box) return;
    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 3;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
}

// --- Detect Face ---
document.getElementById('detectBtn').onclick = async () => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = preprocessSCRFD(imageData);

    const feeds = {};
    feeds[scrfdSession.inputNames[0]] = inputTensor;
    const output = await scrfdSession.run(feeds);

    faceBox = postprocessSCRFD(output, canvas.width, canvas.height);
    if (faceBox) {
        drawBox(faceBox);
        document.getElementById('result').textContent = 'Face detected!';
    } else {
        document.getElementById('result').textContent = 'No face detected.';
    }
};

// --- Init ---
window.onload = async () => {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    await initSCRFD();
    document.getElementById('result').textContent = 'Model ready.';
};
