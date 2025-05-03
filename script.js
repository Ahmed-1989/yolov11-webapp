
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let session = null;

// Class names (from your data.yaml)
const classNames = ['Boom Lift', 'Concrete Casting', 'Concrete Mixer', 'Concrete Pump', 'Excavator', 'Formwork',
    'Mobile Crane', 'Painting', 'Skid Steer', 'Steel Work', 'Telescopic Handler',
    'Tower Crane', 'Truck', 'Vibrating', 'Wheel Loader'];

// Load ONNX model
async function loadModel() {
    session = new onnx.InferenceSession();
    await session.loadModel("yolov11_export.onnx");
    console.log("✅ YOLOv11 ONNX model loaded");
}

loadModel();

// Access the camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            requestAnimationFrame(detectFrame);
        };
    })
    .catch((err) => console.error("Camera error:", err));

// Main detection loop
async function detectFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (session) {
        const inputTensor = await preprocessInput(canvas);
        const outputMap = await session.run({ images: inputTensor });
        const outputData = outputMap.output.data;

        const boxes = postprocessOutput(outputData, canvas.width, canvas.height);
        drawBoxes(boxes);
    }

    requestAnimationFrame(detectFrame);
}

// Convert canvas to ONNX input tensor
async function preprocessInput(canvas) {
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 640;
    tmpCanvas.height = 640;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(canvas, 0, 0, 640, 640);
    const imgData = tmpCtx.getImageData(0, 0, 640, 640);
    const { data } = imgData;

    // Normalize and reorder channels
    const floatData = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
        floatData[i] = data[i * 4] / 255.0; // R
        floatData[i + 640 * 640] = data[i * 4 + 1] / 255.0; // G
        floatData[i + 2 * 640 * 640] = data[i * 4 + 2] / 255.0; // B
    }

    return new onnx.Tensor(floatData, 'float32', [1, 3, 640, 640]);
}

// Decode YOLOv11 output (basic, assumes format [x, y, w, h, conf, ...class_scores])
function postprocessOutput(outputData, imgWidth, imgHeight) {
    const numDetections = outputData.length / 85;
    const boxes = [];
    const confThreshold = 0.4;

    for (let i = 0; i < numDetections; i++) {
        const offset = i * 85;
        const conf = outputData[offset + 4];

        if (conf > confThreshold) {
            let maxClass = 0;
            let maxScore = 0;
            for (let j = 5; j < 85; j++) {
                if (outputData[offset + j] > maxScore) {
                    maxScore = outputData[offset + j];
                    maxClass = j - 5;
                }
            }

            const cx = outputData[offset] * imgWidth;
            const cy = outputData[offset + 1] * imgHeight;
            const w = outputData[offset + 2] * imgWidth;
            const h = outputData[offset + 3] * imgHeight;
            const x1 = cx - w / 2;
            const y1 = cy - h / 2;

            boxes.push({
                x: x1,
                y: y1,
                width: w,
                height: h,
                label: classNames[maxClass],
                score: conf.toFixed(2)
            });
        }
    }

    return boxes;
}

// Draw detection results
function drawBoxes(boxes) {
    ctx.lineWidth = 2;
    ctx.font = "16px Arial";
    ctx.textBaseline = "top";

    boxes.forEach(box => {
        ctx.strokeStyle = "red";
        ctx.fillStyle = "red";
        ctx.strokeRect(box.x, box.y, box.width, box.height);
        ctx.fillText(`${box.label} (${box.score})`, box.x + 4, box.y + 2);
    });
}
