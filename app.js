const classNames = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma'];

document.getElementById("upload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = async () => {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, 224, 224);

    const imageData = ctx.getImageData(0, 0, 224, 224);
    const inputTensor = preprocessImage(imageData); // Float32Array

    const session = await ort.InferenceSession.create("lung_cancer_detection_model.onnx");
    const tensor = new ort.Tensor("float32", inputTensor, [1, 3, 224, 224]);
    const output = await session.run({ input: tensor });

    const outputArray = output.output.data;
    const predicted = outputArray.indexOf(Math.max(...outputArray));
    const resultElement = document.getElementById("result");
    const prediction = classNames[predicted];

    let message = "";
    if (prediction === "Normal") {
      message = "✅ Your scan appears normal. No signs of lung cancer were detected.";
      resultElement.classList.remove("red");
      resultElement.classList.add("green");
    } else {
      message = `⚠️ Possible indication of: ${prediction}. Please consult a medical professional for further evaluation.`;
      resultElement.classList.remove("green");
      resultElement.classList.add("red");
    }

    resultElement.textContent = message;
    resultElement.classList.remove("fade-in");
    resultElement.classList.add("fade-in");
  };

  img.src = URL.createObjectURL(file);
});

function preprocessImage(imageData) {
  const { data, width, height } = imageData;
  const floatData = new Float32Array(3 * width * height);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const r = data[idx] / 255;
      const g = data[idx + 1] / 255;
      const b = data[idx + 2] / 255;

      const pixelIndex = y * width + x;
      floatData[pixelIndex] = (r - mean[0]) / std[0];           // R
      floatData[width * height + pixelIndex] = (g - mean[1]) / std[1]; // G
      floatData[2 * width * height + pixelIndex] = (b - mean[2]) / std[2]; // B
    }
  }

  return floatData;
}
