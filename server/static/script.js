

// const dropArea = document.getElementById("dropArea");
// const input = document.getElementById("imageInput");
// const preview = document.getElementById("preview");

// dropArea.addEventListener("click", () => input.click());

// dropArea.addEventListener("dragover", (e) => {
//     e.preventDefault();
//     dropArea.classList.add("dragover");
// });

// dropArea.addEventListener("dragleave", () => {
//     dropArea.classList.remove("dragover");
// });

// dropArea.addEventListener("drop", (e) => {
//     e.preventDefault();
//     dropArea.classList.remove("dragover");

//     const file = e.dataTransfer.files[0];
//     handleFile(file);
// });

// input.addEventListener("change", () => {
//     handleFile(input.files[0]);
// });

// function handleFile(file) {
//     const reader = new FileReader();

//     reader.onloadend = function () {
//         const base64Image = reader.result;

//         // Preview
//         preview.src = base64Image;

//         sendToServer(base64Image);
//     };

//     reader.readAsDataURL(file);
// }

// function sendToServer(image) {
//     const formData = new FormData();
//     formData.append("image_data", image);

//     fetch("/api/classify", {
//         method: "POST",
//         body: formData
//     })
//     .then(res => res.json())
//     .then(data => {
//         showResult(data);
//     })
//     .catch(err => {
//         console.error(err);
//         alert("Error occurred");
//     });
// }

// function showResult(data) {
//     if (data.error) {
//         document.getElementById("prediction").innerText = data.error;
//         return;
//     }

//     document.getElementById("prediction").innerText =
//         `${data.class} (${data.confidence}%)`;

//     document.getElementById("confidenceFill").style.width =
//         data.confidence + "%";
// }
const dropArea = document.getElementById("dropArea");
const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const predictionText = document.getElementById("prediction");
const confidenceFill = document.getElementById("confidenceFill");
const loader = document.getElementById("loader");

// -----------------------------
// Drag & Drop Events
// -----------------------------
dropArea.addEventListener("click", () => input.click());

dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");

    const file = e.dataTransfer.files[0];
    handleFile(file);
});

// -----------------------------
// File Input
// -----------------------------
input.addEventListener("change", () => {
    handleFile(input.files[0]);
});

// -----------------------------
// Handle File
// -----------------------------
function handleFile(file) {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
        alert("Please upload a valid image file");
        return;
    }

    const img = new Image();
    const reader = new FileReader();

    reader.onload = function (e) {
        img.src = e.target.result;
    };

    img.onload = function () {
        const canvas = document.createElement("canvas");
        const maxSize = 500;  // 🔥 resize limit

        let width = img.width;
        let height = img.height;

        if (width > height) {
            if (width > maxSize) {
                height *= maxSize / width;
                width = maxSize;
            }
        } else {
            if (height > maxSize) {
                width *= maxSize / height;
                height = maxSize;
            }
        }

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, width, height);

        // 🔥 compress image
        const compressedImage = canvas.toDataURL("image/jpeg", 0.7);

        preview.src = compressedImage;
        sendToServer(compressedImage);
    };

    reader.readAsDataURL(file);
}

// -----------------------------
// API Call
// -----------------------------
function sendToServer(image) {
    const formData = new FormData();
    formData.append("image_data", image);

    fetch("/api/classify", {
        method: "POST",
        body: formData
    })
    .then(async (res) => {
        const text = await res.text();   // 👈 get raw response

        try {
            return JSON.parse(text);     // 👈 safely parse
        } catch (e) {
            console.error("Invalid JSON response:", text);
            throw new Error("Server returned invalid response");
        }
    })
    .then(data => {
        showLoader(false);
        showResult(data);
    })
    .catch(err => {
        showLoader(false);
        console.error(err);
        predictionText.innerText = "❌ Server Error";
    });
}
// -----------------------------
// Show Result
// -----------------------------
function showResult(data) {
    if (!data || data.error) {
        predictionText.innerText = data?.error || "No result";
        confidenceFill.style.width = "0%";
        return;
    }

    const confidence = parseFloat(data.confidence).toFixed(2);

    // Update text
    predictionText.innerText = `${data.class} (${confidence}%)`;

    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = confidence + "%";
    }, 100);
}

// -----------------------------
// Loader Toggle
// -----------------------------
function showLoader(show) {
    loader.style.display = show ? "block" : "none";
}

// -----------------------------
// Reset UI
// -----------------------------
function resetUI() {
    predictionText.innerText = "";
    confidenceFill.style.width = "0%";
}