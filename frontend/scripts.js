// Add event listener for the click-to-upload feature
document.getElementById('dropZone').addEventListener('click', () => {
    document.getElementById('imageInput').click();
});

async function predictImage() {
    const fileInput = document.getElementById("imageInput");
    const resultDiv = document.getElementById("result");
    const loadingDiv = document.getElementById("loading");
    const predictBtn = document.getElementById("predictBtn");
    
    if (fileInput.files.length === 0) {
        resultDiv.innerHTML = `<div style='color:var(--danger); text-align:center;'>⚠ Please select an image first.</div>`;
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    loadingDiv.style.display = "block";
    resultDiv.innerHTML = "";
    predictBtn.disabled = true;

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error(`Server Error: ${response.status}`);

        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `<div class="prediction-card" style="border-color:var(--danger)">Error: ${data.error}</div>`;
        } else {
            const isTumor = data.prediction !== "No_Tumor";
            const statusColor = isTumor ? "var(--danger)" : "var(--success)";
            
            let gradcamHtml = data.gradcam_image ? `
                <div style="margin-top: 15px;">
                    <p style="font-size: 0.8rem; color: var(--text-secondary);">Heatmap Analysis</p>
                    <img src="data:image/jpeg;base64,${data.gradcam_image}" class="heatmap-img">
                </div>
            ` : "";
            // Define specific advice for each category
            const healthAdvice = {
                "Glioma": "Gliomas require immediate consultation with a neuro-oncologist. Treatment often involves a combination of surgery, radiation, and chemotherapy.",
                "Meningioma": "Meningiomas are often slow-growing, but require evaluation by a specialist to determine if they are pressing on vital brain tissue.",
                "Pituitary": "Pituitary tumors can affect hormone levels. It is advised to see both a neurologist and an endocrinologist for a full evaluation.",
                "No_Tumor": "The scan shows no immediate signs of a tumor. However, if you are experiencing symptoms like persistent headaches or vision changes, please consult a doctor."
            };
            const currentAdvice = healthAdvice[data.prediction] || "Please consult a medical professional for a detailed analysis of these results.";

            resultDiv.innerHTML = `
                <div class="prediction-card" style="border-color: ${statusColor}">
                    <p style="color: var(--text-secondary); font-size: 0.8rem; margin: 0;">ANALYSIS COMPLETE</p>
                    <h3 style="margin: 5px 0;">Type: <span style="color:${statusColor}">${data.prediction.replace('_', ' ')}</span></h3>
                    <p>Confidence: <strong>${(data.confidence * 100).toFixed(2)}%</strong></p>
                    ${gradcamHtml}
                </div>
                <div class="advice-box" style="margin-top:15px; padding:10px; background:rgba(255,255,255,0.05); border-radius:8px;">
                    <strong style="color:var(--accent-color); font-size:0.9rem;">Health Advisory:</strong>
                    <p style="font-size:0.85rem; margin:5px 0 0 0; line-height:1.4;">${currentAdvice}</p>
                </div>  
            `;
        }
    } catch (error) {
        resultDiv.innerHTML = `<div style="color:var(--danger)">Connection failed. Check if API is live.</div>`;
    } finally {
        loadingDiv.style.display = "none";
        predictBtn.disabled = false;
    }
}

const fileInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const uploadText = dropZone.querySelector("p"); // Selects the text line

// When the file input changes (photo is selected)
fileInput.addEventListener('change', function() {
    if (this.files && this.files.length > 0) {
        const fileName = this.files[0].name; // Get the name
        
        // Update the UI place with the filename
        uploadText.innerHTML = `Selected: <span style="color: #38bdf8; font-weight: bold;">${fileName}</span>`;
        
        // Optional: Change box border to green to show success
        dropZone.style.borderColor = "#22c55e"; 
    }
});

// Function to close the disclaimer modal
document.getElementById('closeModalBtn').addEventListener('click', function() {
    const modal = document.getElementById('disclaimerModal');
    
    // Add a fade-out effect
    modal.style.opacity = '0';
    setTimeout(() => {
        modal.style.display = 'none';
    }, 300); // Wait for transition to finish
});

// Logic for fading out CSS (add this to your styles.css for the smooth effect)
// .modal-overlay { transition: opacity 0.3s ease; }