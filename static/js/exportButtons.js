document.getElementById("exportButton").addEventListener("click", () => {
    const svg = document.getElementById("svgCanvas");

    if(svg)
    {
        const svgData = new XMLSerializer().serializeToString(svg);
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        const svgBoundingBox = svg.getBoundingClientRect();

        // Define scaling factor for higher resolution
        const scaleFactor = 3; // Adjust for desired quality (e.g., 2 = 2x quality)

        canvas.width = svgBoundingBox.width * scaleFactor;
        canvas.height = svgBoundingBox.height * scaleFactor;

        ctx.scale(scaleFactor, scaleFactor); // Scale the context for higher resolution

        const img = new Image();
        const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(svgBlob);

        img.onload = () => {
            // Optional: Set background color for PNGs
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width / scaleFactor, canvas.height / scaleFactor);

            // Draw the SVG onto the canvas
            ctx.drawImage(img, 0, 0);

            // Export the canvas to a PNG or JPEG
            const imageType = "image/png"; // Use "image/jpeg" for JPEG
            const image = canvas.toDataURL(imageType, 1.0); // 1.0 for highest quality (JPEG only)

            // Create a download link
            const link = document.createElement("a");
            link.href = image;
            link.download = "exported_image.png"; // Change extension for JPEG
            link.click();

            // Cleanup
            URL.revokeObjectURL(url);
        };

        img.src = url;
    }
});


document.addEventListener("DOMContentLoaded", function () {
    const exportButton = document.getElementById("exportFeaturesButton");

    exportButton.addEventListener("click", function () {
        // Assuming globalAudioData is a global variable
        // if (typeof globalAudioData === "undefined" || Object.keys(globalAudioData).length === 0) {
        if (!globalAudioData) {
            // alert("No features to export.");
            return;
        }

        // Convert globalAudioData to JSON string
        const jsonData = JSON.stringify(globalAudioData, null, 2);

        // Create a blob and trigger download
        const blob = new Blob([jsonData], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "audio_features.json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        // Optional: Provide feedback
        console.log("Audio features exported:", globalAudioData);
    });
});