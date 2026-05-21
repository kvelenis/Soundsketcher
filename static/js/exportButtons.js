(function () {
    function exportSvgAsPng() {
        const svg = document.getElementById("svgCanvas");

        if (!svg) {
            return;
        }

        const svgData = new XMLSerializer().serializeToString(svg);
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        const svgBoundingBox = svg.getBoundingClientRect();
        const scaleFactor = 3;

        canvas.width = svgBoundingBox.width * scaleFactor;
        canvas.height = svgBoundingBox.height * scaleFactor;
        ctx.scale(scaleFactor, scaleFactor);

        const img = new Image();
        const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(svgBlob);

        img.onload = () => {
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width / scaleFactor, canvas.height / scaleFactor);
            ctx.drawImage(img, 0, 0);

            const link = document.createElement("a");
            link.href = canvas.toDataURL("image/png", 1.0);
            link.download = "exported_image.png";
            link.click();

            URL.revokeObjectURL(url);
        };

        img.onerror = () => {
            URL.revokeObjectURL(url);
        };

        img.src = url;
    }

    function exportFeaturesAsJson() {
        if (!window.globalAudioData && typeof globalAudioData === "undefined") {
            return;
        }

        const audioData = window.globalAudioData || globalAudioData;

        if (!audioData) {
            return;
        }

        const blob = new Blob([JSON.stringify(audioData, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = url;
        link.download = "audio_features.json";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        console.log("Audio features exported:", audioData);
    }

    function bindExportButtons() {
        document.getElementById("exportButton")?.addEventListener("click", exportSvgAsPng);
        document.getElementById("exportFeaturesButton")?.addEventListener("click", exportFeaturesAsJson);
    }

    window.SoundSketcherApp.onReady("export buttons", bindExportButtons);
})();
