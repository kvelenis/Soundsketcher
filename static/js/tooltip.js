function initializeTooltip() {
    const svgCanvas = document.getElementById("svgCanvas");
    const tooltip = document.getElementById("tooltip");

    // Assuming each path has a `data-features` attribute containing its features
    const paths = svgCanvas.querySelectorAll("path");
    console.log("Paths", paths)
    paths.forEach((path) => {
        const features = path.dataset.features; // Ensure features are stored in this attribute

        path.addEventListener("mouseover", (event) => {
            if (features) {
                tooltip.innerHTML = features; // Set the tooltip content
                tooltip.style.display = "block"; // Show the tooltip
            }
        });

        path.addEventListener("mousemove", (event) => {
            // Position the tooltip near the cursor
            tooltip.style.left = `${event.pageX + 10}px`;
            tooltip.style.top = `${event.pageY + 10}px`;
        });

        path.addEventListener("mouseout", () => {
            tooltip.style.display = "none"; // Hide the tooltip when the cursor leaves
        });
    });
}