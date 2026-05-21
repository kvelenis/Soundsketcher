function initializeTooltip() {
    const tooltip = document.getElementById("tooltip");
    const svgCanvas = document.getElementById("svgWrapper");
    if (!tooltip || !svgCanvas) return;

    const rect = svgCanvas.getBoundingClientRect();
    const canvasX = rect.left;
    const canvasY = rect.top;
    const canvasWidth = rect.width;
    const canvasHeight = rect.height;

    const paths = svgCanvas.querySelectorAll("path,circle");
    paths.forEach((path) => {
        const features = path.dataset.features;

        path.addEventListener("mouseover", () => {
            if (features) {
                tooltip.innerHTML = features;
                tooltip.style.display = "block";
            }
        });

        path.addEventListener("mousemove", (event) => {
            const mouseX = event.clientX;
            const mouseY = event.clientY;
            const xEdge = mouseX >= canvasX + window.scrollX + canvasWidth - tooltip.offsetWidth;
            const yEdge = mouseY >= canvasY + window.scrollY + canvasHeight - tooltip.offsetHeight;

            if (!xEdge && !yEdge) {
                tooltip.style.left = `${mouseX + 10}px`;
                tooltip.style.top = `${mouseY + 10}px`;
            } else if (xEdge && !yEdge) {
                tooltip.style.left = `${mouseX - tooltip.offsetWidth}px`;
                tooltip.style.top = `${mouseY + 10}px`;
            } else if (xEdge && yEdge) {
                tooltip.style.left = `${mouseX - tooltip.offsetWidth}px`;
                tooltip.style.top = `${mouseY - tooltip.offsetHeight}px`;
            } else {
                tooltip.style.left = `${mouseX}px`;
                tooltip.style.top = `${mouseY - tooltip.offsetHeight}px`;
            }
        });

        path.addEventListener("mouseout", () => {
            tooltip.style.display = "none";
        });
    });
}
