function initializeTooltip() {

    const tooltip = document.getElementById("tooltip");
    const svgCanvas = document.getElementById("svgWrapper"); //! Using wrapper for correct functionality in scrollable mode
    //! Added these
    const rect = svgCanvas.getBoundingClientRect();
    const canvas_x = rect.left;
    const canvas_y = rect.top;
    const canvas_width = rect.width;
    const canvas_height = rect.height;

    // Assuming each path has a `data-features` attribute containing its features
    const paths = svgCanvas.querySelectorAll("path,circle"); //! Added circle
    // console.log("Paths", paths)
    paths.forEach((path) =>
    {
        const features = path.dataset.features; // Ensure features are stored in this attribute

        path.addEventListener("mouseover",(event) =>
        {
            if (features)
            {
                tooltip.innerHTML = features; // Set the tooltip content
                tooltip.style.display = "block"; // Show the tooltip
            }
        });

        //! Changed logic here to move tooltip position
        path.addEventListener("mousemove",(event) =>
        {
            const mouseX = event.clientX;
            const mouseY = event.clientY;

            //! Set tooltip position
            const x_edge = (mouseX >= canvas_x + window.scrollX + canvas_width - tooltip.offsetWidth);
            const y_edge = (mouseY >= canvas_y + window.scrollY + canvas_height - tooltip.offsetHeight);
            if(!x_edge && !y_edge)
            {
                tooltip.style.left = `${mouseX + 10}px`;
                tooltip.style.top = `${mouseY + 10}px`;
            }
            else if(x_edge && !y_edge)
            {
                tooltip.style.left = `${mouseX - tooltip.offsetWidth}px`;
                tooltip.style.top = `${mouseY + 10}px`;
            }
            else if(x_edge && y_edge)
            {
                tooltip.style.left = `${mouseX - tooltip.offsetWidth}px`;
                tooltip.style.top = `${mouseY - tooltip.offsetHeight}px`;
            }
            else if(!x_edge && y_edge)
            {
                tooltip.style.left = `${mouseX}px`;
                tooltip.style.top = `${mouseY - tooltip.offsetHeight}px`;
            }
        });

        path.addEventListener("mouseout",() =>
        {
            tooltip.style.display = "none"; // Hide the tooltip when the cursor leaves
        });
    });
}