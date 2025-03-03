document.addEventListener("DOMContentLoaded", () => {
    const accordionHeaders = document.querySelectorAll(".accordion-header");

    accordionHeaders.forEach(header => {
        const content = header.nextElementSibling; // Assuming content is immediately after the header

        header.addEventListener("click", () => {
            // Toggle the "open" class for the content
            content.classList.toggle("open");

            // Smoothly adjust max-height
            if (content.classList.contains("open")) {
                content.style.maxHeight = content.scrollHeight + "px";
                console.log("Accordion opened.");
            } else {
                content.style.maxHeight = null;
                console.log("Accordion closed.");
            }
        });
    });
});