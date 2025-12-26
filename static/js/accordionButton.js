// document.addEventListener("DOMContentLoaded",() =>
// {
//     const accordionHeaders = document.querySelectorAll(".accordion-header");

//     accordionHeaders.forEach(header => {
//         const content = header.nextElementSibling; // Assuming content is immediately after the header

//         header.addEventListener("click", () => {
//             // Toggle the "open" class for the content
//             content.classList.toggle("open");

//             // Smoothly adjust max-height
//             if (content.classList.contains("open")) {
//                 content.style.maxHeight = content.scrollHeight + "px";
//                 console.log("Accordion opened.");
//             } else {
//                 content.style.maxHeight = null;
//                 console.log("Accordion closed.");
//             }
//         });
//     });
// });

document.addEventListener("DOMContentLoaded",() =>
{
    const accordionHeaders = document.querySelectorAll(".accordion-header");

    accordionHeaders.forEach(header => {
        const modal = header.nextElementSibling; // Assuming modal is immediately after the header
        const closeButton = modal.querySelector('.close-button');

        header.addEventListener("click", () => {
            // Toggle the "open" class for the modal
            modal.style.display = "block";

        });

        closeButton.addEventListener("click", () =>
        {
            modal.style.display = "none";
        });

        window.addEventListener("click", (event) =>
        {
            if (event.target === modal) {
                modal.style.display = "none";
            }
        });
    });
});