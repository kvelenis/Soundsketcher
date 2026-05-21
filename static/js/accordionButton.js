(function () {
    function bindAccordionModals() {
        document.querySelectorAll(".accordion-header").forEach((header) => {
            const modal = header.nextElementSibling;
            const closeButton = modal?.querySelector(".close-button");
            if (!modal || !closeButton) return;

            header.addEventListener("click", () => {
                modal.style.display = "block";
            });

            closeButton.addEventListener("click", () => {
                modal.style.display = "none";
            });

            window.addEventListener("click", (event) => {
                if (event.target === modal) {
                    modal.style.display = "none";
                }
            });
        });
    }

    window.SoundSketcherApp.onReady("accordion modals", bindAccordionModals);
})();
