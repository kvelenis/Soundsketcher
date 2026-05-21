(function () {
  window.SoundSketcherApp.onReady("header controls", () => {
    bindPrintButton();
    bindExamplesMenu();
    bindTutorialModal();
  });

  function bindPrintButton() {
    const printButton = document.getElementById("printBtn");
    if (!printButton) return;
    printButton.addEventListener("click", () => window.print());
  }

  function bindExamplesMenu() {
    const toggle = document.getElementById("examplesToggle");
    const list = document.getElementById("examplesList");
    if (!toggle || !list) return;

    let hasLoaded = false;

    list.addEventListener("click", (event) => event.stopPropagation());
    toggle.addEventListener("click", async (event) => {
      event.stopPropagation();

      if (list.hidden) {
        if (!hasLoaded) {
          hasLoaded = await populateExamples(list);
        }
        list.hidden = false;
      } else {
        list.hidden = true;
      }
    });
    toggle.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      toggle.click();
    });

    document.addEventListener("click", () => {
      list.hidden = true;
    });
  }

  async function populateExamples(list) {
    list.replaceChildren(createExamplesStatus("Loading examples..."));

    try {
      const cacheClient = await (
        window.SoundSketcher?.whenAudioClient?.("cache") || Promise.resolve(window.SoundSketcherAudioCacheClient)
      );
      const cachedFiles = await cacheClient.listCachedFiles();
      list.replaceChildren();

      if (cachedFiles.length === 0) {
        list.appendChild(createExamplesStatus("No cached examples yet."));
        return;
      }

      cachedFiles.forEach(({ filename, hash, is_preferred: isPreferred }) => {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.className = isPreferred
          ? "examples-dropdown__button examples-dropdown__button--preferred"
          : "examples-dropdown__button";
        button.textContent = filename;
        if (isPreferred) {
          button.setAttribute("aria-label", `${filename} current QA example`);
        }
        button.addEventListener("click", () => {
          if (typeof fetchPreviouslyProcessed === "function") {
            fetchPreviouslyProcessed(filename, hash);
          }
          list.hidden = true;
        });
        item.appendChild(button);
        list.appendChild(item);
      });
      return true;
    } catch (error) {
      console.warn("Could not load cached examples.", error);
      list.replaceChildren(createExamplesStatus("Could not load examples."));
      return false;
    }
  }

  function createExamplesStatus(text) {
    const item = document.createElement("li");
    item.className = "examples-dropdown__status";
    item.textContent = text;
    return item;
  }

  function bindTutorialModal() {
    const openButton = document.getElementById("openTutorialModal");
    const modal = document.getElementById("tutorialModal");
    const closeButton = document.getElementById("closeTutorialModal");
    const searchInput = document.getElementById("tutorialSearchInput");
    const clearSearchButton = document.getElementById("clearSearchBtn");
    const noResultsMessage = document.getElementById("noResultsMsg");

    if (openButton && modal) {
      openButton.addEventListener("click", (event) => {
        event.preventDefault();
        modal.style.display = "block";
      });
    }

    if (closeButton && modal) {
      closeButton.addEventListener("click", () => {
        modal.style.display = "none";
      });
    }

    if (modal) {
      window.addEventListener("click", (event) => {
        if (event.target === modal) modal.style.display = "none";
      });
    }

    if (searchInput && noResultsMessage) {
      searchInput.addEventListener("input", () => {
        const query = searchInput.value.toLowerCase();
        let found = false;

        document.querySelectorAll(".tutorial-content section").forEach((section) => {
          const match = section.innerText.toLowerCase().includes(query);
          section.style.display = match ? "block" : "none";
          if (match) found = true;
        });

        noResultsMessage.style.display = found || query === "" ? "none" : "block";
      });
    }

    if (clearSearchButton && searchInput) {
      clearSearchButton.addEventListener("click", () => {
        searchInput.value = "";
        searchInput.dispatchEvent(new Event("input"));
      });
    }

    bindTutorialTocObserver();
  }

  function bindTutorialTocObserver() {
    const tocLinks = document.querySelectorAll("#tutorialTOC a");
    const sections = [...document.querySelectorAll(".tutorial-content section")];
    if (!tocLinks.length || !sections.length || !("IntersectionObserver" in window)) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .map((entry) => ({
            id: entry.target.id,
            top: entry.target.getBoundingClientRect().top,
          }));

        if (!visible.length) return;

        const topmost = visible.sort((a, b) => a.top - b.top)[0];
        tocLinks.forEach((link) => link.classList.remove("active"));

        const activeLink = document.querySelector(`#tutorialTOC a[href='#${topmost.id}']`);
        if (activeLink) activeLink.classList.add("active");
      },
      {
        root: null,
        threshold: 0.2,
        rootMargin: "0px 0px -60% 0px",
      },
    );

    sections.forEach((section) => observer.observe(section));
  }
})();
