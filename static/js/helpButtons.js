// Get elements
const helpIcon = document.getElementById("audioFeaturesHelp");
const modal = document.getElementById("helpAudioFeatureModal");
const closeModal = document.getElementById("closeAudioFeatureModal");

// Open modal on question mark click
helpIcon.addEventListener("click", () => {
    modal.style.display = "flex";
});

// Close modal on close button click
closeModal.addEventListener("click", () => {
    modal.style.display = "none";
});

// Close modal on outside click
window.addEventListener("click", (event) => {
    if (event.target === modal) {
        modal.style.display = "none";
    }
});

// Get elements
const mappingHelpIcon = document.getElementById("mappindSettingsHelp");
const mappingModal = document.getElementById("helpMappingModal");
const closeMappingModal = document.getElementById("closeMappingModal");

// Open modal on question mark click
mappingHelpIcon.addEventListener("click", () => {
    mappingModal.style.display = "flex";
});

// Close modal on close button click
closeMappingModal.addEventListener("click", () => {
    mappingModal.style.display = "none";
});

// Close modal on outside click
window.addEventListener("click", (event) => {
    if (event.target === mappingModal) {
        mappingModal.style.display = "none";
    }
});

document.addEventListener("DOMContentLoaded", () => {
    // Get elements
    const sonificatorsHelp = document.getElementById("sonificatorsHelp");
    const sonificatorsModal = document.getElementById("helpSonificatorsModal");
    const closeSonificatorsModal = document.getElementById("closeSonificatorsModal");

    // Open modal on question mark click
    sonificatorsHelp.addEventListener("click", () => {
        sonificatorsModal.style.display = "flex";
    });

    // Close modal on close button click
    closeSonificatorsModal.addEventListener("click", () => {
        sonificatorsModal.style.display = "none";
    });

    // Close modal on outside click
    window.addEventListener("click", (event) => {
        if (event.target === sonificatorsModal) {
            sonificatorsModal.style.display = "none";
        }
    });
});

// Open button: link this to your UI button!
document.getElementById("openClampFeatureModal").addEventListener("click", function() {
    document.getElementById("clampFeatureModal").style.display = "block";
  });
  
  // Close button
  document.getElementById("closeClampFeatureModal").addEventListener("click", function() {
    document.getElementById("clampFeatureModal").style.display = "none";
  });
  
  // Close if user clicks outside the modal content
  window.addEventListener("click", function(event) {
    const modal = document.getElementById("clampFeatureModal");
    if (event.target === modal) {
      modal.style.display = "none";
    }
  });


  function openContactInfoModal(event) {
    event.preventDefault();
    document.getElementById('contactInfoModal').style.display = 'block';
  }
  
  document.getElementById('closeContactInfoModal').addEventListener('click', function() {
    document.getElementById('contactInfoModal').style.display = 'none';
  });
  
  // Optional: close when clicking outside the modal
  window.addEventListener('click', function(event) {
    const modal = document.getElementById('contactInfoModal');
    if (event.target === modal) {
      modal.style.display = 'none';
    }
  });
  
