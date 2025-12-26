//! New script for automatic resketching

document.addEventListener("DOMContentLoaded",() =>
{
    const resketch_button = document.getElementById('submitButton');
    const auto_resketch_button = document.getElementById('auto_resketch_button');
    auto_resketch_button.addEventListener("click",auto_resketch_handler)

    function auto_resketch_handler()
    {
        if(auto_resketch_button.checked)
        {
            if(globalAudioData && globalFile)
            {
                resketch_button.click();
            }
        }
    }

    function slider_handler()
    {
        const slider_element = this.target;
        const toggle_id = slider_element.dataset.toggle;
        if(toggle_id)
        {
            const toggle_button = document.querySelector(toggle_id);
            if(!toggle_button.checked)
            {
                return;
            }
        }
        auto_resketch_handler();
    }

    // Color SLiders
    document.addEventListener("change",(event) =>
    {
        if(event.target.classList.contains("color-slider"))
        {
            auto_resketch_handler(event);
        }
    });
    
    // 'Audio Features' Selectors
    const feature_selectors = document.querySelectorAll('.featureSelect');
    feature_selectors.forEach
    (
        (selector) =>
        {
            selector.addEventListener("change",auto_resketch_handler);
        }
    );
    
    // 'Invert Mapping' Checkboxes
    const invert_checkboxes = document.querySelectorAll('.invertMappingCheckbox');
    invert_checkboxes.forEach
    (
        (checkbox) =>
        {
            checkbox.addEventListener("click",auto_resketch_handler);
        }
    );
    
    // 'Min/Max' + 'Hard/Soft Clipping' Sliders
    const range_sliders = document.querySelectorAll('.range-slider');
    range_sliders.forEach
    (
        (slider) =>
        {
            slider.noUiSlider.on("set",slider_handler);
        }
    );

    // 'Randomize' Button
    const randomize_button = document.getElementById("randomizeFeaturesButton");
    randomize_button.addEventListener("click",auto_resketch_handler);

    // 'Reset All' Button
    const reset_button = document.getElementById("resetButton");
    reset_button.addEventListener("click",auto_resketch_handler);

    // 'Preset' Buttons
    const preset_buttons = document.querySelectorAll(".preset-button");
    preset_buttons.forEach
    (
        (preset_button) =>
        {
            preset_button.addEventListener("click",auto_resketch_handler);
        }
    );

    // Global Settings
    const global_settings = document.querySelectorAll('.global-settings__checkbox:not(#auto_resketch_button):not(#filter_button)');
    global_settings.forEach
    (
        (global_setting) =>
        {
            global_setting.addEventListener("click",auto_resketch_handler);
        }
    );

    // Feature Clamping Control
    const clamp = document.getElementById("toggleClamp");
    const softclip = document.getElementById("toggleSoftclip");
    document.getElementById("closeClampFeatureModal").addEventListener("click",() =>
    {
        if(clamp.checked || softclip.checked)
        {
            auto_resketch_handler();
        }
    });
    document.addEventListener("click",event =>
    {
        const modal = document.getElementById("clampFeatureModal");
        if (event.target === modal)
        {
            if(clamp.checked || softclip.checked)
            {
                auto_resketch_handler();
            }
        }
    });

    // 'Randomize Features' Button Hotkey
    document.addEventListener("keydown",(event) =>
    {
        if (event.key.toLowerCase() === "s")
        {
            auto_resketch_handler()
        }
    });

    // 'Reset All' Button Hotkey
    document.addEventListener("keydown",(event) =>
    {
        if (event.key.toLowerCase() === "d")
        {
            auto_resketch_handler()
        }
    });

    // 'Preset A' Button Hotkey
    document.addEventListener("keydown",(event) =>
    {
        if (event.key.toLowerCase() === "z")
        {
            auto_resketch_handler()
        }
    });

    // 'Preset B' Button Hotkey
    document.addEventListener("keydown",(event) =>
    {
        if (event.key.toLowerCase() === "x")
        {
            auto_resketch_handler()
        }
    });

    // 'Preset C' Button Hotkey
    document.addEventListener("keydown",(event) =>
    {
        if (event.key.toLowerCase() === "c")
        {
            auto_resketch_handler()
        }
    });

    //! Temp #tmpf0
    document.getElementById("tmp-slider").addEventListener("click",auto_resketch_handler);
    document.getElementById("gamma-slider").addEventListener("click",auto_resketch_handler);
});