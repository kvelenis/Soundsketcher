// Disabled legacy SVG drag-selection experiment.
//
// This is intentionally not loaded by app_scripts.html yet. It depends on
// svgDragSelect, gsap, Draggable, and selection globals that need a proper
// feature boundary before this can be safely reactivated.

function initializeSvgDragSelect() {
    return svgDragSelect({
        svg: svgElement,
        referenceElement: null,
        selector: "enclosure",

        onSelectionStart({ svg, pointerEvent, cancel }) {
            if (pointerEvent.button !== 0) {
                cancel();
                return;
            }

            selectedElements = svg.querySelectorAll('[data-selected]');
            for (let i = 0; i < selectedElements.length; i++) {
                selectedElements[i].removeAttribute('data-selected');
            }
        },

        onSelectionChange({ newlySelectedElements, newlyDeselectedElements }) {
            newlyDeselectedElements.forEach(element => element.removeAttribute('data-selected'));
            newlySelectedElements.forEach(element => element.setAttribute('data-selected', ''));
        },

        onSelectionEnd({ svg, selectedElements: newSelectedElements }) {
            selectedElements = newSelectedElements;
            if (selectedElements.length > 0) {
                const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
                group.setAttribute('id', 'selected-group');

                selectedElements.forEach(el => {
                    group.appendChild(el.cloneNode(true));
                    el.remove();
                });

                svg.appendChild(group);

                gsap.registerPlugin(Draggable);
                Draggable.create(group, {
                    type: "xAxis,y",
                    onPress: function() {
                        dragSelectInstance.cancel();
                    },
                    onRelease: function() {
                        dragSelectInstance = initializeSvgDragSelect();
                    },
                });
            }
        },
    });
}

// const svgElement = document.getElementById("svgCanvas");
// let selectedElements = [];
// let dragSelectInstance = initializeSvgDragSelect();
