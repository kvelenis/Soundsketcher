export function drawYAxisScale(canvasHeight, minValue, maxValue, scale = "linear", topPadding = 50, labelFormatter = (v) => v.toFixed(0), invert = false) {
    const svgCanvas = document.getElementById("svgCanvas");

    const oldScale = svgCanvas.querySelector(".y-axis-scale");
    if (oldScale) svgCanvas.removeChild(oldScale);

    const scaleGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    scaleGroup.setAttribute("class", "y-axis-scale");
    scaleGroup.setAttribute("style", "user-select: none;");

    const numTicks = minValue !== maxValue ? 10 : 1;
    const mapY = (value, min, max) => {
        if (invert) {
            return map(value, min, max, topPadding, canvasHeight - topPadding);
        }
        return map(value, min, max, canvasHeight - topPadding, topPadding);
    };

    if (scale === "mel") {
        const melMin = hzToMel(minValue);
        const melMax = hzToMel(maxValue);
        for (let i = 0; i <= numTicks; i++) {
            const melValue = melMin + (i / numTicks) * (melMax - melMin);
            const value = melToHz(melValue);
            const y = mapY(melValue, melMin, melMax);
            drawTick(y, labelFormatter(value), scaleGroup);
        }
    } else if (scale === "log") {
        const logMin = Math.log10(minValue + 1);
        const logMax = Math.log10(maxValue + 1);

        for (let i = 0; i <= numTicks; i++) {
            const logValue = logMin + (i / numTicks) * (logMax - logMin);
            const value = Math.pow(10, logValue) - 1;
            const y = mapY(logValue, logMin, logMax);
            drawTick(y, labelFormatter(value), scaleGroup);
        }
    } else if (scale === "linear") {
        for (let i = 0; i <= numTicks; i++) {
            const value = minValue + (i / numTicks) * (maxValue - minValue);
            const y = mapY(value, minValue, maxValue);
            drawTick(y, labelFormatter(value), scaleGroup);
        }
    }

    svgCanvas.appendChild(scaleGroup);
}

export function drawTick(y, labelValue, group) {
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", 0);
    tick.setAttribute("x2", 10);
    tick.setAttribute("y1", y);
    tick.setAttribute("y2", y);
    tick.setAttribute("stroke", "black");
    group.appendChild(tick);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", 15);
    label.setAttribute("y", y + 3);
    label.setAttribute("font-size", "10");
    label.textContent = labelValue;
    group.appendChild(label);
}
