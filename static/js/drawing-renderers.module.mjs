import {
    createPattern,
    generatePolygonPath,
} from "./drawing-patterns.module.mjs?v=frontend-migration-121";

export function appendLineSketchFrame(pathGroup, frame, options) {
    const {
        xAxis,
        yAxis,
        lineLength,
        lineWidth,
        colorSaturation,
        colorLightness,
        angle,
        dashArray,
    } = frame;
    let previousDots = options.previousDots;

    const x1 = xAxis - (lineLength / 2) * Math.cos(angle);
    const y1 = yAxis + (lineLength / 2) * Math.sin(angle);
    const x2 = xAxis + (lineLength / 2) * Math.cos(angle);
    const y2 = yAxis - (lineLength / 2) * Math.sin(angle);

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M${x1},${y1} L${x2},${y2}`);
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");

    if (dashArray == 0) {
        path.removeAttribute("stroke-dasharray");
    } else {
        const linepx = Math.max(dashArray, 0.5);
        const spacepx = linepx;
        path.setAttribute("stroke-dasharray", `${linepx},${spacepx}`);
    }

    path.setAttribute("stroke", `hsl(${options.colorHue}, ${colorSaturation}%, ${colorLightness}%)`);
    path.setAttribute("stroke-width", lineWidth);
    path.setAttribute("fill", "none");
    path.setAttribute("data-features", options.featureDescription);
    pathGroup.appendChild(path);

    if (options.isThresholdCircleEnabled) {
        const hue = options.hue1;
        let radius = lineWidth / 2 + 0;

        let circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", xAxis);
        circle.setAttribute("cy", yAxis);
        circle.setAttribute("r", radius);
        circle.setAttribute("fill", `hsl(${hue}, ${colorSaturation}%,${colorLightness}%)`);
        circle.setAttribute("data-features", options.featureDescription);
        pathGroup.appendChild(circle);

        const width = 4;
        radius = lineLength / 2 + lineWidth / 2 - width / 2;
        circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", xAxis);
        circle.setAttribute("cy", yAxis);
        circle.setAttribute("r", radius);
        circle.setAttribute("stroke", `hsl(${hue}, ${colorSaturation}%,${colorLightness}%)`);
        circle.setAttribute("stroke-width", width);
        circle.setAttribute("fill", "none");
        circle.setAttribute("data-features", options.featureDescription);
        pathGroup.appendChild(circle);
    }

    if (options.isJoinPathsEnabled) {
        const numDots = 10;
        const scatterRange = lineLength;
        const currentDots = [];

        for (let j = 0; j < numDots; j++) {
            const offset = j * scatterRange / (numDots - 1) - scatterRange / 2;
            const scatteredX = xAxis + offset * Math.cos(angle);
            const scatteredY = yAxis + offset * Math.sin(angle);
            currentDots.push({ xAxis: scatteredX, y: scatteredY });
        }

        if (previousDots.length > 0) {
            let hueDiff = (options.hue2 - options.hue1 + 360) % 360;
            if (hueDiff > 180) {
                hueDiff -= 360;
            }

            for (let j = 0; j < numDots; j++) {
                const hue = (options.hue1 + j * hueDiff / (numDots - 1) + 360) % 360;
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", previousDots[j].xAxis);
                line.setAttribute("y1", previousDots[j].y);
                line.setAttribute("x2", currentDots[j].xAxis);
                line.setAttribute("y2", currentDots[j].y);
                line.setAttribute("stroke", `hsl(${hue}, 100%,50%)`);
                line.setAttribute("stroke-width", 1);
                pathGroup.appendChild(line);
            }
        }

        previousDots = currentDots;
    }

    return previousDots;
}

export function appendPolygonFrame(pathGroup, defs, frame, options) {
    const {
        xAxis,
        yAxis,
        lineLength,
        lineWidth,
        colorSaturation,
        colorLightness,
        angleDegrees,
        dashArray,
    } = frame;

    const polygonCorners = Math.max(3, Math.floor(lineWidth));
    const polygonRadius = lineLength;
    const polygonPath = generatePolygonPath(xAxis, yAxis, polygonCorners, polygonRadius);
    const strokeWidth = 1;
    const opacity = 1;
    const skewX = angleDegrees - 90;
    const texture = dashArray / 100;

    const fillColor = `hsl(${options.colorHue}, ${colorSaturation}%, ${colorLightness}%)`;
    const strokeColor = `hsl(${options.colorHue}, ${colorSaturation}%, ${Math.max(colorLightness - 20, 0)}%)`;
    const patternColor = `hsl(${options.colorHue}, ${Math.max(colorSaturation - 20, 0)}%, ${Math.min(colorLightness + 20, 100)}%)`;

    const { pattern, patternId } = createPattern(texture, xAxis, yAxis, polygonRadius, opacity, fillColor, patternColor);
    defs.appendChild(pattern);

    const polygonContainer = document.createElementNS("http://www.w3.org/2000/svg", "g");
    polygonContainer.setAttribute("transform", `translate(${xAxis},${yAxis}) skewX(${skewX}) translate(${-xAxis},${-yAxis})`);

    const polygonElement = document.createElementNS("http://www.w3.org/2000/svg", "path");
    polygonElement.setAttribute("d", polygonPath);
    polygonElement.setAttribute("fill", `url(#${patternId})`);
    polygonElement.setAttribute("stroke", strokeColor);
    polygonElement.setAttribute("stroke-width", strokeWidth);
    polygonElement.setAttribute("stroke-opacity", opacity);
    polygonElement.setAttribute("data-features", options.featureDescription);

    polygonContainer.appendChild(polygonElement);
    pathGroup.appendChild(polygonContainer);
}
