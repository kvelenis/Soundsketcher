export function createAdaptiveTexturePattern(draw, roughness, regionId) {
    const patternId = `texture-${regionId}`;
    const patternSize = 4 + (1 - roughness) * 12;
    const stripeWidth = 0.4 + roughness * 1.2;
    const dotRadius = 0.5 + roughness * 1.3;

    const pattern = draw.pattern(patternSize, patternSize, function (add) {
        if (roughness < 0.3) {
            for (let y = 0; y < patternSize; y += dotRadius * 2 + 1) {
                for (let xAxis = 0; xAxis < patternSize; xAxis += dotRadius * 2 + 1) {
                    add.circle(dotRadius * 2)
                        .center(xAxis, y)
                        .fill('#333');
                }
            }
        } else if (roughness < 0.65) {
            add.line(0, 0, patternSize, patternSize)
                .stroke({ color: '#444', width: stripeWidth });

            add.line(0, patternSize, patternSize, 0)
                .stroke({ color: '#444', width: stripeWidth });
        } else {
            for (let y = 0; y < patternSize; y += dotRadius * 2 + 2) {
                for (let xAxis = 0; xAxis < patternSize; xAxis += dotRadius * 2 + 2) {
                    add.circle(dotRadius * 2)
                        .center(xAxis, y)
                        .fill('#222');
                }
            }
            add.line(0, 0, patternSize, patternSize)
                .stroke({ color: '#555', width: stripeWidth * 0.8, opacity: 0.6 });

            add.line(0, patternSize, patternSize, 0)
                .stroke({ color: '#555', width: stripeWidth * 0.8, opacity: 0.6 });
        }
    });

    pattern.id(patternId);
    return pattern;
}

export function createHatchPattern(defs, patternId, roughnessRaw) {
    const roughness = Math.max(0, Math.min(1, roughnessRaw));
    const spacing = Math.max(3, 5 + (1 - roughness) * 20);

    const pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
    pattern.setAttribute("id", patternId);
    pattern.setAttribute("patternUnits", "userSpaceOnUse");
    pattern.setAttribute("width", spacing);
    pattern.setAttribute("height", spacing);

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", "0");
    line.setAttribute("y1", "0");
    line.setAttribute("x2", spacing.toString());
    line.setAttribute("y2", spacing.toString());
    line.setAttribute("stroke", "white");
    line.setAttribute("stroke-width", "0.5");

    pattern.appendChild(line);
    defs.appendChild(pattern);
}

export function createDotPattern(defs, patternId, roughnessRaw) {
    const roughness = Math.max(0, Math.min(1, roughnessRaw));
    const spacing = Math.max(4, 10 - roughness * 8);

    const pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
    pattern.setAttribute("id", patternId);
    pattern.setAttribute("patternUnits", "userSpaceOnUse");
    pattern.setAttribute("width", spacing);
    pattern.setAttribute("height", spacing);

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", spacing / 2);
    circle.setAttribute("cy", spacing / 2);
    circle.setAttribute("r", spacing * 0.1);
    circle.setAttribute("fill", "white");

    pattern.appendChild(circle);
    defs.appendChild(pattern);
}

export function createCrossHatchPattern(defs, patternId, roughnessRaw) {
    const roughness = Math.max(0, Math.min(1, roughnessRaw));
    const spacing = Math.max(4, 8 - roughness * 6);

    const pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
    pattern.setAttribute("id", patternId);
    pattern.setAttribute("patternUnits", "userSpaceOnUse");
    pattern.setAttribute("width", spacing);
    pattern.setAttribute("height", spacing);

    const line1 = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line1.setAttribute("x1", "0");
    line1.setAttribute("y1", "0");
    line1.setAttribute("x2", spacing.toString());
    line1.setAttribute("y2", spacing.toString());
    line1.setAttribute("stroke", "white");
    line1.setAttribute("stroke-width", "1");

    const line2 = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line2.setAttribute("x1", spacing.toString());
    line2.setAttribute("y1", "0");
    line2.setAttribute("x2", "0");
    line2.setAttribute("y2", spacing.toString());
    line2.setAttribute("stroke", "white");
    line2.setAttribute("stroke-width", "1");

    pattern.appendChild(line1);
    pattern.appendChild(line2);
    defs.appendChild(pattern);
}

export function createCirclePattern(draw, regionId, density = 5, radius = 2, color = '#555') {
    const patternSize = 10;
    const pattern = draw.pattern(patternSize, patternSize, add => {
        for (let xAxis = 0; xAxis < patternSize; xAxis += density) {
            for (let y = 0; y < patternSize; y += density) {
                add.circle(radius).move(xAxis, y).fill(color);
            }
        }
    }).id(`circle-pattern-${regionId}`);
    return pattern;
}

export function average(values) {
    if (!values || values.length === 0) return 0;
    const valid = values.filter(v => typeof v === 'number' && isFinite(v));
    return valid.length > 0 ? valid.reduce((sum, v) => sum + v, 0) / valid.length : 0;
}

export function catmullRomToPath(points) {
    if (points.length < 2) return '';

    let d = `M ${points[0][0]},${points[0][1]}`;

    for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i - 1] || points[i];
        const p1 = points[i];
        const p2 = points[i + 1];
        const p3 = points[i + 2] || p2;

        const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
        const cp1y = p1[1] + (p2[1] - p0[1]) / 6;

        const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
        const cp2y = p2[1] - (p3[1] - p1[1]) / 6;

        d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2[0]},${p2[1]}`;
    }

    return d + ' Z';
}

export function generatePolygonPath(cx, cy, corners, radius) {
    let path = "";

    for (let i = 0; i < corners; i++) {
        const angle = (2 * Math.PI * i) / corners - Math.PI / 2;
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        path += (i === 0 ? "M " : "L ") + x + " " + y + " ";
    }

    return `${path}Z`;
}

export function createPattern(density, cx, cy, radius, opacity = 1, fillColor = "white", strokeColor = "gray") {
    if (typeof createPattern.counter === "undefined") {
        createPattern.counter = 0;
    }

    const patternId = `pattern${createPattern.counter++}`;
    const pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
    const spacing = Math.max(3, radius * 1.4 * (1 - density * 0.8));

    pattern.setAttribute("id", patternId);
    pattern.setAttribute("width", spacing);
    pattern.setAttribute("height", spacing);
    pattern.setAttribute("patternUnits", "userSpaceOnUse");

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", 0);
    rect.setAttribute("y", 0);
    rect.setAttribute("width", spacing);
    rect.setAttribute("height", spacing);
    rect.setAttribute("fill", fillColor);
    rect.setAttribute("fill-opacity", opacity);
    pattern.appendChild(rect);

    const line1 = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line1.setAttribute("x1", 0);
    line1.setAttribute("y1", 0);
    line1.setAttribute("x2", spacing);
    line1.setAttribute("y2", spacing);
    line1.setAttribute("stroke", strokeColor);
    line1.setAttribute("stroke-width", 2);
    line1.setAttribute("stroke-opacity", density);

    const line2 = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line2.setAttribute("x1", 0);
    line2.setAttribute("y1", spacing);
    line2.setAttribute("x2", spacing);
    line2.setAttribute("y2", 0);
    line2.setAttribute("stroke", strokeColor);
    line2.setAttribute("stroke-width", 2);
    line2.setAttribute("stroke-opacity", density);

    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.appendChild(line1);
    group.appendChild(line2);
    pattern.appendChild(group);

    pattern.setAttribute(
        "patternTransform",
        `translate(${cx - spacing / 2}, ${cy - spacing / 2}) rotate(45, ${spacing / 2}, ${spacing / 2})`
    );

    return { pattern, patternId };
}
