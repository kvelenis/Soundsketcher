function drawClusterOverlays(clusters,features,svgContainer,canvasWidth,canvasHeight,maxDuration,fileIndex = 0)
{
    const draw = SVG().addTo('#svgCanvas').size('100%',canvasHeight).id(`audio-path-${fileIndex}`);

    if(!clusters)
    {
        return;
    }
    assignFeaturesToRegions(clusters,features);
    computeVisualDataForRegions(clusters,maxDuration,canvasWidth);

    const hue = Number(document.getElementById(`color-slider-${fileIndex}`).value);
    const loudness_threshold = document.getElementById("slider-gate").noUiSlider.get()/100;

    clusters.forEach(cluster =>
    {
        // === Cluster-level background blob ===
        const clusterStart = cluster["start_time"];
        const clusterEnd = cluster["end_time"];
        const clusterX = map(clusterStart,0,maxDuration,0,canvasWidth);
        const clusterWidth = map(clusterEnd,0,maxDuration,0,canvasWidth) - clusterX;

        const clusterFeatures = cluster["regions"].flatMap(r => r["features"] || []);
        const cluster_features_length = clusterFeatures.length;
        let avgY,avgBandwidth,avgLoudness;
        if(cluster_features_length)
        {
            avgY = clusterFeatures.reduce((sum,feature) => sum + (feature["visual"]?.yAxis ?? canvasHeight / 2),0) / cluster_features_length;
            avgBandwidth = clusterFeatures.reduce((sum,feature) => sum + (feature["visual"]?.lineLength ?? 50),0) / cluster_features_length;
            avgLoudness =  clusterFeatures.reduce((sum,feature) => sum + (feature["normalized_loudness"] ?? 0),0) / cluster_features_length;
        }
        else
        {
            avgY = canvasHeight / 2;
            avgBandwidth = 50;
            avgLoudness = 0;
        }

        if(avgLoudness > loudness_threshold)
        {
            drawVagueClusterBlob(draw,clusterFeatures,cluster.color,0.5,maxDuration,canvasWidth,canvasHeight,loudness_threshold);
        }
        else
        {
            console.log("Skipping blob due to silence.");
        }
            
        // drawLigetiEnvelopeBlob({
        //     draw,
        //     features: clusterFeatures,
        //     color: cluster.color,
        //     opacity: 0.35,
        //     maxDuration,
        //     canvasWidth,
        //     canvasHeight
        // });
            
        // drawLigetiClusterBlob({
        //     draw,
        //     features: clusterFeatures,
        //     x_axis: clusterX,
        //     width: clusterWidth,
        //     color: cluster.color,
        //     opacity: 0.35,
        //     canvasHeight
        // });
            
        // === Region-level overlays and features ===
        cluster["regions"].forEach(region =>
        {
            const startX = region["visual"].startX;
            const width = region["visual"].width;

            const overlay = document.createElementNS("http://www.w3.org/2000/svg","rect");
            overlay.setAttribute("x_axis",startX);
            overlay.setAttribute("y",0);
            overlay.setAttribute("width",width);
            overlay.setAttribute("height",canvasHeight);
            overlay.setAttribute("fill",cluster.color);
            overlay.setAttribute("fill-opacity",0);
            overlay.setAttribute("opacity",0.5);
            overlay.setAttribute("class","cluster-overlay");
            svgContainer.appendChild(overlay);

            // const avg = region["visual"].avg;
            drawUnifiedRegionPathFromVisual(svgContainer,region,maxDuration,canvasWidth,cluster.color,loudness_threshold,draw);

            const clipId = `region-clip-${region["start_time"].toFixed(2)}`;
            const clipPath = createClipPathFromRegion(draw,region,maxDuration,canvasWidth,canvasHeight,clipId);

            if(clipPath)
            {
                drawRoughnessSketch(draw,region,region["features"],canvasWidth,maxDuration,canvasHeight,clipPath,hue);
            }

            // ✅ Only draw gesture shapes if region is perceptually strong enough
            if (region["features"] && region["features"].length > 0) {
                const avgLoudness = region["features"].reduce((sum,feature) => sum + (feature["normalized_loudness"] ?? 0),0) / region["features"].length;

                if(avgLoudness > loudness_threshold)
                {
                    drawSubregionGestures(draw,region["features"],region,canvasWidth,maxDuration,canvasHeight,hue);
                }
                else
                {
                    console.log(`Skipping subregion gestures: too quiet, avg loudness = ${avgLoudness}`);
                }
            }
        });
    });
}

function assignFeaturesToRegions(clusters,features)
{
    clusters.forEach(cluster =>
    {
        cluster["regions"].forEach(region =>
        {
            region["features"] = features.filter(feature => feature["timestamp"] >= region["start_time"] && feature["timestamp"] <= region["end_time"]);
        });
    });
}

function computeVisualDataForRegions(clusters,maxDuration,canvasWidth)
{
    clusters.forEach(cluster =>
    {
        cluster["regions"].forEach(region =>
        {
            const startX = map(region["start_time"],0,maxDuration,0,canvasWidth);
            const endX = map(region["end_time"],0,maxDuration,0,canvasWidth);
            const width = endX - startX;
            const avg = computeAverageFeatures(region["features"] || []);
            region["visual"] = {startX,width,avg};
        });
    });
}

function computeAverageFeatures(features)
{
    const count = features.length || 1;
    const avg = Object.fromEntries(rawFeatureNames.map(key => [key,0]));
    features.forEach(feature =>
    {
        rawFeatureNames.forEach(name =>
        {
            avg[name] += feature[name] || 0;
        });
    });
    rawFeatureNames.forEach(name =>
    {
        avg[name] /= count;
    });
    return avg;
}

function drawVagueClusterBlob(draw,features,color,opacity,maxDuration,canvasWidth,canvasHeight,loudness_threshold)
{
    if (!features || features.length < 2) return;
  
    const padding = 40;
    const stride = Math.max(1, Math.floor(features.length / 20));
    const minLineLength = 20;
  
    const topPoints = [];
    const bottomPoints = [];
  
    // === Compute cluster Y stats for clamping ===
    const ys = features.map(f => f["visual"]?.yAxis ?? canvasHeight / 2);
    const ysSorted = [...ys].sort((a, b) => a - b);
    const lowerPercentile = ysSorted[Math.floor(ysSorted.length * 0.1)];
    const upperPercentile = ysSorted[Math.floor(ysSorted.length * 0.9)];
    const medianY = ysSorted[Math.floor(ysSorted.length / 2)];

    const clampMin = lowerPercentile;
    const clampMax = upperPercentile;

    let lastY = medianY;
    let lastH = 50;
  
    for (let i = 0; i < features.length; i += stride) {
      const f = features[i];
      const loudness = f["normalized_loudness"] ?? 0;
  
      let y, h;
  
      if (loudness > loudness_threshold) {
        y = f["visual"]?.yAxis ?? lastY;
        h = f["visual"]?.lineLength ?? lastH;
        lastY = y;
        lastH = h;
      } else {
        y = lastY;
        h = Math.max(lastH * 0.8, minLineLength);
      }
  
      // ✅ Clamp Y-axis to stay inside reasonable blob band
      y = clamp(y, clampMin, clampMax);
  
      const x_axis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
  
      topPoints.push([x_axis, y - h / 2 - padding]);
      bottomPoints.unshift([x_axis, y + h / 2 + padding]);
    }
  
    const blobPoints = topPoints.concat(bottomPoints, [topPoints[0]]);
  
    const xValues = blobPoints.map(p => p[0]);
    const yValues = blobPoints.map(p => p[1]);
    const minX = Math.min(...xValues),
      maxX = Math.max(...xValues);
    const minYShape = Math.min(...yValues),
      maxYShape = Math.max(...yValues);
  
    const roundedBox = [
      [minX, minYShape],
      [maxX, minYShape],
      [maxX, maxYShape],
      [minX, maxYShape],
      [minX, minYShape]
    ];
  
    const vaguePath = flubber.interpolate(
      flubber.toPathString(roundedBox),
      flubber.toPathString(blobPoints),
      { maxSegmentLength: 10 }
    )(0.8);
  
    draw.path(vaguePath)
      .fill(color)
      .stroke({ width: 0 })
      .opacity(opacity)
      .attr({
        'fill-opacity': opacity,
        'vector-effect': 'non-scaling-stroke',
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
        'filter': 'url(#blur)'
      });
  }

function drawUnifiedRegionPathFromVisual(
    svg,
    region,
    maxDuration,
    canvasWidth,
    baseColor = "hsl(0, 70%, 50%)",
    loudnessThreshold = 0,
    draw
  ) {
    const regionFrames = region["features"];
  
    if (!regionFrames || regionFrames.length < 2) return;
  
    const avgRoughness = regionFrames.reduce(
      (sum, f) => sum + (f["visual"]?.roughness ?? 0.3),
      0
    ) / regionFrames.length;
    const pattern = createAdaptiveTexturePattern(
      draw,
      avgRoughness,
      `${region["start_time"]}-${region["end_time"]}`
    );
    const grainPattern = createGrainTexturePattern(
      draw,
      avgRoughness,
      `${region["start_time"]}-${region["end_time"]}`
    );
  
    // === Robust stats ===
    const ys = regionFrames.map(f => f["visual"]?.yAxis ?? 0);
    const ysSorted = [...ys].sort((a, b) => a - b);
    const medianY = ysSorted[Math.floor(ysSorted.length / 2)];
    const p10 = ysSorted[Math.floor(ysSorted.length * 0.1)];
    const p90 = ysSorted[Math.floor(ysSorted.length * 0.9)];
  
    const lengths = regionFrames.map(f => f["visual"]?.lineLength ?? 60);
    const lengthsSorted = [...lengths].sort((a, b) => a - b);
    const medianLength = lengthsSorted[Math.floor(lengthsSorted.length / 2)];
    const p10Length = lengthsSorted[Math.floor(lengthsSorted.length * 0.1)];
    const p90Length = lengthsSorted[Math.floor(lengthsSorted.length * 0.9)];
  
    const topPoints = [];
    const bottomPoints = [];
  
    for (let i = 0; i < regionFrames.length; i++) {
      const f = regionFrames[i];
      if (!f["visual"]) continue;
  
      const x_axis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
  
      // === Clamp yAxis and lineLength to robust range ===
      let y = f["visual"].yAxis;
      y = clamp(y, p10, p90);
  
      let height = f["visual"].lineLength ?? medianLength;
      height = clamp(height, p10Length, p90Length);
  
      const mod = f["visual"].mod ?? 0;
  
      const yTop = y - height / 2 - mod;
      const yBot = y + height / 2 + mod;
  
      topPoints.push([x_axis, yTop]);
      bottomPoints.unshift([x_axis, yBot]);
    }
  
    if (topPoints.length < 2 || bottomPoints.length < 2) return;
  
    const fullPoints = [...topPoints, ...bottomPoints];
  
    const pathData = catmullRomToPath(fullPoints); // Smooth closed path
  
    const avgLoudness =
      regionFrames.reduce((acc, f) => acc + (f["normalized_loudness"] ?? 0), 0) /
      regionFrames.length;
    const perceivedOpacity = loudness =>
      Math.max(0, Math.min(1, (loudness - 0.2) / (loudnessThreshold - 0.2)));
    const alpha = perceivedOpacity(avgLoudness);
  
    draw.path(pathData)
      .fill({ color: "#000", opacity: 0 })
      .opacity(alpha)
      .stroke({ width: 0.5, color: "#000", opacity: 0.1 });
  }

  function createClipPathFromRegion(draw, region, maxDuration, canvasWidth, canvasHeight, clipId) {
    const regionFrames = region["features"];
    if (!regionFrames || regionFrames.length < 2) return null;

    const topPoints = [], bottomPoints = [];

    regionFrames.forEach(f => {
        if (!f["visual"]) return;

        const x_axis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
        const y = f["visual"].yAxis;
        const h = f["visual"].lineLength ?? 60;
        const mod = f["visual"].mod ?? 0;

        topPoints.push([x_axis, y - h / 2 - mod]);
        bottomPoints.unshift([x_axis, y + h / 2 + mod]);
    });

    const allPoints = topPoints.concat(bottomPoints);
    const pathData = catmullRomToPath(allPoints);

    const clipPath = draw.clip().id(clipId);
    clipPath.path(pathData).fill('#000'); // fill color is irrelevant for clip

    return clipPath; // ✅ RETURN this
}

function createGrainTexturePattern(draw, roughness, regionId) {
    const patternId = `grain-${regionId}`;

    const patternSize = 14 + (1 - roughness) * 12;
    const density = Math.floor(40 + roughness * 120);
    const baseOpacity = 0.08 + roughness * 0.4;
    const maxRadius = 1.8 - roughness * 1.2;

    const pattern = draw.pattern(patternSize, patternSize, function (add) {
        if (roughness < 0.2) {
            // ✨ Smooth fill — fog-like micro blur
            add.rect(patternSize, patternSize)
                .fill('#333')
                .opacity(0.02 + (1 - roughness) * 0.06);
        }

        for (let i = 0; i < density; i++) {
            const x_axis = Math.random() * patternSize;
            const y = Math.random() * patternSize;
            const r = Math.random() * maxRadius + 0.4;

            add.circle(r * 2)
                .center(x_axis, y)
                .fill('#222')
                .opacity(baseOpacity);
        }
    });

    pattern.id(patternId);
    return pattern;
}

function drawRoughnessSketch(draw, region, features, canvasWidth, maxDuration, canvasHeight, clipElement,hue = 0) {
    const group = draw.group().id(`sketch-${region["start_time"].toFixed(2)}`).clipWith(clipElement);

    const maxSampleFrames = 150;
    const step = Math.ceil(features.length / maxSampleFrames);
    const MAX_TOTAL_LINES = 55000;

    const roughnessSamples = [];

    const sketch_color = hslToHex(hue,100,50);

    // Step 1: Collect graininess per frame using dashArray
    for (let i = 0; i < features.length; i += step) {
        const f = features[i];
        const dashArray = f["visual"]?.dashArray ?? 0;
        const lineWidth = f["visual"]?.lineWidth ?? 0.3;


        // Map: low dashArray → dense texture, high dashArray → sparse
        const graininess = map(dashArray, 0, 10, 4000, 10); // up to 120 lines at smoothness

        roughnessSamples.push({ f, lineWidth, graininess });
    }

    // Step 2: Scaling factor if we exceed total line limit
    const totalLines = roughnessSamples.reduce((sum, d) => sum + Math.floor(d.graininess), 0);
    const scaleFactor = totalLines > MAX_TOTAL_LINES ? MAX_TOTAL_LINES / totalLines : 1;

    let globalPathData = '';
    let opacitySum = 0;
    let lineCount = 0;

    // Step 3: Generate path lines for sketch texture
    roughnessSamples.forEach(({ f, lineWidth, graininess }) => {
        const x_axis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
        const numLines = Math.floor(graininess * scaleFactor);
        const lineOpacity = map(lineWidth, 0, 5, 0.05, 1);


        for (let j = 0; j < numLines; j++) {
            const offset = (Math.random() - 0.5) * 16;
            const y1 = 100 + Math.random() * (canvasHeight - 200);
            const y2 = y1 + Math.random() * 40 - 20;
            globalPathData += `M ${x_axis + offset} ${y1} L ${x_axis + offset} ${y2} `;
            opacitySum += lineOpacity;
            lineCount++;
        }
    });

    if (lineCount > 0) {
        const avgOpacity = Math.min(1, opacitySum / lineCount);
        group.path(globalPathData.trim())
             .stroke({ color: sketch_color, width: avgOpacity, opacity: avgOpacity });
    }
}

function drawSubregionGestures(draw, features, region, canvasWidth, maxDuration, canvasHeight,hue = 0) {
    if (!features || features.length === 0) return;

    // Sort features by loudness (descending)
    const sortedByLoudness = [...features].sort((a, b) => (b.loudness ?? 0) - (a.loudness ?? 0));
    const top3 = sortedByLoudness.slice(0, 3);
    const next2 = sortedByLoudness.slice(3, 5); // Optional

    const stroke_color = hslToHex(hue,0,0);

    // === TRIANGLES for top 3 ===
    top3.forEach(f => {
        const x_axis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
        const y = f["visual"]?.yAxis ?? canvasHeight / 2;
        const rawLineWidth = f["visual"]?.lineWidth ?? 1;
        const size = map(rawLineWidth, 1, 5, 5, 20);
        const angle = f["visual"]?.angle ?? 0;

        const fill_color = hslToHex(hue,f["visual"].colorSaturation,f["visual"].colorLightness);

        // Define an upward-pointing triangle centered at (x_axis, y)
        const halfSize = size / 2;
        const points = [
            [x_axis, y - halfSize],
            [x_axis - halfSize * Math.sin(Math.PI / 3), y + halfSize / 2],
            [x_axis + halfSize * Math.sin(Math.PI / 3), y + halfSize / 2]
        ];

        draw.polygon(points.map(p => p.join(',')).join(' '))
            .fill(fill_color)
            .stroke({ color: stroke_color, width: 1, opacity: 1 })
            .rotate((angle * 180) / Math.PI, x_axis, y);
    });

    // === CIRCLES for next 2 (optional) ===
    next2.forEach(f => {
        const x_axis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
        const y = f["visual"]?.yAxis ?? canvasHeight / 2;
        const rawLineWidthForRadius = f["visual"]?.lineWidth ?? 1;
        const radius = map(rawLineWidthForRadius, 1, 5, 1, 5);

        const rotation = (f["visual"]?.angle ?? 0) * (180 / Math.PI);

        const fill_color = hslToHex(hue,f["visual"].colorSaturation,f["visual"].colorLightness);

        draw.circle(radius * 2)
            .center(x_axis, y)
            .fill(fill_color)
            .stroke({ color: stroke_color, width: 1, opacity: 1 })
            .rotate(rotation, x_axis, y);
    });
}

function drawLigetiClusterBlob({ draw, features, x_axis, width, color, opacity, canvasHeight }) {
    if (!features || features.length === 0) return;

    const padding = 100;
    const numPoints = 20;

    // Compute min/max y based on feature positions
    const yValues = features.map(f => f.visual?.yAxis ?? canvasHeight / 2);
    const minY = Math.min(...yValues) - padding;
    const maxY = Math.max(...yValues) + padding;

    const centerX = x_axis + width / 2;
    const points = [];

    for (let i = 0; i < numPoints; i++) {
        const angle = (Math.PI * 2 * i) / numPoints;
        const radiusX = width / 2 + (Math.random() - 0.5) * width * 0.2;
        const radiusY = (maxY - minY) / 2 + (Math.random() - 0.5) * (maxY - minY) * 0.2;
        const px = centerX + Math.cos(angle) * radiusX;
        const py = (minY + maxY) / 2 + Math.sin(angle) * radiusY;

        points.push([px, py]);
    }

    // Close the shape
    points.push(points[0]);

    const pathStr = flubber.toPathString(points);

    draw.path(pathStr)
        .fill(color)
        .stroke({ width: 0 })
        .opacity(opacity)
        .attr({
            'fill-opacity': opacity,
            'vector-effect': 'non-scaling-stroke',
            'stroke-linejoin': 'round',
            'stroke-linecap': 'round',
            'stroke-width': 0,
            'filter': 'url(#blur)'
        });
}