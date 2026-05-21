export function drawClusterOverlays(clusters,features,svgContainer,canvasWidth,canvasHeight,maxDuration,fileIndex = 0,objectifierState = null)
{
    const draw = SVG().addTo('#svgCanvas').size('100%',canvasHeight).id(`audio-path-${fileIndex}`);

    if(!clusters)
    {
        return;
    }
    hydrateObjectifierClusterLabels(objectifierState,fileIndex);
    hydrateObjectifierRegionEdits(objectifierState,fileIndex,svgContainer);
    hydrateObjectifierVisibility(objectifierState,fileIndex,svgContainer);
    svgContainer.dataset.objectifierFileIndex = String(fileIndex);
    applyObjectifierRegionOverrides(clusters,objectifierState || currentObjectifierState(fileIndex));
    assignFeaturesToRegions(clusters,features);
    computeVisualDataForRegions(clusters,maxDuration,canvasWidth);

    const hue = Number(document.getElementById(`color-slider-${fileIndex}`).value);
    const loudness_threshold = document.getElementById("slider-gate").noUiSlider.get()/100;
    const detailMode = objectifierDetailMode();
    const showTexture = detailMode === "detailed";
    const showGestures = detailMode === "detailed";
    const showLegend = detailMode !== "simple";
    const enableSelection = detailMode !== "simple";

    let regionCounter = 0;
    clusters.forEach(cluster =>
    {
        // === Region-level overlays and features ===
        cluster["regions"].forEach(region =>
        {
            regionCounter += 1;
            const regionKey = `objectifier-region-${fileIndex}-${regionCounter}`;
            region["_objectifierKey"] = regionKey;
            region["_objectifierNumber"] = regionCounter;
            region["_objectifierClusterLabel"] = region["label"] ?? cluster["label"];
            region["_objectifierFileIndex"] = fileIndex;

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
            overlay.setAttribute("class","cluster-overlay objectifier-region-hit");
            attachRegionDataset(overlay,region,cluster);
            svgContainer.appendChild(overlay);

            drawUnifiedRegionPathFromVisual(svgContainer,region,maxDuration,canvasWidth,cluster.color,loudness_threshold,draw,cluster);

            const clipId = `region-clip-${region["start_time"].toFixed(2)}`;
            const clipPath = createClipPathFromRegion(draw,region,maxDuration,canvasWidth,canvasHeight,clipId);

            if(showTexture && clipPath)
            {
                drawRoughnessSketch(draw,region,region["features"],canvasWidth,maxDuration,canvasHeight,clipPath,hue);
            }

            // ✅ Only draw gesture shapes if region is perceptually strong enough
            if (showGestures && region["features"] && region["features"].length > 0) {
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

    drawObjectifierInfoLayer(clusters,svgContainer,canvasWidth,canvasHeight,maxDuration,{ showLegend });
    if(enableSelection)
    {
        wireObjectifierInteractions(svgContainer);
    }
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

function objectifierDetailMode()
{
    const mode = document.getElementById("objectifierDetailMode")?.value;
    return ["simple","balanced","detailed"].includes(mode) ? mode : "balanced";
}

function objectifierClusterLabels()
{
    window.SoundSketcher = window.SoundSketcher || {};
    window.SoundSketcher.objectifierClusterLabels = window.SoundSketcher.objectifierClusterLabels || {};
    return window.SoundSketcher.objectifierClusterLabels;
}

function hydrateObjectifierClusterLabels(objectifierState,fileIndex)
{
    const state = objectifierState || currentObjectifierState(fileIndex);
    const labels = state?.cluster_labels;
    window.SoundSketcher = window.SoundSketcher || {};
    window.SoundSketcher.objectifierClusterLabels = labels && typeof labels === "object" ? labels : {};
    if(state && !state.cluster_labels)
    {
        state.cluster_labels = window.SoundSketcher.objectifierClusterLabels;
    }
}

function currentObjectifierState(fileIndex)
{
    return window.SoundSketcher?.state?.globalAudioData?.data?.[Number(fileIndex) || 0] || null;
}

function currentObjectifierFileRef(fileIndex)
{
    const audioData = window.SoundSketcher?.state?.globalAudioData;
    const index = Number(fileIndex) || 0;
    return {
        fileData: audioData?.data?.[index] || null,
        filename: audioData?.filename?.[index] || "",
        hash: audioData?.hash?.[index] || "",
    };
}

function objectifierClusterPlayback()
{
    window.SoundSketcher = window.SoundSketcher || {};
    window.SoundSketcher.objectifierClusterPlayback = window.SoundSketcher.objectifierClusterPlayback || {
        token: 0,
        clusterLabel: null
    };
    return window.SoundSketcher.objectifierClusterPlayback;
}

function objectifierVisibilityState(svgContainer)
{
    svgContainer._objectifierVisibility = svgContainer._objectifierVisibility || {
        hiddenClusters: new Set(),
        onlyCluster: null
    };
    return svgContainer._objectifierVisibility;
}

function objectifierRegionEditState(svgContainer)
{
    svgContainer._objectifierRegionEdits = svgContainer._objectifierRegionEdits || {
        deletedRegionKeys: new Set()
    };
    return svgContainer._objectifierRegionEdits;
}

function hydrateObjectifierRegionEdits(objectifierState,fileIndex,svgContainer)
{
    const state = objectifierState || currentObjectifierState(fileIndex);
    const deletedRegions = Array.isArray(state?.deleted_regions) ? state.deleted_regions : [];
    const boundaryOverrides = state?.region_overrides && typeof state.region_overrides === "object"
        ? state.region_overrides
        : {};
    svgContainer._objectifierRegionEdits = {
        deletedRegionKeys: new Set(deletedRegions.map(regionKey => String(regionKey))),
        boundaryOverrides: new Map(
            Object.entries(boundaryOverrides).map(([regionKey,override]) => [
                String(regionKey),
                {
                    start_time: Number(override?.start_time),
                    end_time: Number(override?.end_time)
                }
            ])
        )
    };
    if(state && !Array.isArray(state.deleted_regions))
    {
        state.deleted_regions = [];
    }
    if(state && (!state.region_overrides || typeof state.region_overrides !== "object"))
    {
        state.region_overrides = {};
    }
}

function hydrateObjectifierVisibility(objectifierState,fileIndex,svgContainer)
{
    const state = objectifierState || currentObjectifierState(fileIndex);
    const hiddenClusters = Array.isArray(state?.hidden_clusters) ? state.hidden_clusters : [];
    const onlyCluster = typeof state?.only_cluster === "string" && state.only_cluster ? state.only_cluster : null;
    svgContainer._objectifierVisibility = {
        hiddenClusters: new Set(hiddenClusters.map(c => String(c))),
        onlyCluster
    };
    if(state && !Array.isArray(state.hidden_clusters))
    {
        state.hidden_clusters = [];
    }
    if(state && state.only_cluster === undefined)
    {
        state.only_cluster = null;
    }
}

function regionEditKeyFromParts(clusterLabel,start,end)
{
    const label = String(clusterLabel ?? "?");
    const startText = Number(start).toFixed(3);
    const endText = Number(end).toFixed(3);
    return `${label}::${startText}::${endText}`;
}

function regionEditKeyFromDataset(dataset)
{
    return dataset?.regionEditKey || regionEditKeyFromParts(
        dataset?.clusterLabel,
        dataset?.startSeconds,
        dataset?.endSeconds
    );
}

function applyObjectifierRegionOverrides(clusters,objectifierState)
{
    const overrides = objectifierState?.region_overrides && typeof objectifierState.region_overrides === "object"
        ? objectifierState.region_overrides
        : {};
    clusters.forEach(cluster =>
    {
        const regions = Array.isArray(cluster["regions"]) ? cluster["regions"] : [];
        regions.forEach(region =>
        {
            const label = region["label"] ?? cluster["label"];
            const originalStart = Number(region["_objectifierOriginalStart"] ?? region["start_time"]);
            const originalEnd = Number(region["_objectifierOriginalEnd"] ?? region["end_time"]);
            const editKey = region["_objectifierEditKey"] || regionEditKeyFromParts(label,originalStart,originalEnd);
            region["_objectifierEditKey"] = editKey;
            region["_objectifierOriginalStart"] = originalStart;
            region["_objectifierOriginalEnd"] = originalEnd;
            const override = overrides[editKey];
            const start = Number(override?.start_time);
            const end = Number(override?.end_time);
            if(Number.isFinite(start) && Number.isFinite(end) && start >= 0 && end > start)
            {
                region["start_time"] = start;
                region["end_time"] = end;
            }
        });
    });
}

function customClusterName(label)
{
    return objectifierClusterLabels()[String(label)] || "";
}

function clusterDisplayName(label)
{
    const labelText = String(label);
    const custom = customClusterName(labelText);
    return custom ? `${custom} (${labelText})` : `Cluster ${labelText}`;
}

function drawObjectifierInfoLayer(clusters,svgContainer,canvasWidth,canvasHeight,maxDuration,options = {})
{
    const timelineRegions = collectTimelineRegions(clusters);
    if(!timelineRegions.length) return;

    const layer = document.createElementNS("http://www.w3.org/2000/svg","g");
    layer.setAttribute("class","objectifier-info-layer");
    svgContainer.appendChild(layer);

    drawRegionTimeline(layer,timelineRegions,canvasWidth,canvasHeight,maxDuration);
    if(options.showLegend !== false)
    {
        drawClusterLegend(layer,timelineRegions,canvasWidth);
    }
}

function collectTimelineRegions(clusters)
{
    const timeline = [];
    clusters.forEach(cluster =>
    {
        const regions = Array.isArray(cluster["regions"]) ? cluster["regions"] : [];
        regions.forEach(region =>
        {
            const start = Number(region["start_time"]);
            const end = Number(region["end_time"]);
            if(!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
            const label = region["label"] ?? cluster["label"];
            timeline.push({
                start,
                end,
                label,
                color: cluster["color"] || "hsl(210, 80%, 52%)",
                topTerms: cluster["top_terms"] || region["top_terms"] || [],
                key: region["_objectifierKey"],
                editKey: region["_objectifierEditKey"] || regionEditKeyFromParts(label,start,end),
                number: region["_objectifierNumber"],
                fileIndex: region["_objectifierFileIndex"],
                avg: region["visual"]?.avg || {}
            });
        });
    });
    return timeline.sort((a,b) => a.start - b.start);
}

function drawRegionTimeline(layer,timelineRegions,canvasWidth,canvasHeight,maxDuration)
{
    const bandHeight = 22;
    const bandY = Math.max(8,canvasHeight - bandHeight - 8);
    const background = document.createElementNS("http://www.w3.org/2000/svg","rect");
    background.setAttribute("x",0);
    background.setAttribute("y",bandY - 4);
    background.setAttribute("width",canvasWidth);
    background.setAttribute("height",bandHeight + 8);
    background.setAttribute("fill","rgba(255,255,255,0.78)");
    background.setAttribute("stroke","rgba(0,0,0,0.22)");
    background.setAttribute("stroke-width","1");
    layer.appendChild(background);

    timelineRegions.forEach((region,index) =>
    {
        const startX = map(region.start,0,maxDuration,0,canvasWidth);
        const endX = map(region.end,0,maxDuration,0,canvasWidth);
        const x = startX + 1;
        const width = Math.max(1,endX - startX - 2);
        const rect = document.createElementNS("http://www.w3.org/2000/svg","rect");
        rect.setAttribute("x",x);
        rect.setAttribute("y",bandY);
        rect.setAttribute("width",width);
        rect.setAttribute("height",bandHeight);
        rect.setAttribute("rx",2);
        rect.setAttribute("fill",region.color);
        rect.setAttribute("fill-opacity","0.9");
        rect.setAttribute("stroke","rgba(0,0,0,0.36)");
        rect.setAttribute("stroke-width","0.7");
        rect.setAttribute("class","objectifier-timeline-region");
        rect.setAttribute("cursor","pointer");
        rect.dataset.clusterLabel = String(region.label);
        attachTimelineDataset(rect,region);

        const title = document.createElementNS("http://www.w3.org/2000/svg","title");
        title.textContent = regionTooltipText(region,index);
        rect.appendChild(title);
        layer.appendChild(rect);

        if(width > 42)
        {
            const text = document.createElementNS("http://www.w3.org/2000/svg","text");
            text.setAttribute("class","objectifier-timeline-label");
            text.setAttribute("x",x + width / 2);
            text.setAttribute("y",bandY + 15);
            text.setAttribute("text-anchor","middle");
            text.setAttribute("font-size","11");
            text.setAttribute("font-family","system-ui, sans-serif");
            text.setAttribute("fill",textColorForFill(region.color));
            text.setAttribute("pointer-events","none");
            text.dataset.clusterLabel = String(region.label);
            text.dataset.regionKey = region.key;
            text.dataset.regionEditKey = region.editKey;
            text.textContent = String(region.label);
            layer.appendChild(text);
        }
    });
}

function drawClusterLegend(layer,timelineRegions,canvasWidth)
{
    const summaries = summarizeClusters(timelineRegions);
    const visibleRows = summaries.slice(0,10);
    if(!visibleRows.length) return;

    const rowHeight = 18;
    const padding = 8;
    const width = 190;
    const height = padding * 2 + 18 + visibleRows.length * rowHeight;
    const x = Math.max(8,canvasWidth - width - 12);
    const y = 12;

    const panel = document.createElementNS("http://www.w3.org/2000/svg","g");
    panel.setAttribute("class","objectifier-cluster-legend");
    layer.appendChild(panel);

    const background = document.createElementNS("http://www.w3.org/2000/svg","rect");
    background.setAttribute("x",x);
    background.setAttribute("y",y);
    background.setAttribute("width",width);
    background.setAttribute("height",height);
    background.setAttribute("rx",6);
    background.setAttribute("fill","rgba(255,255,255,0.84)");
    background.setAttribute("stroke","rgba(0,0,0,0.24)");
    background.setAttribute("stroke-width","1");
    panel.appendChild(background);

    const title = document.createElementNS("http://www.w3.org/2000/svg","text");
    title.setAttribute("x",x + padding);
    title.setAttribute("y",y + 18);
    title.setAttribute("font-size","12");
    title.setAttribute("font-weight","700");
    title.setAttribute("font-family","system-ui, sans-serif");
    title.setAttribute("fill","#111");
    title.textContent = "Objectifier clusters";
    panel.appendChild(title);

    visibleRows.forEach((summary,index) =>
    {
        const rowY = y + padding + 24 + index * rowHeight;
        const row = document.createElementNS("http://www.w3.org/2000/svg","g");
        row.setAttribute("class","objectifier-legend-row");
        row.setAttribute("cursor","pointer");
        row.dataset.clusterLabel = summary.label;
        row.dataset.clusterName = customClusterName(summary.label);
        row.dataset.regionCount = String(summary.count);
        row.dataset.duration = formatSeconds(summary.duration);
        panel.appendChild(row);

        const rowHit = document.createElementNS("http://www.w3.org/2000/svg","rect");
        rowHit.setAttribute("x",x + padding - 3);
        rowHit.setAttribute("y",rowY - 14);
        rowHit.setAttribute("width",width - padding * 2 + 6);
        rowHit.setAttribute("height",rowHeight);
        rowHit.setAttribute("fill","transparent");
        row.appendChild(rowHit);

        const swatch = document.createElementNS("http://www.w3.org/2000/svg","rect");
        swatch.setAttribute("x",x + padding);
        swatch.setAttribute("y",rowY - 10);
        swatch.setAttribute("width",10);
        swatch.setAttribute("height",10);
        swatch.setAttribute("rx",2);
        swatch.setAttribute("fill",summary.color);
        row.appendChild(swatch);

        const label = document.createElementNS("http://www.w3.org/2000/svg","text");
        label.setAttribute("class","objectifier-legend-label");
        label.setAttribute("x",x + padding + 16);
        label.setAttribute("y",rowY);
        label.setAttribute("font-size","11");
        label.setAttribute("font-family","system-ui, sans-serif");
        label.setAttribute("fill","#111");
        label.textContent = clusterLegendText(summary.label,summary.count,formatSeconds(summary.duration));
        row.appendChild(label);
    });
}

function clusterLegendText(label,count,duration)
{
    return `${clusterDisplayName(label)}: ${count} regions, ${duration}`;
}

function applyClusterLabels(svgContainer)
{
    svgContainer.querySelectorAll(".objectifier-legend-row").forEach(row =>
    {
        row.dataset.clusterName = customClusterName(row.dataset.clusterLabel);
        const label = row.querySelector(".objectifier-legend-label");
        if(label)
        {
            label.textContent = clusterLegendText(
                row.dataset.clusterLabel,
                row.dataset.regionCount || "0",
                row.dataset.duration || "n/a"
            );
        }
    });

    svgContainer.querySelectorAll(".objectifier-timeline-region title").forEach(title =>
    {
        const region = title.parentElement?.dataset;
        if(!region) return;
        title.textContent = [
            `Region ${region.regionNumber || "?"}`,
            clusterDisplayName(region.clusterLabel || "?"),
            `${region.start || "?"} - ${region.end || "?"}`,
            `Duration ${region.duration || "?"}`
        ].join("\n");
    });
}

async function persistClusterLabels(svgContainer)
{
    const fileIndex = Number(svgContainer.dataset.objectifierFileIndex || 0);
    const ref = currentObjectifierFileRef(fileIndex);
    if(!ref.fileData)
    {
        return;
    }
    ref.fileData.cluster_labels = { ...objectifierClusterLabels() };
    if(!ref.filename || !ref.hash)
    {
        return;
    }

    try
    {
        const response = await fetch(window.SoundSketcher.url("/objectifier_cluster_labels"),{
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: ref.filename,
                audio_hash: ref.hash,
                cluster_labels: ref.fileData.cluster_labels,
            }),
        });
        if(!response.ok)
        {
            console.warn("Objectifier cluster label save failed",response.status);
        }
    }
    catch(error)
    {
        console.warn("Objectifier cluster label save failed",error);
    }
}

async function persistClusterVisibility(svgContainer)
{
    const fileIndex = Number(svgContainer.dataset.objectifierFileIndex || 0);
    const ref = currentObjectifierFileRef(fileIndex);
    if(!ref.fileData) return;
    const visibility = objectifierVisibilityState(svgContainer);
    const hiddenClusters = [...visibility.hiddenClusters];
    const onlyCluster = visibility.onlyCluster || null;
    ref.fileData.hidden_clusters = hiddenClusters;
    ref.fileData.only_cluster = onlyCluster;
    if(!ref.filename || !ref.hash) return;
    try
    {
        const response = await fetch(window.SoundSketcher.url("/objectifier_visibility"),{
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: ref.filename,
                audio_hash: ref.hash,
                hidden_clusters: hiddenClusters,
                only_cluster: onlyCluster,
            }),
        });
        if(!response.ok)
        {
            console.warn("Objectifier visibility save failed",response.status);
        }
    }
    catch(error)
    {
        console.warn("Objectifier visibility save failed",error);
    }
}

async function persistObjectifierNotes(svgContainer,notes)
{
    const fileIndex = Number(svgContainer.dataset.objectifierFileIndex || 0);
    const ref = currentObjectifierFileRef(fileIndex);
    if(!ref.fileData) return;
    const trimmed = typeof notes === "string" ? notes.trim() : "";
    ref.fileData.notes = trimmed;
    if(!ref.filename || !ref.hash) return;
    try
    {
        const response = await fetch(window.SoundSketcher.url("/objectifier_notes"),{
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: ref.filename,
                audio_hash: ref.hash,
                notes: trimmed,
            }),
        });
        if(!response.ok)
        {
            console.warn("Objectifier notes save failed",response.status);
        }
    }
    catch(error)
    {
        console.warn("Objectifier notes save failed",error);
    }
}

function summarizeClusters(timelineRegions)
{
    const byLabel = new Map();
    timelineRegions.forEach(region =>
    {
        const label = String(region.label);
        const summary = byLabel.get(label) || {
            label,
            color: region.color,
            count: 0,
            duration: 0
        };
        summary.count += 1;
        summary.duration += region.end - region.start;
        byLabel.set(label,summary);
    });
    return [...byLabel.values()].sort((a,b) => b.duration - a.duration);
}

function regionTooltipText(region,index)
{
    const terms = Array.isArray(region.topTerms)
        ? region.topTerms.slice(0,3).map(term => Array.isArray(term) ? term[0] : term).filter(Boolean)
        : [];
    const lines = [
        `Region ${index + 1}`,
        clusterDisplayName(region.label),
        `${formatSeconds(region.start)} - ${formatSeconds(region.end)}`,
        `Duration ${formatSeconds(region.end - region.start)}`
    ];
    if(terms.length)
    {
        lines.push(`Terms: ${terms.join(", ")}`);
    }
    return lines.join("\n");
}

function formatSeconds(value)
{
    return `${Number(value).toFixed(2)}s`;
}

function textColorForFill(fill)
{
    const match = String(fill).match(/^#?([0-9a-f]{6})$/i);
    if(!match) return "#111";
    const hex = match[1];
    const r = parseInt(hex.slice(0,2),16);
    const g = parseInt(hex.slice(2,4),16);
    const b = parseInt(hex.slice(4,6),16);
    const brightness = (r * 299 + g * 587 + b * 114) / 1000;
    return brightness > 150 ? "#111" : "#fff";
}

function attachRegionDataset(element,region,cluster)
{
    const avg = region["visual"]?.avg || {};
    element.dataset.regionKey = region["_objectifierKey"];
    element.dataset.regionEditKey = regionEditKeyFromParts(
        region["_objectifierClusterLabel"],
        region["_objectifierOriginalStart"] ?? region["start_time"],
        region["_objectifierOriginalEnd"] ?? region["end_time"]
    );
    element.dataset.regionNumber = String(region["_objectifierNumber"]);
    element.dataset.clusterLabel = String(region["_objectifierClusterLabel"]);
    element.dataset.fileIndex = String(region["_objectifierFileIndex"] ?? 0);
    element.dataset.startSeconds = String(region["start_time"]);
    element.dataset.endSeconds = String(region["end_time"]);
    element.dataset.start = formatSeconds(region["start_time"]);
    element.dataset.end = formatSeconds(region["end_time"]);
    element.dataset.duration = formatSeconds(Number(region["end_time"]) - Number(region["start_time"]));
    element.dataset.color = cluster["color"] || "hsl(210, 80%, 52%)";
    element.dataset.loudness = formatMetric(avg["loudness"]);
    element.dataset.centroid = formatMetric(avg["spectral_centroid"],"Hz");
    element.dataset.roughness = formatMetric(avg["mir_mps_roughness"]);
    element.dataset.pitch = formatMetric(avg["f0_librosa"],"Hz");
    element.dataset.periodicity = formatMetric(avg["raw_periodicity"]);
}

function attachTimelineDataset(element,region)
{
    element.dataset.regionKey = region.key;
    element.dataset.regionEditKey = region.editKey;
    element.dataset.regionNumber = String(region.number);
    element.dataset.clusterLabel = String(region.label);
    element.dataset.fileIndex = String(region.fileIndex ?? 0);
    element.dataset.startSeconds = String(region.start);
    element.dataset.endSeconds = String(region.end);
    element.dataset.start = formatSeconds(region.start);
    element.dataset.end = formatSeconds(region.end);
    element.dataset.duration = formatSeconds(region.end - region.start);
    element.dataset.color = region.color;
    element.dataset.loudness = formatMetric(region.avg["loudness"]);
    element.dataset.centroid = formatMetric(region.avg["spectral_centroid"],"Hz");
    element.dataset.roughness = formatMetric(region.avg["mir_mps_roughness"]);
    element.dataset.pitch = formatMetric(region.avg["f0_librosa"],"Hz");
    element.dataset.periodicity = formatMetric(region.avg["raw_periodicity"]);
}

function formatMetric(value,unit = "")
{
    const numeric = Number(value);
    if(!Number.isFinite(numeric)) return "n/a";
    const formatted = Math.abs(numeric) >= 100 ? numeric.toFixed(1) : numeric.toFixed(3);
    return unit ? `${formatted} ${unit}` : formatted;
}

function wireObjectifierInteractions(svgContainer)
{
    const regionTargets = svgContainer.querySelectorAll(".objectifier-region-hit,.objectifier-region-shape,.objectifier-timeline-region");
    regionTargets.forEach(target =>
    {
        target.addEventListener("click",event =>
        {
            event.stopPropagation();
            selectObjectifierRegion(svgContainer,target.dataset.regionKey,target.dataset);
        });
    });

    svgContainer.querySelectorAll(".objectifier-legend-row").forEach(row =>
    {
        row.addEventListener("click",event =>
        {
            event.stopPropagation();
            selectObjectifierCluster(svgContainer,row.dataset);
        });
    });

    svgContainer.addEventListener("click",() =>
    {
        clearObjectifierSelection(svgContainer);
    });
}

function selectObjectifierRegion(svgContainer,regionKey,dataset)
{
    if(objectifierRegionEditState(svgContainer).deletedRegionKeys.has(regionEditKeyFromDataset(dataset))) return;
    clearObjectifierSelection(svgContainer);

    svgContainer.querySelectorAll(".objectifier-region-shape,.objectifier-timeline-region").forEach(element =>
    {
        const isSelected = element.dataset.regionKey === regionKey;
        element.setAttribute("opacity",isSelected ? "1" : "0.24");
        if(isSelected)
        {
            element.setAttribute("stroke","#000");
            element.setAttribute("stroke-width","2.4");
            element.setAttribute("stroke-opacity","0.9");
            element.setAttribute("fill-opacity",element.dataset.baseFillOpacity || "0.9");
        }
    });

    svgContainer.querySelectorAll(".objectifier-region-hit").forEach(element =>
    {
        element.setAttribute("pointer-events","all");
    });

    drawObjectifierSelectionPanel(svgContainer,[
        `Region ${dataset.regionNumber || "?"}`,
        clusterDisplayName(dataset.clusterLabel || "?"),
        `${dataset.start || "?"} - ${dataset.end || "?"}`,
        `Duration ${dataset.duration || "?"}`,
        `Loudness ${dataset.loudness || "n/a"}`,
        `Centroid ${dataset.centroid || "n/a"}`,
        `Roughness ${dataset.roughness || "n/a"}`,
        `Pitch ${dataset.pitch || "n/a"}`,
        `Periodicity ${dataset.periodicity || "n/a"}`
    ],dataset,{ regionKey });
}

function selectObjectifierCluster(svgContainer,dataset)
{
    const clusterLabel = dataset.clusterLabel;
    if(svgContainer.dataset.objectifierSoloCluster === clusterLabel)
    {
        clearObjectifierSelection(svgContainer);
        return;
    }

    clearObjectifierSelection(svgContainer);
    svgContainer.dataset.objectifierSoloCluster = clusterLabel;

    svgContainer.querySelectorAll(".objectifier-region-shape,.objectifier-timeline-region").forEach(element =>
    {
        const isSelected = element.dataset.clusterLabel === clusterLabel;
        element.setAttribute("opacity",isSelected ? "1" : "0.18");
        if(isSelected)
        {
            element.setAttribute("stroke","#000");
            element.setAttribute("stroke-width","1.8");
            element.setAttribute("stroke-opacity","0.78");
        }
    });

    svgContainer.querySelectorAll(".objectifier-legend-row").forEach(row =>
    {
        const isSelected = row.dataset.clusterLabel === clusterLabel;
        row.dataset.active = isSelected ? "true" : "false";
        const hit = row.querySelector("rect:first-child");
        if(hit)
        {
            hit.setAttribute("fill",isSelected ? "rgba(0,0,0,0.11)" : "transparent");
        }
        row.querySelectorAll("text").forEach(text =>
        {
            text.setAttribute("font-weight",isSelected ? "700" : "400");
        });
        row.querySelectorAll("rect").forEach((rect,index) =>
        {
            if(index > 0)
            {
                rect.setAttribute("stroke",isSelected ? "#000" : "none");
                rect.setAttribute("stroke-width",isSelected ? "1.4" : "0");
            }
        });
    });

    const fileIndex = Number(svgContainer.dataset.objectifierFileIndex || 0);
    const semanticLabels = clusterSemanticLabels(fileIndex,clusterLabel);
    const semanticLines = semanticLabels
        ? semanticLabels.map(l => `${l.term}  ${Math.round(l.score * 100)}%`)
        : [];
    drawObjectifierSelectionPanel(svgContainer,[
        clusterDisplayName(clusterLabel),
        `${dataset.regionCount || "0"} regions`,
        `Total duration ${dataset.duration || "n/a"}`,
        clusterVisibilityStatus(svgContainer,clusterLabel),
        ...(semanticLines.length ? ["Semantic labels:","  " + semanticLines.join("  ·  ")] : []),
        "Click again to clear solo",
        "Click a timeline block for region details"
    ],dataset,{ clusterLabel });
}

function clusterVisibilityStatus(svgContainer,clusterLabel)
{
    const visibility = objectifierVisibilityState(svgContainer);
    if(visibility.hiddenClusters.has(String(clusterLabel))) return "Visibility: hidden";
    if(visibility.onlyCluster === String(clusterLabel)) return "Visibility: only cluster";
    if(visibility.onlyCluster) return "Visibility: filtered";
    return "Visibility: visible";
}

function clusterSemanticLabels(fileIndex,clusterLabel)
{
    const clusters = currentObjectifierState(fileIndex)?.clusters;
    if(!Array.isArray(clusters)) return null;
    const cluster = clusters.find(c => String(c.label) === String(clusterLabel));
    const labels = cluster?.semantic_labels;
    if(!Array.isArray(labels) || labels.length === 0) return null;
    return labels;
}

async function triggerObjectifierSemanticLabels(svgContainer,buttonElement)
{
    const fileIndex = Number(svgContainer.dataset.objectifierFileIndex || 0);
    const ref = currentObjectifierFileRef(fileIndex);
    if(!ref.filename || !ref.hash) return;
    if(buttonElement) buttonElement.textContent = "Generating…";
    try
    {
        const response = await fetch(window.SoundSketcher.url("/objectifier_semantic_labels"),{
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: ref.filename,
                audio_hash: ref.hash,
                force: true,
            }),
        });
        if(!response.ok)
        {
            console.warn("Semantic labels request failed",response.status);
            if(buttonElement) buttonElement.textContent = "Generate semantic labels";
            return;
        }
    }
    catch(err)
    {
        console.warn("Semantic labels request failed",err);
        if(buttonElement) buttonElement.textContent = "Generate semantic labels";
        return;
    }

    let pollAttempts = 0;
    const MAX_POLL_ATTEMPTS = 90; // 3 minutes max
    const pollInterval = setInterval(async () =>
    {
        pollAttempts++;
        if(pollAttempts > MAX_POLL_ATTEMPTS)
        {
            clearInterval(pollInterval);
            if(buttonElement) buttonElement.textContent = "Generate semantic labels";
            console.warn("Semantic labels: timed out waiting for job");
            return;
        }
        try
        {
            const statusResponse = await fetch(window.SoundSketcher.url(`/semantic_labels_status?audio_hash=${encodeURIComponent(ref.hash)}&filename=${encodeURIComponent(ref.filename)}`));
            if(!statusResponse.ok) return;
            const statusData = await statusResponse.json();
            const jobStatus = statusData?.job?.status;
            if(jobStatus === "done")
            {
                clearInterval(pollInterval);
                const submitButton = document.getElementById("submitButton");
                if(submitButton) submitButton.click();
                else window.location.reload();
            }
            else if(jobStatus === "failed")
            {
                clearInterval(pollInterval);
                if(buttonElement) buttonElement.textContent = "Generate semantic labels";
                console.warn("Semantic labels job failed:",statusData?.job?.error);
                window.alert(`Semantic labels failed:\n${statusData?.job?.error || "unknown error"}`);
            }
            else if(statusData?.job === null || statusData?.job === undefined)
            {
                // Server restarted — job state lost
                clearInterval(pollInterval);
                if(buttonElement) buttonElement.textContent = "Generate semantic labels";
            }
        }
        catch(err)
        {
            console.warn("Semantic labels poll failed",err);
        }
    },2000);
}

function clusterRegionSegments(svgContainer,clusterLabel)
{
    return [...svgContainer.querySelectorAll(".objectifier-timeline-region")]
        .filter(region => region.dataset.clusterLabel === String(clusterLabel))
        .filter(region => !objectifierRegionEditState(svgContainer).deletedRegionKeys.has(regionEditKeyFromDataset(region.dataset)))
        .map(region => ({
            start: Number(region.dataset.startSeconds),
            end: Number(region.dataset.endSeconds),
            fileIndex: Number(region.dataset.fileIndex || 0),
            regionKey: region.dataset.regionKey
        }))
        .filter(region => Number.isFinite(region.start) && Number.isFinite(region.end) && region.end > region.start)
        .sort((a,b) => a.start - b.start);
}

function setClusterPlaybackHighlight(svgContainer,regionKey)
{
    svgContainer.querySelectorAll(".objectifier-timeline-region").forEach(region =>
    {
        const isCurrent = region.dataset.regionKey === regionKey;
        if(isCurrent)
        {
            region.setAttribute("stroke","#d00000");
            region.setAttribute("stroke-width","2.2");
        }
        else if(region.dataset.clusterLabel === svgContainer.dataset.objectifierSoloCluster)
        {
            region.setAttribute("stroke","#000");
            region.setAttribute("stroke-width","1.8");
        }
    });
}

function objectifierVisibilityTargets(svgContainer)
{
    return svgContainer.querySelectorAll(
        ".objectifier-region-shape,.objectifier-region-hit,.objectifier-timeline-region,.objectifier-timeline-label"
    );
}

function applyClusterVisibility(svgContainer)
{
    const visibility = objectifierVisibilityState(svgContainer);
    const edits = objectifierRegionEditState(svgContainer);
    const onlyCluster = visibility.onlyCluster;
    objectifierVisibilityTargets(svgContainer).forEach(element =>
    {
        const label = String(element.dataset.clusterLabel || "");
        const regionKey = String(element.dataset.regionEditKey || "");
        const isDeleted = regionKey && edits.deletedRegionKeys.has(regionKey);
        const isHidden = visibility.hiddenClusters.has(label);
        const isOutsideOnly = Boolean(onlyCluster) && label !== onlyCluster;
        const isVisible = !isDeleted && !isHidden && !isOutsideOnly;
        element.style.display = isVisible ? "" : "none";
        if(element.classList?.contains("objectifier-region-hit"))
        {
            element.setAttribute("pointer-events",isVisible ? "all" : "none");
        }
    });

    svgContainer.querySelectorAll(".objectifier-legend-row").forEach(row =>
    {
        const label = String(row.dataset.clusterLabel || "");
        const isHidden = visibility.hiddenClusters.has(label);
        const isOnly = onlyCluster === label;
        const isDimmed = isHidden || (Boolean(onlyCluster) && !isOnly);
        row.dataset.hidden = isHidden ? "true" : "false";
        row.dataset.only = isOnly ? "true" : "false";
        row.setAttribute("opacity",isDimmed ? "0.42" : "1");
    });
}

async function persistObjectifierRegionEdits(svgContainer)
{
    const fileIndex = Number(svgContainer.dataset.objectifierFileIndex || 0);
    const ref = currentObjectifierFileRef(fileIndex);
    if(!ref.fileData)
    {
        return;
    }
    const deletedRegions = [...objectifierRegionEditState(svgContainer).deletedRegionKeys];
    const regionOverrides = Object.fromEntries(
        [...objectifierRegionEditState(svgContainer).boundaryOverrides.entries()]
            .filter(([,override]) => Number.isFinite(override.start_time) && Number.isFinite(override.end_time))
    );
    ref.fileData.deleted_regions = deletedRegions;
    ref.fileData.region_overrides = regionOverrides;
    if(!ref.filename || !ref.hash)
    {
        return;
    }

    try
    {
        const response = await fetch(window.SoundSketcher.url("/objectifier_region_edits"),{
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: ref.filename,
                audio_hash: ref.hash,
                deleted_regions: deletedRegions,
                region_overrides: regionOverrides,
            }),
        });
        if(!response.ok)
        {
            console.warn("Objectifier region edits save failed",response.status);
        }
    }
    catch(error)
    {
        console.warn("Objectifier region edits save failed",error);
    }
}

function resetObjectifierRegionBoundaries(svgContainer,dataset)
{
    const key = regionEditKeyFromDataset(dataset);
    if(!key) return;
    objectifierRegionEditState(svgContainer).boundaryOverrides.delete(key);
    persistObjectifierRegionEdits(svgContainer);
    const submitButton = document.getElementById("submitButton");
    if(submitButton)
    {
        submitButton.click();
    }
    else
    {
        window.location.reload();
    }
}

function resetAllObjectifierBoundaries(svgContainer)
{
    objectifierRegionEditState(svgContainer).boundaryOverrides.clear();
    persistObjectifierRegionEdits(svgContainer);
    const submitButton = document.getElementById("submitButton");
    if(submitButton)
    {
        submitButton.click();
    }
    else
    {
        window.location.reload();
    }
}

async function adjustObjectifierRegionBoundaries(svgContainer,dataset)
{
    const key = regionEditKeyFromDataset(dataset);
    if(!key) return;
    const currentStart = Number(dataset.startSeconds);
    const currentEnd = Number(dataset.endSeconds);
    const startText = window.prompt("Start time in seconds",Number.isFinite(currentStart) ? currentStart.toFixed(3) : "");
    if(startText === null) return;
    const endText = window.prompt("End time in seconds",Number.isFinite(currentEnd) ? currentEnd.toFixed(3) : "");
    if(endText === null) return;
    const start = Number(startText);
    const end = Number(endText);
    if(!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start)
    {
        window.alert("Please enter valid boundaries: start >= 0 and end > start.");
        return;
    }

    objectifierRegionEditState(svgContainer).boundaryOverrides.set(key,{
        start_time: Number(start.toFixed(3)),
        end_time: Number(end.toFixed(3))
    });
    await persistObjectifierRegionEdits(svgContainer);
    const submitButton = document.getElementById("submitButton");
    if(submitButton)
    {
        submitButton.click();
    }
    else
    {
        window.location.reload();
    }
}

function deleteObjectifierRegion(svgContainer,dataset)
{
    const key = regionEditKeyFromDataset(dataset);
    if(!key) return;
    objectifierRegionEditState(svgContainer).deletedRegionKeys.add(key);
    window.SoundSketcher?.playback?.stopSegment?.();
    stopObjectifierClusterPlayback(svgContainer);
    svgContainer.querySelectorAll(".objectifier-selection-panel").forEach(panel => panel.remove());
    applyClusterVisibility(svgContainer);
    persistObjectifierRegionEdits(svgContainer);
}

function restoreObjectifierRegions(svgContainer)
{
    objectifierRegionEditState(svgContainer).deletedRegionKeys.clear();
    applyClusterVisibility(svgContainer);
    persistObjectifierRegionEdits(svgContainer);
}

function toggleObjectifierClusterHidden(svgContainer,clusterLabel)
{
    const label = String(clusterLabel);
    const visibility = objectifierVisibilityState(svgContainer);
    if(visibility.hiddenClusters.has(label))
    {
        visibility.hiddenClusters.delete(label);
    }
    else
    {
        visibility.hiddenClusters.add(label);
        if(visibility.onlyCluster === label)
        {
            visibility.onlyCluster = null;
        }
    }
    applyClusterVisibility(svgContainer);
    persistClusterVisibility(svgContainer);
}

function toggleObjectifierOnlyCluster(svgContainer,clusterLabel)
{
    const label = String(clusterLabel);
    const visibility = objectifierVisibilityState(svgContainer);
    visibility.onlyCluster = visibility.onlyCluster === label ? null : label;
    if(visibility.onlyCluster)
    {
        visibility.hiddenClusters.delete(label);
    }
    applyClusterVisibility(svgContainer);
    persistClusterVisibility(svgContainer);
}

function showAllObjectifierClusters(svgContainer)
{
    const visibility = objectifierVisibilityState(svgContainer);
    visibility.hiddenClusters.clear();
    visibility.onlyCluster = null;
    applyClusterVisibility(svgContainer);
    persistClusterVisibility(svgContainer);
}

function stopObjectifierClusterPlayback(svgContainer)
{
    const playback = objectifierClusterPlayback();
    playback.token += 1;
    playback.clusterLabel = null;
    window.SoundSketcher?.playback?.stopSegment?.();
    setClusterPlaybackHighlight(svgContainer,null);
}

async function playObjectifierCluster(svgContainer,clusterLabel)
{
    const playback = objectifierClusterPlayback();
    if(playback.clusterLabel === String(clusterLabel))
    {
        stopObjectifierClusterPlayback(svgContainer);
        return;
    }

    stopObjectifierClusterPlayback(svgContainer);
    const token = playback.token;
    playback.clusterLabel = String(clusterLabel);
    const regions = clusterRegionSegments(svgContainer,clusterLabel);
    const gapMs = 150;

    for(const region of regions)
    {
        if(playback.token !== token || playback.clusterLabel !== String(clusterLabel)) break;
        setClusterPlaybackHighlight(svgContainer,region.regionKey);
        await window.SoundSketcher?.playback?.playSegment?.(region.start,region.end,region.fileIndex);
        if(playback.token !== token || playback.clusterLabel !== String(clusterLabel)) break;
        await new Promise(resolve => setTimeout(resolve,gapMs));
    }

    if(playback.token === token && playback.clusterLabel === String(clusterLabel))
    {
        playback.clusterLabel = null;
        setClusterPlaybackHighlight(svgContainer,null);
    }
}

function clearObjectifierSelection(svgContainer)
{
    delete svgContainer.dataset.objectifierSoloCluster;
    svgContainer.querySelectorAll(".objectifier-selection-panel").forEach(panel => panel.remove());
    svgContainer.querySelectorAll(".objectifier-region-shape").forEach(element =>
    {
        element.setAttribute("opacity","1");
        element.setAttribute("stroke","#000");
        element.setAttribute("stroke-width","1");
        element.setAttribute("stroke-opacity","0.18");
        element.setAttribute("fill-opacity",element.dataset.baseFillOpacity || "0.35");
    });
    svgContainer.querySelectorAll(".objectifier-timeline-region").forEach(element =>
    {
        element.setAttribute("opacity","1");
        element.setAttribute("stroke","rgba(0,0,0,0.36)");
        element.setAttribute("stroke-width","0.7");
        element.setAttribute("fill-opacity","0.9");
    });
    svgContainer.querySelectorAll(".objectifier-legend-row").forEach(row =>
    {
        row.dataset.active = "false";
        const hit = row.querySelector("rect:first-child");
        if(hit)
        {
            hit.setAttribute("fill","transparent");
        }
        row.querySelectorAll("text").forEach(text =>
        {
            text.setAttribute("font-weight","400");
        });
        row.querySelectorAll("rect").forEach((rect,index) =>
        {
            if(index > 0)
            {
                rect.setAttribute("stroke","none");
                rect.setAttribute("stroke-width","0");
            }
        });
    });
    applyClusterVisibility(svgContainer);
}

function drawObjectifierSelectionPanel(svgContainer,lines,dataset = null,options = {})
{
    const panel = document.createElementNS("http://www.w3.org/2000/svg","g");
    panel.setAttribute("class","objectifier-selection-panel");
    panel.addEventListener("click",event => event.stopPropagation());
    svgContainer.appendChild(panel);

    const x = 12;
    const y = 12;
    const width = 238;
    const lineHeight = 16;
    const hasPlayableRegion = dataset?.startSeconds && dataset?.endSeconds;
    const buttonDefs = [];
    if(hasPlayableRegion)
    {
        buttonDefs.push({
            label: "Play selected region",
            onClick: () =>
            {
                window.SoundSketcher?.playback?.playSegment?.(
                    Number(dataset.startSeconds),
                    Number(dataset.endSeconds),
                    Number(dataset.fileIndex || 0)
                );
            }
        });
        buttonDefs.push({
            label: "Delete region",
            onClick: () =>
            {
                deleteObjectifierRegion(svgContainer,dataset);
            }
        });
        buttonDefs.push({
            label: "Adjust boundaries",
            onClick: () =>
            {
                adjustObjectifierRegionBoundaries(svgContainer,dataset);
            }
        });
        const editKey = dataset?.regionEditKey || regionEditKeyFromDataset(dataset);
        if(editKey && objectifierRegionEditState(svgContainer).boundaryOverrides.has(editKey))
        {
            buttonDefs.push({
                label: "Reset to original boundaries",
                onClick: () =>
                {
                    resetObjectifierRegionBoundaries(svgContainer,dataset);
                }
            });
        }
    }
    if(objectifierRegionEditState(svgContainer).deletedRegionKeys.size)
    {
        buttonDefs.push({
            label: "Restore deleted regions",
            onClick: () =>
            {
                restoreObjectifierRegions(svgContainer);
                svgContainer.querySelectorAll(".objectifier-selection-panel").forEach(panel => panel.remove());
            }
        });
    }
    if(objectifierRegionEditState(svgContainer).boundaryOverrides?.size)
    {
        buttonDefs.push({
            label: "Reset all boundary edits",
            onClick: () =>
            {
                resetAllObjectifierBoundaries(svgContainer);
                svgContainer.querySelectorAll(".objectifier-selection-panel").forEach(panel => panel.remove());
            }
        });
    }
    if(options.clusterLabel !== undefined)
    {
        const visibility = objectifierVisibilityState(svgContainer);
        const selectedCluster = String(options.clusterLabel);
        buttonDefs.push({
            label: objectifierClusterPlayback().clusterLabel === selectedCluster ? "Stop cluster" : "Play cluster",
            onClick: () =>
            {
                playObjectifierCluster(svgContainer,options.clusterLabel);
            }
        });
        buttonDefs.push({
            label: visibility.hiddenClusters.has(selectedCluster) ? "Unhide cluster" : "Hide cluster",
            onClick: () =>
            {
                toggleObjectifierClusterHidden(svgContainer,options.clusterLabel);
                delete svgContainer.dataset.objectifierSoloCluster;
                selectObjectifierCluster(svgContainer,dataset);
            }
        });
        buttonDefs.push({
            label: visibility.onlyCluster === selectedCluster ? "Show all clusters" : "Show only cluster",
            onClick: () =>
            {
                toggleObjectifierOnlyCluster(svgContainer,options.clusterLabel);
                delete svgContainer.dataset.objectifierSoloCluster;
                selectObjectifierCluster(svgContainer,dataset);
            }
        });
        if(visibility.hiddenClusters.size || visibility.onlyCluster)
        {
            buttonDefs.push({
                label: "Reset visibility",
                onClick: () =>
                {
                    showAllObjectifierClusters(svgContainer);
                    delete svgContainer.dataset.objectifierSoloCluster;
                    selectObjectifierCluster(svgContainer,dataset);
                }
            });
        }
        buttonDefs.push({
            label: "Rename cluster",
            onClick: () =>
            {
                const label = String(options.clusterLabel);
                const next = window.prompt("Name this cluster",customClusterName(label));
                if(next === null) return;
                const trimmed = next.trim();
                if(trimmed)
                {
                    objectifierClusterLabels()[label] = trimmed;
                }
                else
                {
                    delete objectifierClusterLabels()[label];
                }
                applyClusterLabels(svgContainer);
                persistClusterLabels(svgContainer);
                delete svgContainer.dataset.objectifierSoloCluster;
                selectObjectifierCluster(svgContainer,dataset);
            }
        });
        const fileIndex = Number(svgContainer.dataset.objectifierFileIndex || 0);
        const currentNotes = currentObjectifierState(fileIndex)?.notes || "";
        buttonDefs.push({
            label: currentNotes ? "Edit session notes" : "Add session notes",
            onClick: () =>
            {
                const next = window.prompt("Session notes for this audio",currentNotes);
                if(next === null) return;
                persistObjectifierNotes(svgContainer,next);
            }
        });
        const hasSemanticLabels = Boolean(clusterSemanticLabels(fileIndex,options.clusterLabel));
        buttonDefs.push({
            label: hasSemanticLabels ? "Regenerate semantic labels" : "Generate semantic labels",
            isAsync: true,
            onClick: (buttonTextElement) =>
            {
                triggerObjectifierSemanticLabels(svgContainer,buttonTextElement);
            }
        });
    }
    const buttonHeight = buttonDefs.length ? buttonDefs.length * 26 : 0;
    const height = 18 + lines.length * lineHeight + buttonHeight + (buttonDefs.length ? 8 : 0);

    const background = document.createElementNS("http://www.w3.org/2000/svg","rect");
    background.setAttribute("x",x);
    background.setAttribute("y",y);
    background.setAttribute("width",width);
    background.setAttribute("height",height);
    background.setAttribute("rx",6);
    background.setAttribute("fill","rgba(255,255,255,0.9)");
    background.setAttribute("stroke","rgba(0,0,0,0.28)");
    background.setAttribute("stroke-width","1");
    panel.appendChild(background);

    lines.forEach((line,index) =>
    {
        const text = document.createElementNS("http://www.w3.org/2000/svg","text");
        text.setAttribute("x",x + 10);
        text.setAttribute("y",y + 18 + index * lineHeight);
        text.setAttribute("font-size",index === 0 ? "12" : "11");
        text.setAttribute("font-weight",index === 0 ? "700" : "400");
        text.setAttribute("font-family","system-ui, sans-serif");
        text.setAttribute("fill","#111");
        text.textContent = line;
        panel.appendChild(text);
    });

    buttonDefs.forEach((buttonDef,index) =>
    {
        const buttonY = y + 18 + lines.length * lineHeight + 4 + index * 26;
        const button = document.createElementNS("http://www.w3.org/2000/svg","g");
        button.setAttribute("class","objectifier-panel-button");
        button.setAttribute("cursor","pointer");
        panel.appendChild(button);

        const buttonRect = document.createElementNS("http://www.w3.org/2000/svg","rect");
        buttonRect.setAttribute("x",x + 10);
        buttonRect.setAttribute("y",buttonY);
        buttonRect.setAttribute("width",width - 20);
        buttonRect.setAttribute("height",22);
        buttonRect.setAttribute("rx",4);
        buttonRect.setAttribute("fill","#111");
        buttonRect.setAttribute("fill-opacity","0.86");
        button.appendChild(buttonRect);

        const buttonText = document.createElementNS("http://www.w3.org/2000/svg","text");
        buttonText.setAttribute("x",x + width / 2);
        buttonText.setAttribute("y",buttonY + 15);
        buttonText.setAttribute("text-anchor","middle");
        buttonText.setAttribute("font-size","11");
        buttonText.setAttribute("font-weight","700");
        buttonText.setAttribute("font-family","system-ui, sans-serif");
        buttonText.setAttribute("fill","#fff");
        buttonText.setAttribute("pointer-events","none");
        buttonText.textContent = buttonDef.label;
        button.appendChild(buttonText);

        button.addEventListener("click",event =>
        {
            event.stopPropagation();
            buttonDef.onClick(buttonDef.isAsync ? buttonText : undefined);
        });
    });
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
    draw,
    cluster = {}
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
    const alpha = avgLoudness > loudnessThreshold
      ? clamp(0.24 + avgLoudness * 0.28, 0.26, 0.56)
      : 0.12;
  
    const path = draw.path(pathData)
      .fill({ color: baseColor, opacity: alpha })
      .stroke({ width: 1, color: "#000", opacity: 0.18 })
      .attr({
        'class': 'objectifier-region-shape',
        'data-region-key': region["_objectifierKey"],
        'data-region-number': String(region["_objectifierNumber"]),
        'data-cluster-label': String(region["_objectifierClusterLabel"]),
        'data-base-fill-opacity': String(alpha),
        'fill-opacity': alpha,
        'vector-effect': 'non-scaling-stroke',
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
        'cursor': 'pointer'
      });
    attachRegionDataset(path.node,region,cluster);
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
