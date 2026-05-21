function createDrawingProfiler(label) {
    const enabled = window.SOUNDSKETCHER_PROFILE !== false;
    const startedAt = performance.now();
    const totals = {};
    const counts = {};

    function addTiming(name, duration) {
        totals[name] = (totals[name] || 0) + duration;
        counts[name] = (counts[name] || 0) + 1;
    }

    return {
        measure(name, work) {
            if (!enabled) return work();

            const sectionStartedAt = performance.now();
            try {
                return work();
            } finally {
                addTiming(name, performance.now() - sectionStartedAt);
            }
        },

        count(name, amount = 1) {
            counts[name] = (counts[name] || 0) + amount;
        },

        finish(meta = {}) {
            if (!enabled) return;

            const totalMs = performance.now() - startedAt;
            const sections = Object.entries(totals)
                .map(([name, ms]) => ({
                    section: name,
                    ms: Number(ms.toFixed(2)),
                    calls: counts[name] || 0,
                }))
                .sort((a, b) => b.ms - a.ms);

            window.__soundSketcherLastDrawProfile = {
                label,
                totalMs: Number(totalMs.toFixed(2)),
                meta,
                sections,
                counts: { ...counts },
            };

            console.groupCollapsed(`[SoundSketcher] ${label}: ${totalMs.toFixed(2)}ms`);
            console.table(sections);
            console.log("meta", meta);
            console.groupEnd();
        },
    };
}
