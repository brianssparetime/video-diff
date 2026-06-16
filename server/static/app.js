const STAGE_LABELS = {
    starting: "Starting comparison...",
    hashing: "Computing file hashes...",
    fingerprint_a: "Analyzing Video A",
    fingerprint_b: "Analyzing Video B",
    comparing: "Comparing fingerprints...",
    done: "Done",
};

// Stored after comparison for scrub viewer
let scrubState = null;

async function runCompare(force) {
    const fileA = document.getElementById("file-a").value.trim();
    const fileB = document.getElementById("file-b").value.trim();
    const granularity = parseFloat(document.getElementById("granularity").value);

    if (!fileA || !fileB) {
        showStatus("Please enter both file paths.", "error");
        return;
    }

    const btn = document.getElementById("compare-btn");
    btn.disabled = true;
    const rerunBtn = document.getElementById("rerun-btn");
    if (rerunBtn) rerunBtn.disabled = true;
    showProgress("starting", 0);
    hideResult();
    scrubState = null;

    try {
        const resp = await fetch("/api/compare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                file_a: fileA,
                file_b: fileB,
                granularity: granularity,
                force: !!force,
            }),
        });

        const data = await resp.json();

        if (!resp.ok) {
            showStatus(data.error || "Comparison failed.", "error");
            btn.disabled = false;
            return;
        }

        pollJob(data.job_id, btn, fileA, fileB);
    } catch (e) {
        showStatus("Request failed: " + e.message, "error");
        btn.disabled = false;
    }
}

async function pollJob(jobId, btn, fileA, fileB) {
    try {
        const resp = await fetch("/api/status/" + jobId);
        const data = await resp.json();

        if (data.status === "running") {
            showProgress(data.stage, data.detail.percent || 0, data.detail.cached || false);
            setTimeout(() => pollJob(jobId, btn, fileA, fileB), 1000);
            return;
        }

        if (data.status === "done") {
            hideStatus();
            showResult(data.result, fileA, fileB);
        } else if (data.status === "error") {
            showStatus(data.error || "Comparison failed.", "error");
        }
    } catch (e) {
        showStatus("Lost connection to server.", "error");
    }

    btn.disabled = false;
    const rerunBtn = document.getElementById("rerun-btn");
    if (rerunBtn) rerunBtn.disabled = false;
}

function showProgress(stage, percent, cached) {
    const el = document.getElementById("status");
    const label = STAGE_LABELS[stage] || stage;
    const pct = Math.min(100, Math.max(0, percent || 0));
    const suffix = cached ? " (cached)" : (pct > 0 && pct < 100 ? " (" + pct + "%)" : "");

    el.innerHTML = `
        <div class="progress-label">${label}${suffix}</div>
        <div class="progress-bar-track">
            <div class="progress-bar-fill" style="width: ${pct}%"></div>
        </div>
    `;
    el.className = "status loading";
}

function showStatus(msg, cls) {
    const el = document.getElementById("status");
    el.textContent = msg;
    el.className = "status " + cls;
}

function hideStatus() {
    document.getElementById("status").className = "status hidden";
}

function hideResult() {
    document.getElementById("result-section").className = "result-section hidden";
    document.getElementById("scrub-viewer").className = "scrub-viewer hidden";
}

function showResult(data, fileA, fileB) {
    const section = document.getElementById("result-section");
    section.className = "result-section";

    const rerunBtn = document.getElementById("rerun-btn");
    if (rerunBtn) {
        rerunBtn.className = "btn-secondary";
        rerunBtn.disabled = false;
    }

    const badge = document.getElementById("match-badge");
    const details = document.getElementById("match-details");

    let badgeText, badgeClass;
    if (data.match_type === "byte") {
        badgeText = "Byte Match";
        badgeClass = "match-badge match-green";
    } else if (data.match_type === "transcode") {
        badgeText = "Transcode Match";
        badgeClass = "match-badge match-yellow";
    } else {
        badgeText = "No Match";
        badgeClass = "match-badge match-red";
    }
    badge.textContent = badgeText;
    badge.className = badgeClass;

    let detailParts = [];
    if (data.subtype) {
        let subtypeLabel = data.subtype.charAt(0).toUpperCase() + data.subtype.slice(1);
        if (data.subtype === "subset" && data.subset_direction) {
            if (data.subset_direction === "a_in_b") {
                subtypeLabel += " (A is contained in B)";
            } else {
                subtypeLabel += " (B is contained in A)";
            }
        }
        detailParts.push("Subtype: " + subtypeLabel);
    }
    detailParts.push("A: " + formatDuration(data.a_duration));
    detailParts.push("B: " + formatDuration(data.b_duration));
    if (!data.video_available) detailParts.push("No video track");
    if (!data.audio_available) detailParts.push("No audio track");
    details.textContent = detailParts.join("  |  ");

    const vizSection = document.getElementById("viz-section");
    const needsViz = data.match_type !== "byte" && data.subtype !== "straight";

    if (needsViz && data.video_available && data.video_segments.length > 0) {
        vizSection.className = "viz-section";
        renderDiffViz("video-viz", data.video_segments, data.a_duration, data.b_duration);
        initScrub(data, fileA, fileB);
    } else {
        vizSection.className = "viz-section hidden";
    }

    const audioHeading = document.getElementById("audio-heading");
    const audioViz = document.getElementById("audio-viz");
    if (needsViz && data.audio_available && data.audio_segments.length > 0) {
        audioHeading.className = "";
        audioViz.className = "diff-viz";
        renderDiffViz("audio-viz", data.audio_segments, data.a_duration, data.b_duration);
    } else {
        audioHeading.className = "hidden";
        audioViz.className = "diff-viz hidden";
    }
}

// ---- Scrub viewer ----

function initScrub(data, fileA, fileB) {
    scrubState = {
        fileA: fileA,
        fileB: fileB,
        aDuration: data.a_duration,
        bDuration: data.b_duration,
        segments: data.video_segments,
        debounceTimer: null,
        locked: false,
        lockedPrimaryIsA: true,
        lockedPrimaryT: 0,
    };

    const svg = document.getElementById("video-viz");

    svg.removeEventListener("mousemove", onScrubMove);
    svg.removeEventListener("mouseleave", onScrubLeave);
    svg.removeEventListener("click", onScrubClick);
    document.removeEventListener("click", onDocumentClick);
    document.removeEventListener("keydown", onScrubKey);

    svg.addEventListener("mousemove", onScrubMove);
    svg.addEventListener("mouseleave", onScrubLeave);
    svg.addEventListener("click", onScrubClick);
    document.addEventListener("click", onDocumentClick);
    document.addEventListener("keydown", onScrubKey);
}

function onScrubMove(e) {
    if (!scrubState || scrubState.locked) return;

    clearTimeout(scrubState.debounceTimer);
    scrubState.debounceTimer = setTimeout(() => doScrub(), 80);

    drawScrubLines(e);
}

function onScrubLeave() {
    if (!scrubState || scrubState.locked) return;

    const svg = document.getElementById("video-viz");
    svg.querySelectorAll(".scrub-line").forEach(el => el.remove());
    document.getElementById("scrub-viewer").className = "scrub-viewer hidden";
}

function onScrubClick(e) {
    if (!scrubState) return;
    e.stopPropagation();

    // Compute position from the click
    drawScrubLines(e);

    scrubState.locked = true;
    scrubState.lockedPrimaryIsA = scrubState._lastPrimaryIsA;
    scrubState.lockedPrimaryT = scrubState._lastPrimaryT;
    doScrub();
}

function onDocumentClick(e) {
    if (!scrubState || !scrubState.locked) return;

    // If click is inside the bars container, ignore (onScrubClick handles it)
    const container = document.getElementById("bars-container");
    if (container && container.contains(e.target)) return;

    // Unlock
    scrubState.locked = false;
}

function onScrubKey(e) {
    if (!scrubState || !scrubState.locked) return;

    const FINE_STEP = 0.5;
    const COARSE_STEP = 30;

    let delta = 0;
    if (e.key === "ArrowRight") delta = FINE_STEP;
    else if (e.key === "ArrowLeft") delta = -FINE_STEP;
    else if (e.key === "ArrowUp") delta = COARSE_STEP;
    else if (e.key === "ArrowDown") delta = -COARSE_STEP;
    else return;

    e.preventDefault();

    const duration = scrubState.lockedPrimaryIsA ? scrubState.aDuration : scrubState.bDuration;
    scrubState.lockedPrimaryT = Math.max(0, Math.min(duration, scrubState.lockedPrimaryT + delta));

    updateLockedView();
}

function drawScrubLines(e) {
    const svg = document.getElementById("video-viz");
    const rect = svg.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const width = svg.clientWidth || 850;
    const height = 160;
    const padding = 20;
    const barHeight = 28;
    const barY_A = 20;
    const barY_B = height - barHeight - 20;
    const barWidth = width - 2 * padding;

    const midY = (barY_A + barHeight + barY_B) / 2;
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    const svgX = mouseX * scaleX;
    const svgY = mouseY * scaleY;

    const primaryIsA = svgY < midY;
    const clampedX = Math.max(padding, Math.min(svgX, padding + barWidth));
    const fraction = (clampedX - padding) / barWidth;
    const primaryDuration = primaryIsA ? scrubState.aDuration : scrubState.bDuration;
    const primaryT = fraction * primaryDuration;

    scrubState._lastPrimaryIsA = primaryIsA;
    scrubState._lastPrimaryT = primaryT;
    scrubState._lastResult = computeSecondaryTime(primaryT, primaryIsA, scrubState.segments);

    renderScrubLines(primaryT, primaryIsA, scrubState._lastResult);
}

function updateLockedView() {
    const primaryT = scrubState.lockedPrimaryT;
    const primaryIsA = scrubState.lockedPrimaryIsA;
    const result = computeSecondaryTime(primaryT, primaryIsA, scrubState.segments);

    scrubState._lastPrimaryIsA = primaryIsA;
    scrubState._lastPrimaryT = primaryT;
    scrubState._lastResult = result;

    renderScrubLines(primaryT, primaryIsA, result);
    doScrub();
}

function renderScrubLines(primaryT, primaryIsA, result) {
    const svg = document.getElementById("video-viz");
    const width = svg.clientWidth || 850;
    const padding = 20;
    const barHeight = 28;
    const barY_A = 20;
    const barY_B = 160 - barHeight - 20;
    const barWidth = width - 2 * padding;

    svg.querySelectorAll(".scrub-line, .scrub-bar-highlight").forEach(el => el.remove());

    const primaryDuration = primaryIsA ? scrubState.aDuration : scrubState.bDuration;
    const primaryX = padding + (primaryT / primaryDuration) * barWidth;
    const primaryBarY = primaryIsA ? barY_A : barY_B;

    // Blue glow highlight on the primary bar
    drawBarHighlight(svg, padding, primaryBarY, barWidth, barHeight);

    // Both playheads are yellow
    drawScrubLine(svg, primaryX, primaryBarY, barHeight, "#ffeb3b");

    if (result.secondaryT !== null) {
        const secondaryDuration = primaryIsA ? scrubState.bDuration : scrubState.aDuration;
        const secondaryX = padding + (result.secondaryT / secondaryDuration) * barWidth;
        const secondaryBarY = primaryIsA ? barY_B : barY_A;
        drawScrubLine(svg, secondaryX, secondaryBarY, barHeight, "#ffeb3b");
    }
}

function drawBarHighlight(svg, x, y, w, h) {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", w);
    rect.setAttribute("height", h);
    rect.setAttribute("fill", "none");
    rect.setAttribute("stroke", "#42a5f5");
    rect.setAttribute("stroke-width", 2);
    rect.setAttribute("rx", 3);
    rect.setAttribute("filter", "url(#blueGlow)");
    rect.classList.add("scrub-bar-highlight");

    // Add glow filter if not present
    if (!svg.querySelector("#blueGlow")) {
        const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
        defs.innerHTML = `<filter id="blueGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feFlood flood-color="#42a5f5" flood-opacity="0.6" result="color"/>
            <feComposite in="color" in2="blur" operator="in" result="glow"/>
            <feMerge><feMergeNode in="glow"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>`;
        svg.insertBefore(defs, svg.firstChild);
    }

    svg.appendChild(rect);
}

function drawScrubLine(svg, x, y, height, color) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x);
    line.setAttribute("y1", y);
    line.setAttribute("x2", x);
    line.setAttribute("y2", y + height);
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", 2);
    line.classList.add("scrub-line");
    svg.appendChild(line);
}

function computeSecondaryTime(primaryT, primaryIsA, segments) {
    // Find which segment primaryT falls in
    for (const seg of segments) {
        if (seg.status === "match") {
            const pStart = primaryIsA ? seg.a_start : seg.b_start;
            const pEnd = primaryIsA ? seg.a_end : seg.b_end;
            const sStart = primaryIsA ? seg.b_start : seg.a_start;
            const sEnd = primaryIsA ? seg.b_end : seg.a_end;

            if (primaryT >= pStart && primaryT <= pEnd) {
                const frac = (pEnd > pStart) ? (primaryT - pStart) / (pEnd - pStart) : 0;
                return {
                    secondaryT: sStart + frac * (sEnd - sStart),
                    inMatch: true,
                };
            }
        }
    }

    // Not in a match segment -- find nearest preceding match
    let lastMatchPEnd = null;
    let lastMatchSEnd = null;
    for (const seg of segments) {
        if (seg.status !== "match") continue;
        const pEnd = primaryIsA ? seg.a_end : seg.b_end;
        const sEnd = primaryIsA ? seg.b_end : seg.a_end;
        if (pEnd <= primaryT) {
            lastMatchPEnd = pEnd;
            lastMatchSEnd = sEnd;
        }
    }

    if (lastMatchPEnd !== null && lastMatchSEnd !== null) {
        const offset = primaryT - lastMatchPEnd;
        return { secondaryT: lastMatchSEnd + offset, inMatch: false };
    }

    return { secondaryT: null, inMatch: false };
}

function doScrub() {
    if (!scrubState || !scrubState._lastResult) return;

    const primaryIsA = scrubState._lastPrimaryIsA;
    const primaryT = scrubState._lastPrimaryT;
    const result = scrubState._lastResult;

    const viewer = document.getElementById("scrub-viewer");
    viewer.className = "scrub-viewer";

    const wrapA = document.getElementById("scrub-wrap-a");
    const wrapB = document.getElementById("scrub-wrap-b");
    const imgA = document.getElementById("scrub-img-a");
    const imgB = document.getElementById("scrub-img-b");
    const timeA = document.getElementById("scrub-time-a");
    const timeB = document.getElementById("scrub-time-b");

    // Determine timestamps for A and B
    let tA, tB;
    if (primaryIsA) {
        tA = primaryT;
        tB = result.secondaryT;
    } else {
        tB = primaryT;
        tA = result.secondaryT;
    }

    // Set glows
    wrapA.className = "scrub-frame-wrap";
    wrapB.className = "scrub-frame-wrap";

    if (primaryIsA) {
        wrapA.classList.add("glow-blue");
        wrapB.classList.add(result.inMatch ? "glow-green" : "glow-red");
    } else {
        wrapB.classList.add("glow-blue");
        wrapA.classList.add(result.inMatch ? "glow-green" : "glow-red");
    }

    // Update times
    timeA.textContent = tA !== null ? formatDuration(tA) : "--:--";
    timeB.textContent = tB !== null ? formatDuration(tB) : "--:--";

    // Fetch frames
    if (tA !== null) {
        const url = "/api/frame?file=" + encodeURIComponent(scrubState.fileA) + "&t=" + tA.toFixed(1);
        if (imgA.dataset.url !== url) {
            imgA.dataset.url = url;
            imgA.src = url;
        }
    } else {
        imgA.removeAttribute("src");
        imgA.dataset.url = "";
    }

    if (tB !== null) {
        const url = "/api/frame?file=" + encodeURIComponent(scrubState.fileB) + "&t=" + tB.toFixed(1);
        if (imgB.dataset.url !== url) {
            imgB.dataset.url = url;
            imgB.src = url;
        }
    } else {
        imgB.removeAttribute("src");
        imgB.dataset.url = "";
    }
}

// ---- Rendering helpers ----

function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ":" + String(s).padStart(2, "0");
}

function renderDiffViz(svgId, segments, aDuration, bDuration) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";

    const width = svg.clientWidth || 850;
    const height = 160;
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);

    const padding = 20;
    const barHeight = 28;
    const barY_A = 20;
    const barY_B = height - barHeight - 20;
    const barWidth = width - 2 * padding;

    drawRect(svg, padding, barY_A, barWidth, barHeight, "#333");
    drawRect(svg, padding, barY_B, barWidth, barHeight, "#333");

    drawText(svg, 4, barY_A + barHeight / 2 + 4, "A", "#888", 12);
    drawText(svg, 4, barY_B + barHeight / 2 + 4, "B", "#888", 12);

    for (const seg of segments) {
        if (seg.status === "match") {
            const ax1 = padding + (seg.a_start / aDuration) * barWidth;
            const ax2 = padding + (seg.a_end / aDuration) * barWidth;
            const bx1 = padding + (seg.b_start / bDuration) * barWidth;
            const bx2 = padding + (seg.b_end / bDuration) * barWidth;

            drawRect(svg, ax1, barY_A, ax2 - ax1, barHeight, "#2e7d32");
            drawRect(svg, bx1, barY_B, bx2 - bx1, barHeight, "#2e7d32");

            drawConnection(svg, ax1, barY_A + barHeight, bx1, barY_B, "rgba(76, 175, 80, 0.3)");
            drawConnection(svg, ax2, barY_A + barHeight, bx2, barY_B, "rgba(76, 175, 80, 0.3)");
            drawConnectionFill(svg, ax1, ax2, barY_A + barHeight, bx1, bx2, barY_B, "rgba(76, 175, 80, 0.08)");

        } else if (seg.status === "a_only") {
            const ax1 = padding + (seg.a_start / aDuration) * barWidth;
            const ax2 = padding + (seg.a_end / aDuration) * barWidth;
            const color = seg.micro ? "#8d4444" : "#c62828";
            drawRect(svg, ax1, barY_A, ax2 - ax1, barHeight, color);
        } else if (seg.status === "b_only") {
            const bx1 = padding + (seg.b_start / bDuration) * barWidth;
            const bx2 = padding + (seg.b_end / bDuration) * barWidth;
            const color = seg.micro ? "#8d4444" : "#c62828";
            drawRect(svg, bx1, barY_B, bx2 - bx1, barHeight, color);
        }
    }
}

function drawRect(svg, x, y, w, h, fill) {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", Math.max(w, 1));
    rect.setAttribute("height", h);
    rect.setAttribute("fill", fill);
    rect.setAttribute("rx", 3);
    svg.appendChild(rect);
}

function drawText(svg, x, y, text, fill, size) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
    el.setAttribute("x", x);
    el.setAttribute("y", y);
    el.setAttribute("fill", fill);
    el.setAttribute("font-size", size);
    el.setAttribute("font-family", "sans-serif");
    el.textContent = text;
    svg.appendChild(el);
}

function drawConnection(svg, x1, y1, x2, y2, stroke) {
    const midY = (y1 + y2) / 2;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`);
    path.setAttribute("stroke", stroke);
    path.setAttribute("stroke-width", 1.5);
    path.setAttribute("fill", "none");
    svg.appendChild(path);
}

function drawConnectionFill(svg, ax1, ax2, ay, bx1, bx2, by, fill) {
    const midY = (ay + by) / 2;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const d = `M ${ax1} ${ay} C ${ax1} ${midY}, ${bx1} ${midY}, ${bx1} ${by} ` +
              `L ${bx2} ${by} C ${bx2} ${midY}, ${ax2} ${midY}, ${ax2} ${ay} Z`;
    path.setAttribute("d", d);
    path.setAttribute("fill", fill);
    path.setAttribute("stroke", "none");
    svg.appendChild(path);
}
