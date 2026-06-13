// ── Sample texts ────────────────────────────────────────────────────────────
const SAMPLE_TEXTS = {
    rag_intro: "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models (LLMs) by retrieving relevant documents from an external knowledge base. By anchoring the LLM's responses on retrieved facts, RAG minimizes hallucinations. It combines the strengths of retrieval-based models and generative models. This allows models to access dynamic, up-to-date information without requiring expensive retraining.",
    faiss_info: "FAISS (Facebook AI Similarity Search) is a library developed by Meta for efficient similarity search of dense vectors. It is written in C++ with Python bindings. FAISS can index billions of vectors and search them in milliseconds. In RAG applications, FAISS stores the vector embeddings of text chunks. When a user submits a query, FAISS quickly retrieves the closest vector matches using distance metrics like L2 distance or inner product.",
    ml_ai: "Artificial Intelligence (AI) is the broad science of mimicking human abilities. Machine Learning (ML) is a specific subset of AI that trains machines to learn from data. Deep Learning is further a subset of ML based on neural networks. RAG architectures use machine learning models for both embeddings (to translate words to numbers) and generator models (to synthesize text replies). RAG builds upon advanced ML techniques to create intelligent QA agents."
};

// ── State ────────────────────────────────────────────────────────────────────
let appState = { chunks: [], ingestedChunks: [], queryResults: [] };

// ── DOM refs ─────────────────────────────────────────────────────────────────
const docSelector = document.getElementById("doc-selector");
const docTextarea = document.getElementById("document-text");
const fileUploader = document.getElementById("file-uploader");
const chunkMethod = document.getElementById("chunk-method");
const fixedParamsGroup = document.getElementById("fixed-params-group");
const chunkSizeInput = document.getElementById("chunk-size");
const chunkOverlapInput = document.getElementById("chunk-overlap");
const sepKeywordSelect = document.getElementById("sep-keyword");
const vectorStoreSelect = document.getElementById("vector-store-select");

const btnChunk = document.getElementById("btn-chunk");
const btnIngest = document.getElementById("btn-ingest");
const btnQuery = document.getElementById("btn-query");

const queryInput = document.getElementById("query-input");
const searchMethodSel = document.getElementById("search-method");
const topKInput = document.getElementById("top-k");
const rerankModelSel = document.getElementById("rerank-model");
const rerankTopKInput = document.getElementById("rerank-top-k");

const chunksListContainer = document.getElementById("chunks-list-container");
const chunksBadge = document.getElementById("chunks-badge");
const chunksReadyCount = document.getElementById("chunks-ready-count");
const indexStatus = document.getElementById("index-status");
const vectorsIndexed = document.getElementById("vectors-indexed");
const vectorsListContainer = document.getElementById("vectors-list-container");

const activeQueryText = document.getElementById("active-query-text");
const retrievedList = document.getElementById("retrieved-list");
const rerankedList = document.getElementById("reranked-list");

const loadingOverlay = document.getElementById("loading-overlay");
const loaderMessage = document.getElementById("loader-message");

const tabBtns = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");

// Theme Toggle
const themeToggle = document.getElementById("theme-toggle");
const themeIcon = document.getElementById("theme-icon");
const themeLabel = document.getElementById("theme-label");

themeToggle?.addEventListener("click", () => {
    const currentTheme =
        document.documentElement.getAttribute("data-theme");

    if (currentTheme === "dark") {
        document.documentElement.removeAttribute("data-theme");
        themeIcon.textContent = "🌙";
        themeLabel.textContent = "Dark mode";
        localStorage.setItem("theme", "light");
    } else {
        document.documentElement.setAttribute("data-theme", "dark");
        themeIcon.textContent = "☀️";
        themeLabel.textContent = "Light mode";
        localStorage.setItem("theme", "dark");
    }
});

// Load saved theme
const savedTheme = localStorage.getItem("theme");
if (savedTheme === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
    themeIcon.textContent = "☀️";
    themeLabel.textContent = "Light mode";
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    docSelector.addEventListener("change", handleDocSelect);
    fileUploader.addEventListener("change", handleFileUpload);
    handleDocSelect();

    chunkMethod.addEventListener("change", () => {
        fixedParamsGroup.style.display = chunkMethod.value === "fixed" ? "block" : "none";
    });

    btnChunk.addEventListener("click", runChunking);
    btnIngest.addEventListener("click", runIngestion);
    btnQuery.addEventListener("click", runQuery);

    vectorStoreSelect.addEventListener("change", () => {
        updateStepLabel();
        updateSearchMethods();
        resetVectorStoreIndex();
    });
    updateStepLabel();
    updateSearchMethods();

    // Setup tabs listeners
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-target");
            switchTab(targetId);
        });
    });
});

function handleDocSelect() {
    const v = docSelector.value;
    docTextarea.value = (v && v !== "custom") ? SAMPLE_TEXTS[v] : "";
    resetProgress();
}

async function uploadFileToBackend(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
    });

    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "File upload failed");
    }

    const data = await res.json();
    return data.content;
}


// ── Tabs Switching ───────────────────────────────────────────────────────────
function switchTab(targetId) {
    tabBtns.forEach(btn => {
        if (btn.getAttribute("data-target") === targetId) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });
    tabPanels.forEach(panel => {
        if (panel.id === targetId) {
            panel.classList.add("active-panel");
        } else {
            panel.classList.remove("active-panel");
        }
    });
}

// ── Loader ───────────────────────────────────────────────────────────────────
function showLoader(msg) { loaderMessage.textContent = msg; loadingOverlay.style.display = "flex"; }
function hideLoader() { loadingOverlay.style.display = "none"; }

// ── Progress tracker ─────────────────────────────────────────────────────────
const STEP_IDS = ["step-doc", "step-chunk", "step-embed", "step-faiss", "step-retrieve", "step-rerank"];
const CONN_IDS = ["conn-1", "conn-2", "conn-3", "conn-4", "conn-5"];

function resetProgress() {
    STEP_IDS.forEach((id, i) => {
        const el = document.getElementById(id);
        if (el) el.className = "tracker-step" + (i === 0 ? " active" : "");
    });
    CONN_IDS.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.className = "tracker-connector";
    });

    // Clear state
    appState = { chunks: [], ingestedChunks: [], queryResults: [] };

    // Clear UI containers
    chunksListContainer.innerHTML = "";
    chunksListContainer.style.display = "none";
    document.getElementById("placeholder-chunks").style.display = "flex";
    chunksBadge.textContent = "0";

    vectorsListContainer.innerHTML = "";
    vectorsListContainer.style.display = "none";
    document.getElementById("placeholder-vectors").style.display = "flex";

    document.getElementById("results-visualizer-container").style.display = "none";
    document.getElementById("placeholder-results").style.display = "flex";

    // Disable downstream items
    btnIngest.disabled = true;
    btnQuery.disabled = true;
    [queryInput, searchMethodSel, topKInput, rerankModelSel, rerankTopKInput].forEach(el => el.disabled = true);

    // Reset status pills
    chunksReadyCount.textContent = "0 Chunks";
    chunksReadyCount.classList.remove("ready");
    indexStatus.textContent = "Not Built";
    indexStatus.classList.remove("ready");
    vectorsIndexed.textContent = "-";
}

function resetVectorStoreIndex() {
    ["step-embed", "step-faiss", "step-retrieve", "step-rerank"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.className = "tracker-step";
    });
    ["conn-2", "conn-3", "conn-4", "conn-5"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.className = "tracker-connector";
    });

    const stepChunk = document.getElementById("step-chunk");
    if (stepChunk) {
        stepChunk.className = "tracker-step active";
    }

    appState.ingestedChunks = [];
    appState.queryResults = [];

    vectorsListContainer.innerHTML = "";
    vectorsListContainer.style.display = "none";
    document.getElementById("placeholder-vectors").style.display = "flex";

    document.getElementById("results-visualizer-container").style.display = "none";
    document.getElementById("placeholder-results").style.display = "flex";

    btnQuery.disabled = true;
    [queryInput, searchMethodSel, topKInput, rerankModelSel, rerankTopKInput].forEach(el => el.disabled = true);

    indexStatus.textContent = "Not Built";
    indexStatus.classList.remove("ready");
    vectorsIndexed.textContent = "-";
}

function updateStepLabel() {
    const stepFaissLabel = document.querySelector("#step-faiss .step-label");
    if (stepFaissLabel) {
        if (vectorStoreSelect.value === "qdrant") {
            stepFaissLabel.textContent = "Qdrant Index";
        } else {
            stepFaissLabel.textContent = "FAISS Index";
        }
    }
}

function updateSearchMethods() {
    const bm25Option = document.getElementById("option-bm25");
    if (!bm25Option) return;

    if (vectorStoreSelect.value === "qdrant") {
        bm25Option.disabled = false;
        bm25Option.style.display = "block";
    } else {
        bm25Option.disabled = true;
        bm25Option.style.display = "none";
        if (searchMethodSel.value === "bm25") {
            searchMethodSel.value = "similarity";
        }
    }
}

function completeStep(stepId, connId, nextStepId) {
    const s = document.getElementById(stepId);
    if (s) { s.classList.remove("active"); s.classList.add("done"); }
    if (connId) { const c = document.getElementById(connId); if (c) c.classList.add("done"); }
    const n = document.getElementById(nextStepId);
    if (n) n.classList.add("active");
}

// ── Chunking ─────────────────────────────────────────────────────────────────
async function runChunking() {
    const text = docTextarea.value.trim();
    if (!text) { alert("Please enter some document text."); return; }

    showLoader("Splitting document into chunks…");
    resetProgress();

    const payload = {
        text,
        method: chunkMethod.value,
        chunk_size: parseInt(chunkSizeInput.value),
        chunk_overlap: parseInt(chunkOverlapInput.value),
        separate_keywords: sepKeywordSelect.value,
        sep_value: null
    };

    try {
        const res = await fetch("/api/chunk", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Chunking failed."); }

        const data = await res.json();
        appState.chunks = data.chunks;

        renderChunks(data.chunks);
        btnIngest.disabled = false;

        // Lock downstream controls
        [queryInput, searchMethodSel, topKInput, rerankModelSel, rerankTopKInput].forEach(el => el.disabled = true);
        btnQuery.disabled = true;

        // Update count pill
        chunksReadyCount.textContent = `${data.chunks.length} chunks`;
        chunksReadyCount.classList.add("ready");

        completeStep("step-doc", "conn-1", "step-chunk");
        switchTab("panel-chunks-view");
    } catch (e) {
        alert("Chunking error: " + e.message);
    } finally {
        hideLoader();
    }
}

function renderChunks(chunks) {
    chunksListContainer.innerHTML = "";
    chunksBadge.textContent = chunks.length;

    // Hide placeholder and show list
    document.getElementById("placeholder-chunks").style.display = "none";
    chunksListContainer.style.display = "grid";

    chunks.forEach((text, i) => {
        const card = document.createElement("div");
        card.className = "chunk-card";
        card.style.animationDelay = `${i * 0.03}s`;
        card.innerHTML = `
            <div class="chunk-meta">
                <span class="chunk-badge">CHUNK ${String(i + 1).padStart(2, "0")}</span>
                <span class="chunk-size-tag">${text.length} chars</span>
            </div>
            <p class="chunk-content">${escapeHTML(text)}</p>
        `;
        chunksListContainer.appendChild(card);
    });
}

// ── Ingestion ─────────────────────────────────────────────────────────────────
async function runIngestion() {
    if (!appState.chunks.length) return;
    const backend = vectorStoreSelect.value;
    const backendName = backend === "qdrant" ? "Qdrant" : "FAISS";
    showLoader(`Generating embeddings & building ${backendName} index…`);

    try {
        const res = await fetch(`/api/ingest?backend=${backend}`, { method: "POST" });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Ingestion failed."); }

        const data = await res.json();
        appState.ingestedChunks = data;

        renderVectors(data);

        // Update status UI
        indexStatus.textContent = "Built";
        indexStatus.classList.add("ready");
        vectorsIndexed.textContent = `${data.length} vectors`;

        // Enable query controls
        [queryInput, searchMethodSel, topKInput, rerankModelSel, rerankTopKInput].forEach(el => el.disabled = false);
        btnQuery.disabled = false;

        completeStep("step-chunk", "conn-2", "step-embed");
        setTimeout(() => {
            completeStep("step-embed", "conn-3", "step-faiss");
            switchTab("panel-vectors-view");
        }, 500);
    } catch (e) {
        alert("Ingestion error: " + e.message);
    } finally {
        hideLoader();
    }
}

// Helper to render high-dimensional embedding weight heatmaps
function createVectorHeatmap(embedding) {
    const container = document.createElement("div");
    container.className = "vector-heatmap";

    // We visualize up to 128 elements to avoid overloading the DOM, 
    // while representing a statistically significant representation of the vector.
    const subset = embedding.slice(0, 128);

    subset.forEach((val, idx) => {
        const cell = document.createElement("div");
        cell.className = "heatmap-cell";

        const absVal = Math.abs(val);
        // Map embedding amplitude to opacity
        const opacity = Math.min(0.15 + absVal * 8, 1);

        if (val >= 0) {
            cell.style.backgroundColor = `rgba(6, 182, 212, ${opacity})`;
            cell.style.color = `rgba(6, 182, 212, 1)`;
        } else {
            cell.style.backgroundColor = `rgba(244, 63, 94, ${opacity})`;
            cell.style.color = `rgba(244, 63, 94, 1)`;
        }
        cell.title = `Dimension ${idx + 1}: ${val.toFixed(5)}`;
        container.appendChild(cell);
    });

    return container;
}

function renderVectors(list) {
    vectorsListContainer.innerHTML = "";
    document.getElementById("placeholder-vectors").style.display = "none";
    vectorsListContainer.style.display = "flex";

    list.forEach(item => {
        const snippet = item.embedding.slice(0, 6).map(v => v.toFixed(5)).join(", ");
        const fullStr = item.embedding.map(v => v.toFixed(5)).join(", ");

        const el = document.createElement("div");
        el.className = "vector-item";
        el.style.animationDelay = `${item.chunk_id * 0.04}s`;
        el.innerHTML = `
            <div class="vector-item-header">
                <span class="vector-title">Chunk ${item.chunk_id + 1}</span>
                <span class="vector-dimensions">${item.embedding.length}d · ${item.embedding_method}</span>
            </div>
            <div class="vector-chunk-preview">"${escapeHTML(truncate(item.chunk, 75))}"</div>
            <div class="vector-array-container" id="vec-${item.chunk_id}">
                <span class="vec-text">[ ${snippet}, … ]</span>
            </div>
            <div class="heatmap-wrapper"></div>
        `;

        // Render heatmap visualization
        const heatmap = createVectorHeatmap(item.embedding);
        el.querySelector(".heatmap-wrapper").appendChild(heatmap);

        const box = el.querySelector(".vector-array-container");
        const text = el.querySelector(".vec-text");
        box.addEventListener("click", () => {
            box.classList.toggle("expanded");
            text.textContent = box.classList.contains("expanded")
                ? `[ ${fullStr} ]`
                : `[ ${snippet}, … ]`;
        });
        vectorsListContainer.appendChild(el);
    });
}

// ── Query ─────────────────────────────────────────────────────────────────────
async function runQuery() {
    const query = queryInput.value.trim();
    if (!query) { alert("Please enter a search query."); return; }

    showLoader("Retrieving chunks & applying cross-encoder reranking…");

    const payload = {
        query,
        method: searchMethodSel.value,
        topk: parseInt(topKInput.value),
        reranker_model: rerankModelSel.value,
        rerank_topk: parseInt(rerankTopKInput.value),
        backend: vectorStoreSelect.value
    };

    try {
        const res = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Query failed."); }

        const data = await res.json();
        appState.queryResults = data;

        renderResults(data, query);

        completeStep("step-faiss", "conn-4", "step-retrieve");
        setTimeout(() => {
            completeStep("step-retrieve", "conn-5", "step-rerank");
            switchTab("panel-results-view");
        }, 500);
    } catch (e) {
        alert("Query error: " + e.message);
    } finally {
        hideLoader();
    }
}

function renderResults(results, query) {
    retrievedList.innerHTML = "";
    rerankedList.innerHTML = "";

    activeQueryText.textContent = `"${query}"`;
    document.getElementById("placeholder-results").style.display = "none";
    document.getElementById("results-visualizer-container").style.display = "flex";

    const retrieved = results
        .filter(r => r.before_reranker !== null)
        .sort((a, b) => a.before_reranker.rank - b.before_reranker.rank);

    const reranked = results
        .filter(r => r.after_reranker !== null)
        .sort((a, b) => a.after_reranker.rank - b.after_reranker.rank);

    // Render Retrieval column (left)
    retrieved.forEach((item, i) => {
        const label = searchMethodSel.value === "similarity" ? "L2 dist" : "Score";
        const card = document.createElement("div");
        card.className = "result-card";
        card.setAttribute("data-chunk-id", item.chunk_id);
        card.style.animationDelay = `${i * 0.04}s`;
        card.innerHTML = `
            <div class="result-card-header">
                <span class="result-rank-badge column-1-badge">Rank ${item.before_reranker.rank}</span>
                <span class="result-score">${label}: ${item.before_reranker.score.toFixed(4)}</span>
            </div>
            <p class="chunk-content">${escapeHTML(item.chunk)}</p>
            <div class="result-chunk-id">
                <span>Chunk #${item.chunk_id + 1}</span>
            </div>
        `;

        // Highlight logic on hover
        card.addEventListener("mouseenter", () => {
            document.querySelectorAll(`.result-card[data-chunk-id="${item.chunk_id}"]`).forEach(c => {
                c.classList.add("highlight-hover");
            });
        });
        card.addEventListener("mouseleave", () => {
            document.querySelectorAll(`.result-card[data-chunk-id="${item.chunk_id}"]`).forEach(c => {
                c.classList.remove("highlight-hover");
            });
        });

        retrievedList.appendChild(card);
    });

    // Render Reranked column (right)
    reranked.forEach((item, i) => {
        let shiftHTML = "";
        let movementClass = "rank-same";
        let cardClass = "";

        if (item.before_reranker) {
            const diff = item.before_reranker.rank - item.after_reranker.rank;
            if (diff > 0) {
                shiftHTML = `↑ +${diff}`;
                movementClass = "rank-up";
                cardClass = "rank-change-up";
            } else if (diff < 0) {
                shiftHTML = `↓ ${diff}`;
                movementClass = "rank-down";
                cardClass = "rank-change-down";
            } else {
                shiftHTML = `same`;
                movementClass = "rank-same";
            }
        } else {
            shiftHTML = `↑ new`;
            movementClass = "rank-up";
            cardClass = "rank-change-up";
        }

        const card = document.createElement("div");
        card.className = `result-card ${cardClass}`;
        card.setAttribute("data-chunk-id", item.chunk_id);
        card.style.animationDelay = `${i * 0.04}s`;
        card.innerHTML = `
            <div class="result-card-header">
                <span class="result-rank-badge column-2-badge">Rank ${item.after_reranker.rank}</span>
                <span class="result-rank-movement ${movementClass}">${shiftHTML}</span>
                <span class="result-score">CE Score: ${item.after_reranker.score.toFixed(4)}</span>
            </div>
            <p class="chunk-content">${escapeHTML(item.chunk)}</p>
            <div class="result-chunk-id">
                <span>Chunk #${item.chunk_id + 1}</span>
            </div>
        `;

        // Highlight logic on hover
        card.addEventListener("mouseenter", () => {
            document.querySelectorAll(`.result-card[data-chunk-id="${item.chunk_id}"]`).forEach(c => {
                c.classList.add("highlight-hover");
            });
        });
        card.addEventListener("mouseleave", () => {
            document.querySelectorAll(`.result-card[data-chunk-id="${item.chunk_id}"]`).forEach(c => {
                c.classList.remove("highlight-hover");
            });
        });

        rerankedList.appendChild(card);
    });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escapeHTML(str) {
    return str
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function truncate(str, n) {
    return str.length <= n ? str : str.slice(0, n) + "…";
}