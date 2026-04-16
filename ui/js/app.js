/* ============================================================
   BONSAICHAT — APP.JS   (agent-zero-features branch)
   Local-only. No provider/API key logic.
   Additions: model download flow, HF config, sub-agent toggle.
   ============================================================ */

// ── State ─────────────────────────────────────────────────────────────────
let currentBotMsgId = null;
let botBuffers       = {};

// GenUI: Typing speed
let lastKeyTime  = 0;
let keyIntervals = [];
const TYPING_SAMPLE_SIZE = 10;
const SLOW_THRESHOLD_MS  = 400;
const FAST_THRESHOLD_MS  = 150;

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener('pywebviewready', async () => {
    const history = await window.pywebview.api.load_history();
    history.forEach(msg => appendMessage(msg.role, msg.content, false));
    setupDragAndDrop();
    setupTypingSpeedDetection();
    animateHeader();
    loadSessionList();
    triggerBonsaiAutoSetup();
    setupScrollSync();
});

// ── Session management ────────────────────────────────────────────────────
async function newChat() {
    const result = await window.pywebview.api.new_session();
    if (result.status === 'success') {
        clearChatUI();
        loadSessionList();
    }
}

function clearChatUI() {
    document.getElementById('chat-history').innerHTML = '';
    currentBotMsgId = null;
    botBuffers = {};
    checkpointedMessages.clear();
    document.getElementById('checkpoint-blocks').innerHTML = '';
}

async function loadSessionList() {
    const sessions  = await window.pywebview.api.list_sessions();
    const list      = document.getElementById('session-list');
    if (!sessions || sessions.length === 0) {
        list.innerHTML = '<div class="no-sessions">No previous chats</div>';
        return;
    }
    const currentId = await window.pywebview.api.get_current_session_id();
    let html = '';
    sessions.forEach(session => {
        const date        = new Date(session.timestamp).toLocaleDateString();
        const activeClass = session.id === currentId ? 'active' : '';
        const safeTitle   = session.title.replace(/</g, "&lt;").replace(/>/g, "&gt;");
        html += `
            <div class="session-item ${activeClass}" onclick="switchSession('${session.id}')">
                <div class="session-title">${safeTitle}</div>
                <div class="session-date">${date}</div>
            </div>`;
    });
    list.innerHTML = html;
}

async function switchSession(sessionId) {
    const result = await window.pywebview.api.switch_session(sessionId);
    if (result.status === 'success') {
        clearChatUI();
        const history = await window.pywebview.api.load_history();
        history.forEach(msg => appendMessage(msg.role, msg.content, false));
        loadSessionList();
    }
}

// ── GenUI: Typing speed ───────────────────────────────────────────────────
function setupTypingSpeedDetection() {
    const input = document.getElementById('user-input');
    input.addEventListener('keydown', (e) => {
        if (e.key.length !== 1 && e.key !== 'Backspace') return;
        const now = Date.now();
        if (lastKeyTime > 0) {
            keyIntervals.push(now - lastKeyTime);
            if (keyIntervals.length > TYPING_SAMPLE_SIZE) keyIntervals.shift();
            if (keyIntervals.length >= 5) {
                const avg = keyIntervals.reduce((a, b) => a + b, 0) / keyIntervals.length;
                applyTypingTheme(avg);
            }
        }
        lastKeyTime = now;
    });
    input.addEventListener('blur', () => { keyIntervals = []; lastKeyTime = 0; });
}

function applyTypingTheme(avgInterval) {
    document.body.classList.remove('typing-slow', 'typing-fast');
    if (avgInterval > SLOW_THRESHOLD_MS)      document.body.classList.add('typing-slow');
    else if (avgInterval < FAST_THRESHOLD_MS) document.body.classList.add('typing-fast');
}

// ── GenUI: Tone styling ───────────────────────────────────────────────────
function applyToneToMessage(messageId, tone) {
    const el = document.getElementById(messageId);
    if (!el || !tone) return;
    el.classList.remove('tone-calm', 'tone-excited', 'tone-serious', 'tone-playful');
    const cls = `tone-${tone.toLowerCase()}`;
    if (['tone-calm','tone-excited','tone-serious','tone-playful'].includes(cls))
        el.classList.add(cls);
}

// ── Sidebar ───────────────────────────────────────────────────────────────
let currentSidebarView = 'chats';

function toggleSidebar(view = null) {
    const sidebar = document.getElementById('sidebar');
    if (!view) {
        sidebar.classList.remove('visible');
        updateSidebarPosition();
        return;
    }
    if (!sidebar.classList.contains('visible') || currentSidebarView !== view) {
        document.getElementById('view-chats').style.display    = view === 'chats'    ? 'block' : 'none';
        document.getElementById('view-settings').style.display = view === 'settings' ? 'block' : 'none';
        document.getElementById('sidebar-title').textContent   = view === 'chats' ? 'Chats' : 'Settings';
        document.getElementById('tab-chats').classList.toggle('active',    view === 'chats');
        document.getElementById('tab-settings').classList.toggle('active', view === 'settings');
        currentSidebarView = view;
        sidebar.classList.add('visible');
    } else {
        sidebar.classList.remove('visible');
        document.getElementById('tab-chats').classList.remove('active');
        document.getElementById('tab-settings').classList.remove('active');
    }
    updateSidebarPosition();
}

function updateSidebarPosition() {
    const sidebar = document.getElementById('sidebar');
    anime({
        targets:    sidebar,
        translateX: sidebar.classList.contains('visible') ? ['-100%','0%'] : ['0%','-100%'],
        duration:   350,
        easing:     'easeOutQuad',
    });
}

// ── Header ────────────────────────────────────────────────────────────────
function animateHeader() {
    anime({ targets: '.logo', translateY: [-8,0], opacity: [0,1], duration: 600, easing: 'easeOutQuad' });
}

// ── Multi-agent toggle ────────────────────────────────────────────────────
async function toggleAgents() {
    const enabled = document.getElementById('agent-toggle').checked;
    await window.pywebview.api.toggle_multi_agent(enabled);
}

// ── RAG / file handling ───────────────────────────────────────────────────
async function clearRag() {
    const res = await window.pywebview.api.clear_rag_context();
    document.getElementById('file-list').innerHTML = '';
    document.getElementById('sidebar-file-list').innerHTML = '';
    alert(res);
}

function setupDragAndDrop() {
    const dz = document.getElementById('drop-zone');
    ['dragenter','dragover','dragleave','drop'].forEach(evt =>
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }));
    dz.addEventListener('dragover',  () => dz.classList.add('active'));
    dz.addEventListener('dragleave', () => dz.classList.remove('active'));
    dz.addEventListener('drop',      e  => processFiles(e.dataTransfer.files));
}

function handleFileSelect(e) { processFiles(e.target.files); }

async function processFiles(filesList) {
    const dz       = document.getElementById('drop-zone');
    dz.classList.remove('active');
    const files    = Array.from(filesList);
    const uploadData = [];
    for (const file of files) {
        const reader  = new FileReader();
        const promise = new Promise(resolve => {
            reader.onload = e => resolve({ name: file.name, content: e.target.result });
            reader.readAsDataURL(file);
        });
        uploadData.push(await promise);
    }
    if (uploadData.length > 0) {
        dz.innerText = "Ingesting…";
        const res = await window.pywebview.api.upload_files(uploadData);
        if (res.status === 'success') {
            updateFileList(res.files);
            dz.innerText = "Files ready!";
            setTimeout(() => { dz.innerText = "Drag PDF/CSV here\nor Click to upload"; }, 3000);
        } else {
            alert("Error: " + res.message);
            dz.innerText = "Drag PDF/CSV here\nor Click to upload";
        }
    }
}

function updateFileList(files) {
    const html = files.map(f => `<div class="file-tag">${f}</div>`).join('');
    document.getElementById('file-list').innerHTML         = html;
    document.getElementById('sidebar-file-list').innerHTML = html;
    anime({ targets: '.file-tag', opacity: [0,1], translateY: [6,0],
            delay: anime.stagger(40), duration: 300, easing: 'easeOutQuad' });
}

// ── Chat ──────────────────────────────────────────────────────────────────
function handleEnter(e) { if (e.key === 'Enter') sendPrompt(); }

function sendPrompt() {
    const input = document.getElementById('user-input');
    const val   = input.value.trim();
    if (!val) return;
    input.value = '';
    appendMessage('user', val);
    const botId = 'bot-' + Date.now();
    currentBotMsgId = botId;
    botBuffers[botId] = "";
    createBotBubble(botId);
    keyIntervals = []; lastKeyTime = 0;
    window.pywebview.api.start_chat_stream(val);
}

function receiveChunk(chunk, targetId) {
    const id  = targetId || currentBotMsgId;
    const div = document.getElementById(id);
    if (div) {
        botBuffers[id] = (botBuffers[id] || "") + chunk;
        div.innerHTML  = marked.parse(botBuffers[id]);
        scrollToBottom();
    }
}

function createBotBubble(id) {
    const container = document.getElementById('chat-history');
    const wrapper   = document.createElement('div');
    wrapper.className = "message-wrapper bot-wrapper";
    wrapper.setAttribute('data-msg-id', id);
    wrapper.innerHTML = `
        <div class="message bot" id="${id}"><span class="loading-dots">Thinking</span></div>
        <button class="checkpoint-btn" onclick="toggleCheckpoint('${id}')" title="Checkpoint">✓</button>`;
    container.appendChild(wrapper);
    animateMessage(wrapper);
    scrollToBottom();
    createCheckpointBlock(id);
}

function clearBubble(id) {
    const div = document.getElementById(id);
    if (div) { div.innerHTML = ""; botBuffers[id] = ""; }
}

function appendMessage(role, text, animate = true) {
    if (role === 'bot') {
        const id = 'bot-' + Math.random().toString(36).substr(2, 9);
        botBuffers[id] = text;
        createBotBubble(id);
        document.getElementById(id).innerHTML = marked.parse(text);
    } else {
        const container = document.getElementById('chat-history');
        const wrapper   = document.createElement('div');
        wrapper.className = "message-wrapper user-wrapper";
        wrapper.innerHTML = `<div class="message user">${text.replace(/</g,"&lt;")}</div>`;
        container.appendChild(wrapper);
        if (animate) animateMessage(wrapper);
    }
    scrollToBottom();
}

function animateMessage(wrapper) {
    anime({ targets: wrapper, opacity: [0,1], translateY: [10,0], duration: 300, easing: 'easeOutQuad' });
}

function scrollToBottom() {
    const h = document.getElementById('chat-history');
    h.scrollTop = h.scrollHeight;
}

function receiveError(e) { alert("Error: " + e); }

function streamComplete(tone) {
    if (currentBotMsgId && tone) applyToneToMessage(currentBotMsgId, tone);
    updateCheckpointTooltip(currentBotMsgId);
    currentBotMsgId = null;
}

// ── Checkpoint sidebar ────────────────────────────────────────────────────
let checkpointedMessages = new Set();

function createCheckpointBlock(msgId) {
    const container = document.getElementById('checkpoint-blocks');
    const block     = document.createElement('div');
    block.className = 'checkpoint-block';
    block.id        = `checkpoint-${msgId}`;
    block.setAttribute('data-msg-id', msgId);
    block.setAttribute('data-tooltip', 'Loading…');
    block.onclick   = () => navigateToMessage(msgId);
    container.appendChild(block);
    anime({ targets: block, opacity: [0,1], translateX: [10,0], duration: 300, easing: 'easeOutQuad' });
}

function updateCheckpointTooltip(msgId) {
    const block  = document.getElementById(`checkpoint-${msgId}`);
    const msgDiv = document.getElementById(msgId);
    if (block && msgDiv) {
        const text    = msgDiv.textContent.trim();
        const preview = text.length > 30 ? text.substring(0, 30) + '…' : text;
        block.setAttribute('data-tooltip', preview || 'Answer');
    }
}

function toggleCheckpoint(msgId) {
    const btn   = document.querySelector(`.message-wrapper[data-msg-id="${msgId}"] .checkpoint-btn`);
    const block = document.getElementById(`checkpoint-${msgId}`);
    if (checkpointedMessages.has(msgId)) {
        checkpointedMessages.delete(msgId);
        btn?.classList.remove('checked');
        block?.classList.remove('checked');
    } else {
        checkpointedMessages.add(msgId);
        btn?.classList.add('checked');
        block?.classList.add('checked');
        if (block) anime({ targets: block, scale: [1.3,1], duration: 300, easing: 'easeOutBack' });
    }
}

function navigateToMessage(msgId) {
    const el = document.getElementById(msgId);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        anime({ targets: el, boxShadow: ['0 0 0 2px var(--accent)','0 0 0 0px transparent'],
                duration: 1000, easing: 'easeOutQuad' });
    }
}

function setupScrollSync() {
    const chatHistory     = document.getElementById('chat-history');
    const checkpointBlocks = document.getElementById('checkpoint-blocks');
    if (!chatHistory || !checkpointBlocks) return;

    chatHistory.addEventListener('scroll', () => {
        const wrappers  = chatHistory.querySelectorAll('.message-wrapper.bot-wrapper');
        const chatRect  = chatHistory.getBoundingClientRect();
        const chatCenter = chatRect.top + chatRect.height / 2;
        let closestWrapper = null, closestDistance = Infinity;

        wrappers.forEach(wrapper => {
            const rect     = wrapper.getBoundingClientRect();
            const distance = Math.abs(rect.top + rect.height / 2 - chatCenter);
            if (distance < closestDistance) { closestDistance = distance; closestWrapper = wrapper; }
        });

        document.querySelectorAll('.checkpoint-block').forEach(b => b.classList.remove('active'));
        if (closestWrapper) {
            const activeBlock = document.getElementById(`checkpoint-${closestWrapper.getAttribute('data-msg-id')}`);
            if (activeBlock) activeBlock.classList.add('active');
        }
    });
}

// ── Bonsai setup / download (agent-zero-features) ─────────────────────────
let _bonsaiSetupTriggered = false;

async function triggerBonsaiAutoSetup() {
    if (_bonsaiSetupTriggered) return;

    try {
        const status = await window.pywebview.api.get_local_model_status();
        if (status.server_running) {
            onBonsaiSetupProgress('ready', 100, 'Bonsai is ready');
            return;
        }
    } catch (e) { /* pywebview not ready yet */ }

    _bonsaiSetupTriggered = true;
    await window.pywebview.api.begin_auto_setup();
}

async function startModelDownload() {
    document.getElementById('btn-overlay-download').style.display = 'none';
    document.getElementById('btn-download-model').style.display   = 'none';
    document.getElementById('bonsai-setup-overlay').classList.add('visible');
    document.getElementById('setup-overlay-label').textContent = 'Starting download…';
    await window.pywebview.api.download_bonsai();
}

// Called from Python: onBonsaiSetupProgress(phase, pct, msg)
// Phases: 'downloading' | 'starting' | 'ready' | 'missing_model' | 'error'
function onBonsaiSetupProgress(phase, pct, msg) {
    const dot     = document.getElementById('bonsai-status-dot');
    const text    = document.getElementById('bonsai-status-text');
    const overlay = document.getElementById('bonsai-setup-overlay');
    const fill    = document.getElementById('setup-overlay-fill');
    const label   = document.getElementById('setup-overlay-label');

    if (phase === 'missing_model') {
        // Model not found — prompt the user to download
        overlay.classList.add('visible');
        fill.style.width = '0%';
        label.textContent = msg;
        document.getElementById('setup-overlay-sub').textContent =
            'The model file was not found. Download it now (~1 GB).';
        document.getElementById('btn-overlay-download').style.display = 'inline-block';
        document.getElementById('btn-download-model').style.display   = 'inline-block';
        dot.className    = 'status-dot status-offline';
        text.textContent = 'Model missing';
        _bonsaiSetupTriggered = false;

    } else if (phase === 'downloading') {
        overlay.classList.add('visible');
        fill.style.width  = Math.max(0, pct * 100) + '%';
        label.textContent = msg;
        dot.className     = 'status-dot status-busy';
        text.textContent  = `Downloading… ${pct > 0 ? (pct * 100).toFixed(1) + '%' : ''}`;

    } else if (phase === 'starting') {
        overlay.classList.remove('visible');
        dot.className    = 'status-dot status-busy';
        text.textContent = 'Loading model… (may take 2–5 min first time)';

    } else if (phase === 'ready') {
        overlay.classList.remove('visible');
        dot.className    = 'status-dot status-online';
        text.textContent = 'Bonsai is ready';
        _bonsaiSetupTriggered = false;

    } else if (phase === 'error') {
        overlay.classList.remove('visible');
        dot.className    = 'status-dot status-error';
        text.textContent = msg;
        _bonsaiSetupTriggered = false;
        const retryBtn = document.getElementById('btn-bonsai-retry');
        if (retryBtn) retryBtn.style.display = 'block';
    }
}

async function retryBonsaiSetup() {
    document.getElementById('btn-bonsai-retry').style.display = 'none';
    _bonsaiSetupTriggered = false;
    await triggerBonsaiAutoSetup();
}

// ── HF advanced config ────────────────────────────────────────────────────
let _hfConfigVisible = false;

async function toggleHfConfig() {
    _hfConfigVisible = !_hfConfigVisible;
    const section = document.getElementById('hf-config-section');
    section.style.display = _hfConfigVisible ? 'block' : 'none';

    if (_hfConfigVisible) {
        const env = await window.pywebview.api.get_setup_env();
        document.getElementById('hf-repo-input').value = env.hf_repo || '';
        document.getElementById('hf-file-input').value = env.hf_file || '';
    }
}

async function saveHfConfig() {
    const repo = document.getElementById('hf-repo-input').value.trim();
    const file = document.getElementById('hf-file-input').value.trim();
    if (repo) await window.pywebview.api.save_env_key('BONSAI_HF_REPO', repo);
    if (file) await window.pywebview.api.save_env_key('BONSAI_HF_FILE', file);
    alert('HF config saved. Restart BonsaiChat to apply.');
}
