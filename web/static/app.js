/* ─────────────────────────────────────────────────────────────────────────
   ComfyUI Agent — frontend logic
   WebSocket protocol:
     send:    {type:"message", content:"...", files:[{name, mime, b64}]}
     receive: {type:"token"|"tool_call"|"tool_result"|"workflow_update"|"done"|"error"}
 ─────────────────────────────────────────────────────────────────────────── */

'use strict';

// ─── globals ───────────────────────────────────────────────────────────────
let ws = null;
let currentBubble = null;   // streaming assistant bubble element
let currentText = '';        // accumulated text for current turn
let attachedFiles = [];      // {file, name, mime, b64}
let currentNodeId = null;    // node being edited in modal

// ─── DOM refs ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const msgList       = $('messages');
const msgInput      = $('msg-input');
const sendBtn       = $('send-btn');
const fileInput     = $('file-input');
const attachPreview = $('attach-preview');
const wsBadge       = $('ws-badge');
const comfyBadge    = $('comfy-badge');
const providerLabel = $('provider-label');
const workflowNodes = $('workflow-nodes');
const resourcesDiv  = $('resources-content');
const nodeModal     = $('node-modal');
const modalTitle    = $('modal-title');
const modalBody     = $('modal-body');

// ─── WebSocket ─────────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => setBadge(wsBadge, 'online', '● WS');
  ws.onclose = () => {
    setBadge(wsBadge, 'offline', '● WS');
    setTimeout(connectWS, 3000);
  };
  ws.onerror = () => setBadge(wsBadge, 'offline', '● WS');

  ws.onmessage = e => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    handleEvent(ev);
  };
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'token':
      appendToken(ev.content);
      break;
    case 'tool_call':
      finishAssistantBubble();
      appendToolBadge(`→ ${ev.name}  ${JSON.stringify(ev.input).slice(0, 90)}`, 'tool-badge');
      break;
    case 'tool_result':
      const summary = typeof ev.result === 'string'
        ? ev.result.slice(0, 120)
        : JSON.stringify(ev.result).slice(0, 120);
      appendToolBadge(`← ${ev.name}: ${summary}`, 'tool-result-badge');
      break;
    case 'workflow_update':
      renderWorkflow(ev.nodes);
      break;
    case 'done':
      finishAssistantBubble();
      setSending(false);
      break;
    case 'error':
      finishAssistantBubble();
      appendSystemMsg('⚠ ' + (ev.content || 'Unknown error'));
      setSending(false);
      break;
  }
}

// ─── Chat rendering ─────────────────────────────────────────────────────────
function appendToken(text) {
  if (!currentBubble) {
    const msg = mkEl('div', 'msg agent');
    currentBubble = mkEl('div', 'bubble');
    msg.appendChild(currentBubble);
    msgList.appendChild(msg);
  }
  // remove old cursor if any
  const old = currentBubble.querySelector('.cursor');
  if (old) old.remove();

  currentText += text;
  currentBubble.innerHTML = renderMarkdown(currentText);
  // re-add cursor
  const cur = mkEl('span', 'cursor');
  currentBubble.appendChild(cur);
  hljs.highlightAll();
  scrollBottom();
}

function finishAssistantBubble() {
  if (currentBubble) {
    const cur = currentBubble.querySelector('.cursor');
    if (cur) cur.remove();
    currentBubble.innerHTML = renderMarkdown(currentText);
    hljs.highlightAll();
    currentBubble = null;
    currentText = '';
  }
}

function appendUserMsg(content, files) {
  const msg = mkEl('div', 'msg user');
  if (files && files.length) {
    files.forEach(f => {
      if (f.mime.startsWith('image/')) {
        const img = document.createElement('img');
        img.src = `data:${f.mime};base64,${f.b64}`;
        img.className = 'msg-media';
        msg.appendChild(img);
      } else {
        const tag = mkEl('span', 'tool-badge');
        tag.textContent = `📎 ${f.name}`;
        msg.appendChild(tag);
      }
    });
  }
  if (content) {
    const b = mkEl('div', 'bubble');
    b.textContent = content;
    msg.appendChild(b);
  }
  msgList.appendChild(msg);
  scrollBottom();
}

function appendToolBadge(text, cls) {
  const msg = mkEl('div', 'msg system');
  const badge = mkEl('span', cls);
  badge.textContent = text;
  msg.appendChild(badge);
  msgList.appendChild(msg);
  scrollBottom();
}

function appendSystemMsg(text) {
  const msg = mkEl('div', 'msg system');
  const b = mkEl('div', 'bubble');
  b.textContent = text;
  msg.appendChild(b);
  msgList.appendChild(msg);
  scrollBottom();
}

// ─── Simple markdown renderer (no deps) ────────────────────────────────────
function renderMarkdown(src) {
  if (!src) return '';
  let html = src
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    // code blocks
    .replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code class="language-${lang||'plaintext'}">${code.trimEnd()}</code></pre>`)
    // inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // line breaks
    .replace(/\n/g, '<br>');
  return html;
}

// ─── File attachment ────────────────────────────────────────────────────────
fileInput.addEventListener('change', () => {
  Array.from(fileInput.files).forEach(readFile);
  fileInput.value = '';
});

function readFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const b64 = e.target.result.split(',')[1];
    const entry = { file, name: file.name, mime: file.type, b64 };
    attachedFiles.push(entry);
    renderAttachPreview();
  };
  reader.readAsDataURL(file);
}

function renderAttachPreview() {
  if (!attachedFiles.length) {
    attachPreview.classList.add('hidden');
    attachPreview.innerHTML = '';
    return;
  }
  attachPreview.classList.remove('hidden');
  attachPreview.innerHTML = '';
  attachedFiles.forEach((f, i) => {
    const thumb = mkEl('span', 'attach-thumb');
    const icon = f.mime.startsWith('image/') ? '🖼' : f.mime.startsWith('audio/') ? '🎵' : '🎬';
    thumb.innerHTML = `${icon} ${f.name} <span class="rm" data-i="${i}">✕</span>`;
    attachPreview.appendChild(thumb);
  });
  attachPreview.querySelectorAll('.rm').forEach(btn => {
    btn.onclick = () => {
      attachedFiles.splice(+btn.dataset.i, 1);
      renderAttachPreview();
    };
  });
}

// ─── Send message ───────────────────────────────────────────────────────────
function sendMessage() {
  const content = msgInput.value.trim();
  if (!content && !attachedFiles.length) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    appendSystemMsg('WebSocket 未连接，正在重连…');
    return;
  }

  appendUserMsg(content, attachedFiles);
  setSending(true);

  ws.send(JSON.stringify({
    type: 'message',
    content,
    files: attachedFiles.map(f => ({ name: f.name, mime: f.mime, b64: f.b64 })),
  }));

  msgInput.value = '';
  attachedFiles = [];
  renderAttachPreview();
  msgInput.style.height = 'auto';
}

sendBtn.addEventListener('click', sendMessage);
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 140) + 'px';
});

function setSending(sending) {
  sendBtn.disabled = sending;
  msgInput.disabled = sending;
}

// ─── Workflow panel ─────────────────────────────────────────────────────────
function renderWorkflow(nodes) {
  if (!nodes || !nodes.length) {
    workflowNodes.innerHTML = '<div class="empty-hint">暂无工作流</div>';
    return;
  }
  workflowNodes.innerHTML = '';
  nodes.forEach(n => {
    const card = mkEl('div', 'node-card');
    card.dataset.nodeId = n.id;
    card.innerHTML = `
      <div><span class="node-id">[${n.id}]</span> <span class="node-type">${n.class_type}</span></div>
      ${n.title ? `<div class="node-title">${n.title}</div>` : ''}
    `;
    card.addEventListener('click', () => openNodeModal(n));
    workflowNodes.appendChild(card);
  });
}

// ─── Node edit modal ─────────────────────────────────────────────────────────
function openNodeModal(node) {
  currentNodeId = node.id;
  modalTitle.textContent = `[${node.id}] ${node.class_type}`;
  modalBody.innerHTML = '';

  const inputs = node.inputs || {};
  Object.entries(inputs).forEach(([key, val]) => {
    const row = mkEl('div', 'field-row');
    const lbl = mkEl('label');
    lbl.textContent = key;
    row.appendChild(lbl);

    const isLink = Array.isArray(val) && val.length === 2 && typeof val[0] === 'string';
    if (isLink) {
      const span = mkEl('span', 'link-val');
      span.textContent = `← node[${val[0]}] slot ${val[1]}`;
      row.appendChild(span);
    } else {
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.dataset.key = key;
      inp.value = typeof val === 'object' ? JSON.stringify(val) : String(val ?? '');
      row.appendChild(inp);
    }
    modalBody.appendChild(row);
  });

  nodeModal.classList.remove('hidden');
}

$('modal-close').onclick = $('modal-cancel').onclick = () => nodeModal.classList.add('hidden');

$('modal-save').onclick = async () => {
  const updates = {};
  modalBody.querySelectorAll('input[data-key]').forEach(inp => {
    let v = inp.value;
    try { v = JSON.parse(v); } catch {}
    updates[inp.dataset.key] = v;
  });
  const res = await apiFetch('/api/workflow/node', 'PATCH', { node_id: currentNodeId, inputs: updates });
  if (res.nodes) renderWorkflow(res.nodes);
  nodeModal.classList.add('hidden');
};

$('modal-delete').onclick = async () => {
  if (!confirm(`删除节点 [${currentNodeId}]？`)) return;
  const res = await apiFetch(`/api/workflow/node/${currentNodeId}`, 'DELETE');
  if (res.nodes) renderWorkflow(res.nodes);
  nodeModal.classList.add('hidden');
};

// ─── Exec & Save buttons ─────────────────────────────────────────────────────
$('exec-btn').onclick = async () => {
  const res = await apiFetch('/api/workflow/execute', 'POST');
  appendSystemMsg(res.error ? `执行失败: ${res.error}` : `已提交 ComfyUI，prompt_id: ${res.prompt_id}`);
};

$('save-wf-btn').onclick = async () => {
  const name = prompt('工作流名称（可留空）：') ?? '';
  const res = await apiFetch('/api/workflow/save', 'POST', { name });
  appendSystemMsg(`已保存: ${res.saved}`);
};

// ─── Resources panel ─────────────────────────────────────────────────────────
async function loadResources() {
  const res = await apiFetch('/api/resources');
  renderResources(res.markdown || '');
}

$('refresh-res-btn').onclick = async () => {
  $('refresh-res-btn').textContent = '…';
  const res = await apiFetch('/api/resources/refresh', 'POST');
  renderResources(res.markdown || '');
  $('refresh-res-btn').textContent = '↺';
};

function renderResources(md) {
  // Simple parse: lines starting with ### = section, - items = list items
  const lines = md.split('\n');
  let html = '';
  lines.forEach(line => {
    if (line.startsWith('### '))    html += `<h3>${line.slice(4)}</h3>`;
    else if (line.startsWith('## ')) html += `<h3 style="font-size:13px;color:var(--accent)">${line.slice(3)}</h3>`;
    else if (line.startsWith('- ')) html += `<li>${line.slice(2)}</li>`;
    else if (line.trim()) html += `<p style="font-size:11px;color:var(--muted);padding:2px 0">${line}</p>`;
  });
  resourcesDiv.innerHTML = '<ul>' + html + '</ul>';
}

// ─── Settings drawer ─────────────────────────────────────────────────────────
const drawer = $('settings-drawer');

$('settings-btn').onclick = async () => {
  drawer.classList.toggle('hidden');
  if (!drawer.classList.contains('hidden')) {
    const s = await apiFetch('/api/settings');
    $('s-provider').value = s.provider || 'gemini';
    $('s-model').value    = s.model || '';
    $('s-base-url').value = s.base_url || '';
  }
};

$('close-settings-btn').onclick = () => drawer.classList.add('hidden');

$('save-settings-btn').onclick = async () => {
  const payload = {
    provider: $('s-provider').value,
    model:    $('s-model').value.trim(),
    base_url: $('s-base-url').value.trim(),
  };
  const key = $('s-api-key').value.trim();
  if (key) payload.api_key = key;

  const res = await apiFetch('/api/settings', 'POST', payload);
  if (res.ok) {
    providerLabel.textContent = res.provider;
    $('settings-msg').textContent = '✓ 已保存';
    setTimeout(() => { $('settings-msg').textContent = ''; }, 2000);
  }
};

// ─── Status polling ───────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const s = await apiFetch('/api/status');
    setBadge(comfyBadge, s.comfyui, `● ComfyUI`);
    providerLabel.textContent = `${s.provider}${s.model ? ' / ' + s.model : ''}`;
  } catch {}
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function mkEl(tag, cls = '') {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  return el;
}

function setBadge(el, state, text) {
  el.dataset.state = state;
  el.textContent = text;
}

function scrollBottom() {
  msgList.scrollTop = msgList.scrollHeight;
}

async function apiFetch(path, method = 'GET', body = null) {
  const opts = { method, headers: {} };
  if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  try {
    const r = await fetch(path, opts);
    return await r.json();
  } catch (e) {
    return { error: String(e) };
  }
}

// ─── Init ────────────────────────────────────────────────────────────────────
(async () => {
  connectWS();
  await pollStatus();
  await loadResources();
  // load current workflow if any
  const wf = await apiFetch('/api/workflow');
  if (wf.nodes && wf.nodes.length) renderWorkflow(wf.nodes);
  // poll status every 10s
  setInterval(pollStatus, 10000);
  // welcome
  appendSystemMsg('Agent 已就绪。用自然语言描述你想要的工作流，或输入操作指令。');
})();
