(function () {
  'use strict';

  const EXTENSION_ID = 'image.layerdiffuse';
  const PANEL_ID = 'layerdiffuse-extension-panel';
  const SLOT_SELECTORS = [
    '#neo-ext-slot-image-extensions-manager',
    '[data-neo-extension-slot="image.extensions.manager"]',
    '[data-neo-extension-slot="image.panel"]',
    '[data-neo-surface="image"][data-neo-extension-mount="panel"]'
  ];

  const DEFAULT_STATE = Object.freeze({
    enabled: false,
    mode: 'transparent_asset',
    source_type: 'prompt',
    source_image_id: null,
    background_image_id: null,
    foreground_image_id: null,
    decode_mode: 'rgba',
    output_policy: 'new_run',
    replace_target_id: null,
    replace_confirmed: false,
    save_rgba: true,
    save_rgb: false,
    save_alpha: true,
    save_metadata: true,
    compatibility_mode: 'auto'
  });

  const MODES = [
    ['transparent_asset', 'Transparent Asset', 'Prompt → native transparent PNG asset.'],
    ['rgb_alpha_split', 'RGB + Alpha Split', 'Prompt → RGBA, RGB, and alpha mask bundle.'],
    ['foreground_on_background', 'Foreground on Background', 'Prompt + background image → foreground designed for that scene.'],
    ['background_aware_blend', 'Background-Aware Blend', 'Foreground + background → blended result.'],
    ['extract_foreground', 'Extract Foreground', 'Composite/source image + known background → extracted layer.'],
    ['overlay_fx', 'Transparent Overlay FX', 'Prompt → smoke, glow, energy, glass, rain, HUD, and other overlay assets.']
  ];

  const MODE_REQUIREMENTS = Object.freeze({
    transparent_asset: { prompt: true },
    rgb_alpha_split: { prompt: true },
    foreground_on_background: { prompt: true, background_image_id: true },
    background_aware_blend: { prompt: true, background_image_id: true, foreground_image_id: true },
    extract_foreground: { source_image_id: true, background_image_id: true },
    overlay_fx: { prompt: true }
  });

  const DECODE_OPTIONS = [
    ['rgba', 'RGBA PNG'],
    ['split', 'RGB + Alpha Split'],
    ['preview_only', 'Preview Only']
  ];

  const OUTPUT_POLICIES = [
    ['preview', 'Preview only'],
    ['new_run', 'New run'],
    ['append', 'Append to assets'],
    ['replace', 'Replace selected target']
  ];

  const SOURCE_TYPES = [
    ['prompt', 'Prompt only'],
    ['selected_image', 'Selected image'],
    ['upload', 'Uploaded image'],
    ['previous_output', 'Previous output']
  ];

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function cleanText(value) {
    return String(value == null ? '' : value).trim();
  }

  function getStore() {
    return window.NeoExternalExtensionState || null;
  }

  function getSnapshot() {
    return getStore()?.getSnapshot?.() || {};
  }

  function getRawState() {
    const snapshot = getSnapshot();
    return Object.assign({}, DEFAULT_STATE, snapshot.raw?.[EXTENSION_ID] || {});
  }

  function getEffectiveState() {
    const snapshot = getSnapshot();
    return snapshot.effective?.[EXTENSION_ID] || {};
  }

  function setRaw(partial) {
    const store = getStore();
    if (!store) return;
    const next = Object.assign({}, getRawState(), partial || {});
    if (Object.prototype.hasOwnProperty.call(partial || {}, 'enabled')) {
      store.setEnabled?.(EXTENSION_ID, !!partial.enabled);
    }
    store.setRawState?.(EXTENSION_ID, next);
  }

  function optionList(options, selected) {
    return options.map((item) => {
      const value = Array.isArray(item) ? item[0] : item;
      const label = Array.isArray(item) ? item[1] : item;
      return `<option value="${escapeHtml(value)}" ${value === selected ? 'selected' : ''}>${escapeHtml(label)}</option>`;
    }).join('');
  }

  function field(label, control, note) {
    return `<label class="ld-field"><span>${escapeHtml(label)}</span>${control}${note ? `<small>${escapeHtml(note)}</small>` : ''}</label>`;
  }

  function checkbox(key, label, checked) {
    return `<label class="ld-check"><input type="checkbox" data-ld-key="${escapeHtml(key)}" ${checked ? 'checked' : ''}> <span>${escapeHtml(label)}</span></label>`;
  }

  function visibleRequirementBadges(raw) {
    const req = MODE_REQUIREMENTS[raw.mode] || {};
    const badges = [];
    if (req.prompt) badges.push('Prompt required');
    if (req.source_image_id) badges.push('Source image required');
    if (req.background_image_id) badges.push('Background required');
    if (req.foreground_image_id) badges.push('Foreground required');
    badges.push('Batch force 1');
    return badges.map(label => `<span class="ld-pill">${escapeHtml(label)}</span>`).join('');
  }

  function buildLocalWarnings(raw, effective) {
    const warnings = [];
    const req = MODE_REQUIREMENTS[raw.mode] || {};
    if (req.source_image_id && !raw.source_image_id) warnings.push('Source image is required for this mode.');
    if (req.background_image_id && !raw.background_image_id) warnings.push('Background image is required for this mode.');
    if (req.foreground_image_id && !raw.foreground_image_id) warnings.push('Foreground image is required for this mode.');
    if (raw.output_policy === 'replace' && !raw.replace_target_id) warnings.push('Replace output policy requires a selected target.');
    if (raw.output_policy === 'replace' && !raw.replace_confirmed) warnings.push('Replace output policy requires visible confirmation.');
    if (raw.mode === 'rgb_alpha_split' && raw.decode_mode !== 'split') warnings.push('RGB + Alpha Split should use split decode mode.');
    if (raw.mode === 'background_aware_blend' && raw.decode_mode !== 'preview_only') warnings.push('Background-aware blend usually resolves as preview-only/blended output.');
    (effective.warnings || []).forEach(warning => warnings.push(String(warning)));
    if (effective.disabled_reason) warnings.unshift(String(effective.disabled_reason));
    return [...new Set(warnings.filter(Boolean))];
  }

  function findHost() {
    for (const selector of SLOT_SELECTORS) {
      const node = document.querySelector(selector);
      if (node) return node;
    }
    let fallback = document.getElementById('neo-extension-image-panel-slot');
    if (!fallback) {
      fallback = document.createElement('section');
      fallback.id = 'neo-extension-image-panel-slot';
      fallback.className = 'neo-extension-slot neo-extension-slot--fallback';
      fallback.dataset.neoExtensionSlot = 'image.extensions.manager';
      document.body.appendChild(fallback);
    }
    return fallback;
  }

  function render() {
    const host = findHost();
    if (!host) return;
    let panel = document.getElementById(PANEL_ID);
    if (!panel) {
      panel = document.createElement('section');
      panel.id = PANEL_ID;
      panel.className = 'layerdiffuse-panel neo-extension-panel';
      panel.dataset.extensionId = EXTENSION_ID;
      host.insertBefore(panel, host.firstChild || null);
    }

    const raw = getRawState();
    const effective = getEffectiveState();
    const warnings = buildLocalWarnings(raw, effective);
    const active = !!effective.effective_enabled;
    const enabled = !!raw.enabled;
    const selectedMode = MODES.find(item => item[0] === raw.mode) || MODES[0];
    const showSource = ['selected_image', 'upload', 'previous_output'].includes(raw.source_type) || raw.mode === 'extract_foreground';
    const showBackground = ['foreground_on_background', 'background_aware_blend', 'extract_foreground'].includes(raw.mode);
    const showForeground = raw.mode === 'background_aware_blend';
    const showReplace = raw.output_policy === 'replace';

    panel.innerHTML = `
      <header class="ld-header">
        <div>
          <strong>LayerDiffuse</strong>
          <div class="ld-muted">Transparent assets, alpha split, and layer compositing via external ComfyUI nodes.</div>
        </div>
        <div class="ld-status-stack">
          <span class="ld-badge ${active ? 'is-active' : ''}">${active ? 'Active' : (enabled ? 'Blocked' : 'Disabled')}</span>
          <span class="ld-badge">${escapeHtml(effective.batch_policy || 'force_1')}</span>
        </div>
      </header>

      <div class="ld-section">
        <div class="ld-section-title">Status</div>
        ${checkbox('enabled', 'Enable LayerDiffuse', enabled)}
        <div class="ld-visible-state">
          <div><b>Mode:</b> ${escapeHtml(selectedMode[1])}</div>
          <div><b>Target:</b> ${escapeHtml(raw.output_policy)}${raw.replace_target_id ? ` → ${escapeHtml(raw.replace_target_id)}` : ''}</div>
          <div><b>Source:</b> ${escapeHtml(raw.source_type)}</div>
        </div>
        <div class="ld-pills">${visibleRequirementBadges(raw)}</div>
      </div>

      <div class="ld-grid">
        <div class="ld-section">
          <div class="ld-section-title">Mode</div>
          ${field('LayerDiffuse mode', `<select data-ld-key="mode">${optionList(MODES, raw.mode)}</select>`, selectedMode[2])}
          ${field('Compatibility mode', `<select data-ld-key="compatibility_mode">${optionList([['auto','Auto'],['sdxl','Force SDXL'],['sd15','Force SD 1.5']], raw.compatibility_mode)}</select>`, 'Controls which template family the adapter resolves.')}
        </div>

        <div class="ld-section">
          <div class="ld-section-title">Source</div>
          ${field('Source type', `<select data-ld-key="source_type">${optionList(SOURCE_TYPES, raw.source_type)}</select>`, 'All image sources must be visible; no hidden source mutation.')}
          ${showSource ? field('Source image/output ID', `<input data-ld-key="source_image_id" value="${escapeHtml(raw.source_image_id || '')}" placeholder="selected/upload/output id">`, 'Required for extract foreground; optional for selected-image modes.') : ''}
          ${showBackground ? field('Background image ID', `<input data-ld-key="background_image_id" value="${escapeHtml(raw.background_image_id || '')}" placeholder="background image id">`, 'Required for background-aware modes.') : ''}
          ${showForeground ? field('Foreground image ID', `<input data-ld-key="foreground_image_id" value="${escapeHtml(raw.foreground_image_id || '')}" placeholder="foreground image id">`, 'Required for background-aware blend.') : ''}
        </div>
      </div>

      <div class="ld-grid">
        <div class="ld-section">
          <div class="ld-section-title">Output</div>
          ${field('Decode mode', `<select data-ld-key="decode_mode">${optionList(DECODE_OPTIONS, raw.decode_mode)}</select>`, 'RGBA for transparent PNG, split for RGB + alpha mask, preview only for non-asset blend checks.')}
          ${field('Output policy', `<select data-ld-key="output_policy">${optionList(OUTPUT_POLICIES, raw.output_policy)}</select>`, 'Preview/new run/append/replace are explicit and metadata-visible.')}
          ${showReplace ? field('Replace target ID', `<input data-ld-key="replace_target_id" value="${escapeHtml(raw.replace_target_id || '')}" placeholder="target output id">`, 'Required before replace can validate.') : ''}
          ${showReplace ? checkbox('replace_confirmed', 'I confirm this may replace the selected target', !!raw.replace_confirmed) : ''}
        </div>

        <div class="ld-section">
          <div class="ld-section-title">Save Bundle</div>
          ${checkbox('save_rgba', 'Save RGBA PNG', !!raw.save_rgba)}
          ${checkbox('save_rgb', 'Save RGB image', !!raw.save_rgb)}
          ${checkbox('save_alpha', 'Save alpha mask', !!raw.save_alpha)}
          ${checkbox('save_metadata', 'Save metadata', !!raw.save_metadata)}
          <div class="ld-muted">These only describe desired outputs. Workflow graph execution is still controlled by the adapter/template phases.</div>
        </div>
      </div>

      <div class="ld-section">
        <div class="ld-section-title">Validation</div>
        ${warnings.length ? `<div class="ld-warnings">${warnings.map(item => `<div>⚠ ${escapeHtml(item)}</div>`).join('')}</div>` : '<div class="ld-ok">No local UI blockers detected. Server/template validation may still block missing ComfyUI nodes.</div>'}
        <details class="ld-details">
          <summary>Raw vs effective state</summary>
          <pre>${escapeHtml(JSON.stringify({ raw_state: raw, effective_state: effective.effective_state || effective }, null, 2))}</pre>
        </details>
      </div>
    `;
  }

  function readValue(el) {
    if (!el) return null;
    if (el.type === 'checkbox') return !!el.checked;
    const value = cleanText(el.value);
    return value || null;
  }

  function bindEvents() {
    document.addEventListener('change', (event) => {
      const el = event.target;
      if (!el || !el.matches || !el.matches(`#${PANEL_ID} [data-ld-key]`)) return;
      const key = el.getAttribute('data-ld-key');
      const value = readValue(el);
      const patch = { [key]: value };
      if (key === 'mode') {
        if (value === 'rgb_alpha_split' || value === 'overlay_fx' || value === 'extract_foreground') patch.decode_mode = 'split';
        if (value === 'background_aware_blend') patch.decode_mode = 'preview_only';
      }
      if (key === 'output_policy' && value !== 'replace') {
        patch.replace_target_id = null;
        patch.replace_confirmed = false;
      }
      setRaw(patch);
      window.setTimeout(render, 0);
    });
    document.addEventListener('input', (event) => {
      const el = event.target;
      if (!el || !el.matches || !el.matches(`#${PANEL_ID} input[data-ld-key]`)) return;
      if (el.type === 'checkbox') return;
      setRaw({ [el.getAttribute('data-ld-key')]: readValue(el) });
    });
    window.addEventListener('neo:external-extensions:state-changed', () => window.setTimeout(render, 0));
    window.addEventListener('neo:external-extensions:validated', () => window.setTimeout(render, 0));
    window.addEventListener('neo:external-extensions:registry-refreshed', () => window.setTimeout(render, 0));
  }

  function boot() {
    bindEvents();
    const store = getStore();
    store?.refreshRegistry?.();
    window.setTimeout(render, 0);
    window.setTimeout(render, 250);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, { once: true });
  else boot();
})();
