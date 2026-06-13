const { contextBridge, ipcRenderer, webUtils } = require('electron')

contextBridge.exposeInMainWorld('hadesDesktop', {
  getConnection: profile => ipcRenderer.invoke('hades:connection', profile),
  revalidateConnection: () => ipcRenderer.invoke('hades:connection:revalidate'),
  touchBackend: profile => ipcRenderer.invoke('hades:backend:touch', profile),
  getGatewayWsUrl: profile => ipcRenderer.invoke('hades:gateway:ws-url', profile),
  openSessionWindow: sessionId => ipcRenderer.invoke('hades:window:openSession', sessionId),
  getBootProgress: () => ipcRenderer.invoke('hades:boot-progress:get'),
  getConnectionConfig: profile => ipcRenderer.invoke('hades:connection-config:get', profile),
  saveConnectionConfig: payload => ipcRenderer.invoke('hades:connection-config:save', payload),
  applyConnectionConfig: payload => ipcRenderer.invoke('hades:connection-config:apply', payload),
  testConnectionConfig: payload => ipcRenderer.invoke('hades:connection-config:test', payload),
  probeConnectionConfig: remoteUrl => ipcRenderer.invoke('hades:connection-config:probe', remoteUrl),
  oauthLoginConnectionConfig: remoteUrl => ipcRenderer.invoke('hades:connection-config:oauth-login', remoteUrl),
  oauthLogoutConnectionConfig: remoteUrl => ipcRenderer.invoke('hades:connection-config:oauth-logout', remoteUrl),
  profile: {
    get: () => ipcRenderer.invoke('hades:profile:get'),
    set: name => ipcRenderer.invoke('hades:profile:set', name)
  },
  api: request => ipcRenderer.invoke('hades:api', request),
  notify: payload => ipcRenderer.invoke('hades:notify', payload),
  requestMicrophoneAccess: () => ipcRenderer.invoke('hades:requestMicrophoneAccess'),
  readFileDataUrl: filePath => ipcRenderer.invoke('hades:readFileDataUrl', filePath),
  readFileText: filePath => ipcRenderer.invoke('hades:readFileText', filePath),
  selectPaths: options => ipcRenderer.invoke('hades:selectPaths', options),
  writeClipboard: text => ipcRenderer.invoke('hades:writeClipboard', text),
  saveImageFromUrl: url => ipcRenderer.invoke('hades:saveImageFromUrl', url),
  saveImageBuffer: (data, ext) => ipcRenderer.invoke('hades:saveImageBuffer', { data, ext }),
  saveClipboardImage: () => ipcRenderer.invoke('hades:saveClipboardImage'),
  getPathForFile: file => {
    try {
      return webUtils.getPathForFile(file) || ''
    } catch {
      return ''
    }
  },
  normalizePreviewTarget: (target, baseDir) => ipcRenderer.invoke('hades:normalizePreviewTarget', target, baseDir),
  watchPreviewFile: url => ipcRenderer.invoke('hades:watchPreviewFile', url),
  stopPreviewFileWatch: id => ipcRenderer.invoke('hades:stopPreviewFileWatch', id),
  setTitleBarTheme: payload => ipcRenderer.send('hades:titlebar-theme', payload),
  setPreviewShortcutActive: active => ipcRenderer.send('hades:previewShortcutActive', Boolean(active)),
  openExternal: url => ipcRenderer.invoke('hades:openExternal', url),
  fetchLinkTitle: url => ipcRenderer.invoke('hades:fetchLinkTitle', url),
  sanitizeWorkspaceCwd: cwd => ipcRenderer.invoke('hades:workspace:sanitize', cwd),
  settings: {
    getDefaultProjectDir: () => ipcRenderer.invoke('hades:setting:defaultProjectDir:get'),
    setDefaultProjectDir: dir => ipcRenderer.invoke('hades:setting:defaultProjectDir:set', dir),
    pickDefaultProjectDir: () => ipcRenderer.invoke('hades:setting:defaultProjectDir:pick')
  },
  revealLogs: () => ipcRenderer.invoke('hades:logs:reveal'),
  getRecentLogs: () => ipcRenderer.invoke('hades:logs:recent'),
  readDir: dirPath => ipcRenderer.invoke('hades:fs:readDir', dirPath),
  gitRoot: startPath => ipcRenderer.invoke('hades:fs:gitRoot', startPath),
  terminal: {
    dispose: id => ipcRenderer.invoke('hades:terminal:dispose', id),
    resize: (id, size) => ipcRenderer.invoke('hades:terminal:resize', id, size),
    start: options => ipcRenderer.invoke('hades:terminal:start', options),
    write: (id, data) => ipcRenderer.invoke('hades:terminal:write', id, data),
    onData: (id, callback) => {
      const channel = `hades:terminal:${id}:data`
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on(channel, listener)
      return () => ipcRenderer.removeListener(channel, listener)
    },
    onExit: (id, callback) => {
      const channel = `hades:terminal:${id}:exit`
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on(channel, listener)
      return () => ipcRenderer.removeListener(channel, listener)
    }
  },
  onClosePreviewRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('hades:close-preview-requested', listener)
    return () => ipcRenderer.removeListener('hades:close-preview-requested', listener)
  },
  onOpenUpdatesRequested: callback => {
    const listener = () => callback()
    ipcRenderer.on('hades:open-updates', listener)
    return () => ipcRenderer.removeListener('hades:open-updates', listener)
  },
  onWindowStateChanged: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hades:window-state-changed', listener)
    return () => ipcRenderer.removeListener('hades:window-state-changed', listener)
  },
  onPreviewFileChanged: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hades:preview-file-changed', listener)
    return () => ipcRenderer.removeListener('hades:preview-file-changed', listener)
  },
  onBackendExit: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hades:backend-exit', listener)
    return () => ipcRenderer.removeListener('hades:backend-exit', listener)
  },
  onPowerResume: callback => {
    const listener = () => callback()
    ipcRenderer.on('hades:power-resume', listener)
    return () => ipcRenderer.removeListener('hades:power-resume', listener)
  },
  onBootProgress: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hades:boot-progress', listener)
    return () => ipcRenderer.removeListener('hades:boot-progress', listener)
  },
  // First-launch bootstrap progress -- emitted by the install.ps1 stage
  // runner in main.cjs (apps/desktop/electron/bootstrap-runner.cjs).
  // Renderer's install overlay subscribes to live events and queries the
  // current snapshot via getBootstrapState() to recover after a devtools
  // reload mid-bootstrap.
  getBootstrapState: () => ipcRenderer.invoke('hades:bootstrap:get'),
  resetBootstrap: () => ipcRenderer.invoke('hades:bootstrap:reset'),
  repairBootstrap: () => ipcRenderer.invoke('hades:bootstrap:repair'),
  cancelBootstrap: () => ipcRenderer.invoke('hades:bootstrap:cancel'),
  onBootstrapEvent: callback => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('hades:bootstrap:event', listener)
    return () => ipcRenderer.removeListener('hades:bootstrap:event', listener)
  },
  getVersion: () => ipcRenderer.invoke('hades:version'),
  uninstall: {
    summary: () => ipcRenderer.invoke('hades:uninstall:summary'),
    run: mode => ipcRenderer.invoke('hades:uninstall:run', { mode })
  },
  updates: {
    check: () => ipcRenderer.invoke('hades:updates:check'),
    apply: opts => ipcRenderer.invoke('hades:updates:apply', opts),
    getBranch: () => ipcRenderer.invoke('hades:updates:branch:get'),
    setBranch: name => ipcRenderer.invoke('hades:updates:branch:set', name),
    onProgress: callback => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('hades:updates:progress', listener)
      return () => ipcRenderer.removeListener('hades:updates:progress', listener)
    }
  },
  themes: {
    fetchMarketplace: id => ipcRenderer.invoke('hades:vscode-theme:fetch', id),
    searchMarketplace: query => ipcRenderer.invoke('hades:vscode-theme:search', query)
  }
})
