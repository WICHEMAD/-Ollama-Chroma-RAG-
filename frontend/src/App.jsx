import { useState, useCallback } from 'react'
import StatsBar from './components/StatsBar.jsx'
import UploadPanel from './components/UploadPanel.jsx'
import Chat from './components/Chat.jsx'
import ModelSelector from './components/ModelSelector.jsx'
import ConfigModal from './components/ConfigModal.jsx'
import DocumentsPage from './components/DocumentsPage.jsx'

const MODEL_KEY = 'rag_chat_model'
const MODE_KEY = 'rag_chat_mode'
const CONFIG_KEY = 'rag_config_done'

export default function App() {
  // 上传成功后 +1，触发统计栏重新拉取
  const [statsVersion, setStatsVersion] = useState(0)
  const refreshStats = useCallback(() => setStatsVersion((v) => v + 1), [])

  // 当前选中的对话模型（云端模型名，空 = 未选择）
  const [model, setModel] = useState(() => localStorage.getItem(MODEL_KEY) || '')

  // 当前问答模式：'rag'（普通 RAG）| 'agent'（Agent 智能问答）
  const [mode, setMode] = useState(() => localStorage.getItem(MODE_KEY) || 'rag')

  // 主区域视图：'chat'（问答）| 'docs'（知识库文档列表）
  const [view, setView] = useState('chat')

  // 是否显示配置弹窗：首次未配置（无 rag_config_done 标记）时自动弹出；
  // 之后可通过「修改配置」按钮再次打开。
  const [showConfig, setShowConfig] = useState(() => localStorage.getItem(CONFIG_KEY) !== '1')

  function handleModelChange(name) {
    setModel(name || '')
    if (name) localStorage.setItem(MODEL_KEY, name)
    else localStorage.removeItem(MODEL_KEY)
  }

  function handleModeChange(m) {
    setMode(m)
    localStorage.setItem(MODE_KEY, m)
  }

  function handleConfigDone() {
    localStorage.setItem(CONFIG_KEY, '1')
    setShowConfig(false)
  }

  function openConfig() {
    setShowConfig(true)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>📚 Ollama 知识库问答</h1>
        <div className="header-actions">
          <StatsBar version={statsVersion} />
          <button
            className="btn btn-ghost"
            onClick={() => setView(view === 'chat' ? 'docs' : 'chat')}
          >
            {view === 'chat' ? '📄 文档库' : '💬 问答'}
          </button>
          <button className="btn btn-ghost" onClick={openConfig}>
            修改配置
          </button>
        </div>
      </header>

      <aside className="app-sidebar">
        <ModelSelector
          value={model}
          onChange={handleModelChange}
          mode={mode}
          onModeChange={handleModeChange}
        />
        <UploadPanel onUploaded={refreshStats} />
      </aside>

      <main className="app-main">
        {view === 'chat' ? <Chat model={model} mode={mode} /> : <DocumentsPage />}
      </main>

      {showConfig && <ConfigModal onAdded={handleModelChange} onDone={handleConfigDone} />}
    </div>
  )
}
